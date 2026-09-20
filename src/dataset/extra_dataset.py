"""Manually collect a separate, resumable pool, with a cumulative download cap.

No production artifacts are copied or changed. Raw Parquet payloads are downloaded
without a shared Hugging Face cache; cleaned data and SQLite storage are separate
from the download budget. No split assignment or tokenization is performed.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sqlite3

try:
    import clean_dataset as c
except ModuleNotFoundError:
    from src.dataset import clean_dataset as c

GB = 1_000_000_000
DEFAULT_OUTPUT = c.PROJECT_ROOT/'artifacts/corpora/extra'
DEFAULT_EXCLUDE = c.PROJECT_ROOT/'artifacts/corpora/italian_strict_v2'


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.new')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def discover(source, seed):
    """Pin a repository revision and persist its ordered train-shard catalogue."""
    from huggingface_hub import HfApi
    api = HfApi()
    revision = api.dataset_info(source.dataset, revision=source.revision).sha
    prefix = f'{source.config}/' if source.key == 'wiki' else f'data/{source.config}/train/'
    files = [dict(path=f.rfilename, size=f.size,
                  upstream_sha256=getattr(getattr(f,'lfs',None),'sha256',None)) for f in
             api.list_repo_tree(source.dataset, path_in_repo=prefix.rstrip('/'),
                                repo_type='dataset', revision=revision, recursive=True)
             if getattr(f, 'rfilename', '').endswith('.parquet') and
             (source.key != 'wiki' or Path(f.rfilename).name.startswith('train-'))]
    if not files or any(not isinstance(f['size'], int) or f['size'] <= 0 for f in files):
        raise ValueError(f'No sized training Parquet shards found for {source.key}')
    random.Random(seed).shuffle(files)
    return dict(dataset=source.dataset, config=source.config, revision=revision,
                split='train', seed=seed, files=files, next_file=0)


def download_shard(root, state, source, entry):
    """Resume byte ranges. Reserve budget before reads, conservatively on errors."""
    import requests
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers
    identity = hashlib.sha256((source['dataset']+source['revision']+entry['path']).encode()).hexdigest()
    destination = root/'raw'/f'{identity}.parquet'
    partial = destination.with_suffix('.part')
    expected = entry['size']
    if destination.exists():
        if destination.stat().st_size != expected:
            raise ValueError(f'Raw file size changed: {destination}')
        if entry.get('sha256') and c.sha256_file(destination) != entry['sha256']:
            raise ValueError(f'Raw file checksum changed: {destination}')
        return destination
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected:
        raise ValueError('Partial download is larger than its pinned shard')
    remaining = state['download_limit_bytes']-state['download_bytes_reserved']
    if expected-offset > remaining:
        return None
    if shutil.disk_usage(root).free < expected-offset+state['min_free_bytes']:
        raise OSError('Free-space reserve reached; no new shard downloaded')
    if offset < expected:
        headers = build_hf_headers()
        headers['Accept-Encoding'] = 'identity'
        if offset: headers['Range'] = f'bytes={offset}-'
        url = hf_hub_url(source['dataset'], entry['path'], repo_type='dataset', revision=source['revision'])
        with requests.get(url, headers=headers, stream=True, timeout=(30,120)) as response:
            response.raise_for_status()
            if offset and (response.status_code != 206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                raise ValueError('Server did not honor the resume range; partial data preserved')
            with partial.open('ab') as output:
                while offset < expected:
                    allowance = min(1024*1024, expected-offset)
                    # An interrupted read can consume this reservation without saving
                    # bytes, but can never silently reset the lifetime budget.
                    if state['download_bytes_reserved']+allowance > state['download_limit_bytes']:
                        return None
                    state['download_bytes_reserved'] += allowance
                    atomic_json(root/'state.json', state)
                    data = response.raw.read(allowance)
                    if not data: raise OSError('Download ended before the advertised shard size')
                    output.write(data); output.flush()
                    offset += len(data)
                    # Keep unused reservation after a short read: conservative accounting.
    c.pq.ParquetFile(partial)  # Require a complete readable Parquet footer.
    if entry.get('upstream_sha256') and c.sha256_file(partial) != entry['upstream_sha256']:
        raise ValueError('Downloaded shard does not match its upstream checksum')
    partial.replace(destination)
    return destination


def baseline_duplicate(db, key, doc_id, text, chunks):
    if db.execute('SELECT 1 FROM documents WHERE source=? AND id=?', (key,doc_id)).fetchone():
        return 'existing_pool_document_id'
    if db.execute('SELECT 1 FROM documents WHERE hash=?', (c.normalized_text_hash(text),)).fetchone():
        return 'existing_pool_document'
    if any(db.execute('SELECT 1 FROM chunks WHERE hash=?',(c.normalized_text_hash(t),)).fetchone() for t in chunks):
        return 'existing_pool_chunk'
    return None


def process_shard(root, state, key, entry, path, dedup, baseline, predict):
    """Commit accepted text, its decision and input cursor in one transaction."""
    shard_id = path.stem
    db = dedup.db
    done = db.execute('SELECT rows_done FROM progress WHERE shard=?',(shard_id,)).fetchone()
    rows_done = done[0] if done else 0
    parquet = c.pq.ParquetFile(path)
    position = 0
    for batch in parquet.iter_batches(batch_size=32):
        for row in batch.to_pylist():
            position += 1
            if position <= rows_done: continue
            doc_id = c.stable_document_id(c.SOURCES[key], row)
            text = row.get('text') or ''
            # Avoid model work for previously accepted canonical IDs.
            duplicate = baseline.execute('SELECT 1 FROM documents WHERE source=? AND id=?', (key,doc_id)).fetchone()
            if duplicate:
                chunks, signals = [], dict(reasons=['existing_pool_document_id'])
            else:
                chunks, signals = c.clean_source_document(c.SOURCES[key],row,predict,.95)
            kept = False
            doc_hash = c.normalized_text_hash(text)
            with db:
                if chunks:
                    reason = baseline_duplicate(baseline,key,doc_id,text,chunks)
                    if reason is None:
                        kept,reason,doc_hash = dedup.add(key,doc_id,text,chunks,commit=False)
                    if reason: signals['reasons'].append(reason)
                db.execute('INSERT INTO decisions VALUES (?,?,?,?,?,?)',
                           (shard_id,position,key,doc_hash,int(kept),json.dumps(dict(document_id=doc_id,**signals))))
                db.execute('INSERT OR REPLACE INTO progress VALUES (?,?)',(shard_id,position))
            if position % 1000 == 0:
                print(f'Extra {key}: {position:,}/{parquet.metadata.num_rows:,} documents checked',flush=True)
    stage = root/'clean'/key/f'.{shard_id}.tmp'
    if stage.exists(): shutil.rmtree(stage)  # Regenerable unpublished export only.
    writer = c.MaterializedParquetWriter(stage,c.DEFAULT_DOCUMENT_SHARD_BYTES)
    try:
        query = 'SELECT d.id,d.chunks FROM documents d JOIN decisions e ON d.hash=e.hash WHERE e.shard=? AND e.accepted=1 ORDER BY e.row_number'
        for doc_id,chunks in db.execute(query,(shard_id,)):
            for index,text in enumerate(json.loads(chunks)):
                writer.write(dict(document_id=doc_id,chunk_index=index,source=key,
                                  text_sha256=c.normalized_text_hash(text),text=text),len(text.encode()))
    finally:
        writer.close()
    directory = stage.with_name(shard_id)
    if directory.exists():
        # Crash after publication but before state update: compare before reuse.
        old = json.loads((directory/'manifest.json').read_text())
        if old['files'] != writer.files: raise ValueError('Existing shard export differs on resume')
        if any(c.sha256_file(directory/f['file']) != f['sha256'] for f in old['files']):
            raise ValueError('Published extra shard checksum mismatch')
        shutil.rmtree(stage)
        return old
    counts = dict(db.execute('SELECT accepted,COUNT(*) FROM decisions WHERE shard=? GROUP BY accepted',(shard_id,)))
    result = dict(source=key,raw_file=str(path.relative_to(root)),raw_sha256=c.sha256_file(path),
                  documents_seen=position,accepted_documents=counts.get(1,0),rejected_documents=counts.get(0,0),
                  text_bytes=writer.total_text_bytes,chunks=writer.total_chunks,files=writer.files,
                  directory=str(directory.relative_to(root)))
    atomic_json(stage/'manifest.json',result)
    stage.rename(directory)
    return result


def run(args, *, catalog_loader=discover, downloader=download_shard, predictor=None):
    root, exclude = args.output_dir.resolve(), args.exclude_pool.resolve()
    if args.status:
        path=root/'state.json'
        if not path.exists():
            print('{"status":"not_initialized"}')
            return
        state=json.loads(path.read_text())
        summary=dict(status=state['status'],active_source=state.get('active_source'),
                     download_bytes_reserved=state['download_bytes_reserved'],
                     remaining_download_bytes=state['download_limit_bytes']-state['download_bytes_reserved'],
                     completed_batches=len(state['batches']),
                     accepted_in_completed_batches=sum(b['accepted_documents'] for b in state['batches']),
                     clean_text_bytes=sum(b['text_bytes'] for b in state['batches']))
        if state.get('active_shard') and (root/'dedup.sqlite').exists():
            db=sqlite3.connect(f'file:{root / "dedup.sqlite"}?mode=ro',uri=True)
            try:
                row=db.execute('SELECT rows_done FROM progress WHERE shard=?',(state['active_shard'],)).fetchone()
                summary['active_shard_documents_checked']=row[0] if row else 0
            finally:db.close()
        if state.get('error'):summary['error']=state['error']
        print(json.dumps(summary,indent=2))
        return
    if root==exclude or root in exclude.parents or exclude in root.parents:
        raise ValueError('Extra output must be separate from the existing pool')
    if any((parent/'manifest.json').exists() for parent in root.parents):
        raise ValueError('Extra output cannot be nested inside an existing corpus')
    if root.exists() and not (root/'state.json').exists() and (
            {p.name for p in root.iterdir()}-{'run.lock','state.json.new'}):
        raise ValueError('Refusing to use a nonempty directory that is not an extra pool')
    if not math.isfinite(args.max_download_gb) or not 0 < args.max_download_gb <= 20:
        raise ValueError('Download limit must be positive and at most 20 decimal GB')
    if args.max_shards < 1 or not math.isfinite(args.min_free_gb) or args.min_free_gb < 0:
        raise ValueError('Invalid batch or free-space limit')
    model_dir=args.model_dir.resolve()
    if not (model_dir/'model.json').is_file():
        raise FileNotFoundError('Prepare the shared GlotLID model first; extra collection does not download models')
    baseline_manifest=exclude/'manifest.json'
    if json.loads(baseline_manifest.read_text()).get('format_version') != 2 or not (exclude/'dedup.sqlite').is_file():
        raise ValueError('Exclusion reference must be a completed format-2 pool with a duplicate index')
    baseline_hash=c.sha256_file(baseline_manifest)
    stat=(exclude/'dedup.sqlite').stat()
    baseline_index=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    root.mkdir(parents=True,exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another extra-pool collector is already running')
        state_path=root/'state.json'
        policy={name:c.sha256_file(Path(c.__file__).with_name(name)) for name in
                ('clean_dataset.py','language_quality.py','prose_quality.py','extra_dataset.py')}
        model=json.loads((model_dir/'model.json').read_text())
        if state_path.exists():
            state=json.loads(state_path.read_text())
            if state['exclude_manifest_sha256']!=baseline_hash or state['exclude_pool']!=str(exclude):
                raise ValueError('Existing-pool reference changed; use a new extra output directory')
            if state['exclude_index']!=baseline_index:
                raise ValueError('Existing-pool index changed; use a new extra output directory')
            if state['policy_sha256']!=policy or state['model_sha256']!=model['sha256']:
                raise ValueError('Cleaning implementation or model changed; use a new extra output directory')
            if state['download_limit_bytes']!=int(args.max_download_gb*GB):
                raise ValueError('Resume with the original download limit; it never resets per run')
            if state['seed']!=args.seed or state['min_free_bytes']!=int(args.min_free_gb*GB):
                raise ValueError('Resume with the original seed and free-space reserve')
        else:
            state=dict(format_version=1,kind='isolated_extra_pool',status='initialized',
                       download_limit_bytes=int(args.max_download_gb*GB),download_bytes_reserved=0,
                       min_free_bytes=int(args.min_free_gb*GB),exclude_pool=str(exclude),
                       exclude_index=baseline_index,
                       exclude_manifest_sha256=baseline_hash,policy_sha256=policy,model_sha256=model['sha256'],
                       sources={},batches=[],next_source=0,seed=args.seed)
            atomic_json(state_path,state)
        state.pop('error',None)
        (root/'raw').mkdir(exist_ok=True)
        baseline=sqlite3.connect(f'file:{exclude / "dedup.sqlite"}?mode=ro',uri=True)
        dedup=c.GlobalDeduplicator(root/'dedup.sqlite',resume=(root/'dedup.sqlite').exists())
        dedup.db.execute('CREATE TABLE IF NOT EXISTS decisions (shard TEXT,row_number INTEGER,source TEXT,hash TEXT,accepted INTEGER,signals TEXT,PRIMARY KEY(shard,row_number))')
        dedup.db.execute('CREATE TABLE IF NOT EXISTS progress (shard TEXT PRIMARY KEY,rows_done INTEGER)')
        dedup.db.commit()
        try:
            if predictor is None: predictor,_=c.quality_predictor(model_dir)
            processed=0; stalled=0; keys=list(c.SOURCES)
            while processed < args.max_shards and stalled < len(keys):
                key=keys[state['next_source']%len(keys)]
                if key not in state['sources']:
                    state['sources'][key]=catalog_loader(c.SOURCES[key],args.seed+keys.index(key)*1009)
                    atomic_json(state_path,state)
                source=state['sources'][key];offset=source['next_file']
                if offset==len(source['files']):
                    state['next_source']+=1;stalled+=1;continue
                entry=source['files'][offset]
                state['status']='downloading';state['active_source']=key
                state.pop('active_shard',None)
                atomic_json(state_path,state)
                path=downloader(root,state,source,entry)
                if path is None:
                    state['next_source']+=1;stalled+=1;continue
                entry['sha256']=c.sha256_file(path)
                state['status']='cleaning';state['active_shard']=path.stem
                atomic_json(state_path,state)
                result=process_shard(root,state,key,entry,path,dedup,baseline,predictor)
                state['batches'].append(result);source['next_file']+=1;state['next_source']+=1
                processed+=1;stalled=0;state['status']='batch_complete'
                atomic_json(state_path,state)
                print(f"Saved extra {key}: {result['accepted_documents']:,} documents, {result['text_bytes']:,} text bytes",flush=True)
            state['status']='batch_limit_reached' if processed==args.max_shards else 'budget_or_sources_exhausted'
            atomic_json(state_path,state)
            atomic_json(root/'manifest.json',dict(format_version=1,kind='isolated_extra_pool',
                description='Separate source-tagged text; not assigned to production splits or mixed into existing pools',
                exclude_pool=str(exclude),exclude_manifest_sha256=baseline_hash,
                policy_sha256=policy,model_sha256=model['sha256'],sources=state['sources'],batches=state['batches']))
        except BaseException as error:
            state['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
            state['error']=str(error);atomic_json(state_path,state)
            raise
        finally:
            baseline.close();dedup.close()
        print(json.dumps(dict(status=state['status'],download_bytes_reserved=state['download_bytes_reserved'],
                             download_limit_bytes=state['download_limit_bytes'],output=str(root)),indent=2))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--exclude-pool',type=Path,default=DEFAULT_EXCLUDE)
    parser.add_argument('--model-dir',type=Path,default=c.PROJECT_ROOT/'artifacts/models/glotlid')
    parser.add_argument('--max-download-gb',type=float,default=20,help='Cumulative raw payload budget, decimal GB (maximum 20)')
    parser.add_argument('--max-shards',type=int,default=3,help='Maximum raw shards to process per manual invocation')
    parser.add_argument('--min-free-gb',type=float,default=20)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--status',action='store_true')
    run(parser.parse_args(argv))


if __name__=='__main__':
    main()
