"""Reusable accepted-document pools and quota-based training selections.

Pool exports retain all accepted training documents from a closed global index.
Evaluation shards stay frozen. Selection never reruns cleaning or changes the pool.
"""
from collections import Counter
from itertools import groupby
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3

try:
    import clean_dataset as c
except ModuleNotFoundError:
    from src.dataset import clean_dataset as c


def _clone_without_train(parent, output):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if parent == output or parent in output.parents or output in parent.parents:
        raise ValueError('Output must be separate from its parent')
    temporary = output.with_name(f'.{output.name}.pool.tmp')
    if output.exists() or temporary.exists():
        raise FileExistsError(f'Output or incomplete attempt already exists: {output}')
    if any((parent/f'dedup.sqlite{s}').exists() for s in ('-journal', '-wal')):
        raise ValueError('Wait for the parent build to close its index')
    manifest = json.loads((parent/'manifest.json').read_text())
    if manifest.get('format_version') != 2:
        raise ValueError('A format-2 cleaned corpus is required')
    # Independent index: later supplementation must never mutate its parent.
    def clone(src, dst):
        if Path(src).name in ('manifest.json', 'dedup.sqlite'):
            return shutil.copy2(src, dst)
        os.link(src, dst)
        return dst
    shutil.copytree(parent, temporary, copy_function=clone,
                    ignore=lambda path,names: {'train'} if Path(path)==parent else set())
    return parent, output, temporary, manifest


def export_document_pool(documents_dir, output_dir):
    """Export every accepted TRAIN document; preserve existing evaluation shards."""
    parent, output, temporary, manifest = _clone_without_train(documents_dir, output_dir)
    writers = {key:c.MaterializedParquetWriter(temporary/'train'/key,
                    c.DEFAULT_DOCUMENT_SHARD_BYTES) for key in c.SOURCES}
    counts = Counter()
    db = sqlite3.connect(f'file:{temporary / "dedup.sqlite"}?mode=ro', uri=True)
    split = manifest['partitioning']
    if split['validation_buckets_per_1000'] != split['test_buckets_per_1000']:
        raise ValueError('Unsupported asymmetric split assignment')
    try:
        for doc_hash,key,doc_id,chunks in db.execute('SELECT * FROM documents ORDER BY hash'):
            part = c.assign_partition(doc_hash,split['seed'],split['test_buckets_per_1000'])
            if part != 'train':
                continue  # Unselected holdout candidates remain reserved in the index.
            counts[key] += 1
            for index,text in enumerate(json.loads(chunks)):
                writers[key].write(dict(document_id=doc_id,chunk_index=index,source=key,
                    text_sha256=c.normalized_text_hash(text),text=text),len(text.encode()))
    finally:
        db.close()
        for writer in writers.values():writer.close()
    manifest['partitions']['train'] = {key:dict(text_bytes=w.total_text_bytes,
        target_text_bytes=w.total_text_bytes,chunks=w.total_chunks,documents=counts[key],files=w.files)
        for key,w in writers.items()}
    manifest['underfilled'] = [f'{part}/{key}' for part,sources in manifest['partitions'].items()
        for key,r in sources.items() if r['text_bytes'] < r['target_text_bytes']]
    manifest['document_pool'] = dict(parent_manifest_sha256=c.sha256_file(parent/'manifest.json'),
        scope='All accepted training documents; original evaluation shards preserved; remaining holdout candidates reserved in dedup.sqlite',
        selection_order='normalized document hash',code_sha256=c.sha256_file(Path(__file__)))
    (temporary/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    temporary.rename(output)
    print(json.dumps(dict(output=str(output),training_documents=dict(counts),
        training_bytes={k:w.total_text_bytes for k,w in writers.items()}),indent=2),flush=True)


def select_document_pool(documents_dir, output_dir, train_tokens, weights,
                         bytes_per_token=5.0):
    """Select a nested training prefix by source; keep the pool's evaluation fixed.

    Token counts are estimates until tokenization. Complete documents may slightly
    exceed byte targets. A source shortage is rejected before copying any files.
    """
    parent = Path(documents_dir).resolve()
    manifest = json.loads((parent/'manifest.json').read_text())
    if not manifest.get('document_pool'):
        raise ValueError('Export a document pool first')
    if (set(weights) != set(c.SOURCES) or
        any(not isinstance(v,(int,float)) or not math.isfinite(v) or v < 0 for v in weights.values()) or
        not math.isclose(sum(weights.values()),1.0,abs_tol=1e-9)):
        raise ValueError('Weights must specify all sources, be nonnegative and sum to one')
    if train_tokens <= 0 or not math.isfinite(bytes_per_token) or bytes_per_token <= 0:
        raise ValueError('Positive token target and byte estimate required')
    targets = {key:math.ceil(train_tokens*bytes_per_token*weight) for key,weight in weights.items()}
    shortages = {k:target-manifest['partitions']['train'][k]['text_bytes'] for k,target in targets.items()
                 if target > manifest['partitions']['train'][k]['text_bytes']}
    if shortages:
        raise ValueError(f'Pool is too small; missing UTF-8 bytes by source: {shortages}')
    parent,output,temporary,manifest = _clone_without_train(parent,output_dir)
    records = {}
    for key,target in targets.items():
        writer = c.MaterializedParquetWriter(temporary/'train'/key,c.DEFAULT_DOCUMENT_SHARD_BYTES)
        def rows():
            for file in manifest['partitions']['train'][key]['files']:
                path=parent/'train'/key/file['file']
                if c.sha256_file(path) != file['sha256']:
                    raise ValueError(f'Pool shard checksum mismatch: {path}')
                for batch in c.pq.ParquetFile(path).iter_batches(batch_size=256):
                    yield from batch.to_pylist()
        documents=0
        try:
            for doc_id,group in groupby(rows(),key=lambda row:row['document_id']):
                if writer.total_text_bytes >= target:break
                documents+=1
                for row in group:writer.write(row,len(row['text'].encode()))
        finally:
            writer.close()
        records[key]=dict(target_text_bytes=target,text_bytes=writer.total_text_bytes,
                          chunks=writer.total_chunks,documents=documents,files=writer.files)
    manifest['partitions']['train']=records
    manifest.pop('document_pool')
    manifest['mixtures']={'selected':weights}
    manifest['selection']=dict(pool=str(parent),pool_manifest_sha256=c.sha256_file(parent/'manifest.json'),
        train_tokens_estimate=train_tokens,bytes_per_token=bytes_per_token,weights=weights,
        evaluation='Inherited unchanged from pool',code_sha256=c.sha256_file(Path(__file__)))
    manifest['underfilled']=[f'{part}/{key}' for part,sources in manifest['partitions'].items()
        for key,r in sources.items() if r['text_bytes'] < r['target_text_bytes']]
    if manifest['underfilled']:
        raise RuntimeError(f'Incomplete selection preserved at {temporary}: {manifest["underfilled"]}')
    (temporary/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    temporary.rename(output)
    print(f'Selected documents: {output}',flush=True)
