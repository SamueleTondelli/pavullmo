"""Reproducible page and paragraph language audits; never filter production data."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

import pyarrow.parquet as pq

LANGUAGE_POLICY = dict(version='clause-v1', short_min_characters=20,
                       short_min_letters=10, short_confidence=.95,
                       boundaries='blank lines, sentence punctuation, newlines, spaced slash')


def measure(text: str, metadata: dict, threshold: float = .95) -> dict:
    """Character-weighted page-language coverage, with unknowns in denominator.

    Offsets are Python character indices in the original, unnormalized text.
    Invalid alignment is reported as unassessable, never treated as Italian.
    This is a page-label proxy, not word-level language identification.
    """
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be between zero and one')
    ends, labels = metadata.get('page_ends'), metadata.get('per_page_languages')
    error = None
    if not isinstance(ends, list) or not isinstance(labels, list) or not ends or len(ends) != len(labels):
        error = 'missing or misaligned page metadata'
    elif any(type(end) is not int for end in ends):
        error = 'page offsets must be integer character indices'
    elif any(end <= start or end > len(text) for start, end in zip([0] + ends[:-1], ends)):
        error = 'page offsets must increase within the original text'
    if not text:
        error = 'empty text'
    if error:
        return dict(status='unassessable', error=error, italian_share=None,
                    other_share=None, unknown_share=None, would_pass=None, pages=[])
    counts = Counter(italian=0, other=0, unknown=0)
    pages = []
    start = 0
    for index, (end, label) in enumerate(zip(ends, labels)):
        category = ('italian' if label == 'ita_Latn' else
                    'unknown' if not isinstance(label, str) or label.strip().lower() in {'', 'unknown', 'und', 'unk'} else 'other')
        counts[category] += end - start
        pages.append(dict(page=index + 1, start=start, end=end, characters=end-start,
                          label=label, category=category))
        start = end
    if start < len(text):
        counts['unknown'] += len(text) - start
        pages.append(dict(page=None, start=start, end=len(text), characters=len(text)-start,
                          label='unlabelled_tail', category='unknown'))
    return dict(status='measured', characters=len(text), character_counts=dict(counts),
                italian_share=counts['italian']/len(text), other_share=counts['other']/len(text),
                unknown_share=counts['unknown']/len(text),
                would_pass=counts['italian']/len(text) >= threshold, pages=pages)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(input_path, output_dir, count=3, seed=42, threshold=.95, exclude_report=None):
    if count <= 0:
        raise ValueError('count must be positive')
    input_path, output_dir = Path(input_path), Path(output_dir)
    parquet = pq.ParquetFile(input_path)
    excluded = []
    if exclude_report is not None:
        previous = json.loads(Path(exclude_report).read_text())
        if previous['input_sha256'] != sha256(input_path):
            raise ValueError('Previous report belongs to a different input sample')
        excluded = sorted(set(previous.get('excluded_rows', []) + previous['selected_rows']))
    eligible = [i for i in range(parquet.metadata.num_rows) if i not in excluded]
    if count > len(eligible):
        raise ValueError('not enough documents for sampling without replacement')
    indices = sorted(random.Random(seed).sample(eligible, count))
    selected = []
    offset = 0
    for batch in parquet.iter_batches(batch_size=32):
        for row in batch.to_pylist():
            if offset in indices:
                selected.append((offset, row))
            offset += 1
    output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    lines = ['# FinePDFs-Edu: 95% Italian inspection' if threshold == .95 else '# FinePDFs-Edu language inspection', '',
             f'Seed: {seed}. Random sample: {count} of {parquet.metadata.num_rows} local documents.',
             f'Previously inspected rows excluded: {excluded}. Eligible documents: {len(eligible)}.',
             'This is random within the input pool, not a uniform sample of the full upstream corpus.',
             'Shares count characters on labelled pages, including whitespace; they are not foreign-word percentages.',
             'Unknown/unlabelled text stays in the denominator. No production filtering is applied.', '',
             '| Document | Italian | Other | Unknown | Would pass |',
             '|---|---:|---:|---:|---|']
    for index, row in selected:
        metadata = json.loads(row['metadata_json']) if 'metadata_json' in row else row
        text = row.get('text')
        if not isinstance(text, str):
            text = ''
        result = measure(text, metadata, threshold)
        result.update(input_row=index, document_id=row.get('document_id', metadata.get('id')),
                      url=metadata.get('url'), full_doc_lid=metadata.get('full_doc_lid'),
                      full_doc_lid_score=metadata.get('full_doc_lid_score'),
                      text_sha256=hashlib.sha256(text.encode()).hexdigest())
        results.append(result)
        shares = ['n/a' if result[k] is None else f'{result[k]:.2%}' for k in ['italian_share','other_share','unknown_share']]
        lines.append(f'| {index} | ' + ' | '.join(shares) + f" | {result['would_pass']} |")
        # Full source plus page-labelled excerpts allow manual verification.
        (output_dir / f'document_{index}.txt').write_text(text, encoding='utf-8')
        annotated = [f"ID: {result['document_id']}", f"URL: {result['url']}", '']
        for page in result['pages']:
            annotated += [f"PAGE {page['page']} | {page['label']} | characters {page['start']}:{page['end']}",
                          text[page['start']:page['end']], '']
        if result['status'] != 'measured':
            annotated += [result['error']]
        (output_dir / f'document_{index}_pages.txt').write_text('\n'.join(annotated), encoding='utf-8')
    manifest = dict(mode='audit_only', threshold=threshold, seed=seed, input=str(input_path.resolve()),
                    input_sha256=sha256(input_path), input_rows=parquet.metadata.num_rows,
                    selected_rows=indices, excluded_rows=excluded,
                    previous_report_sha256=sha256(exclude_report) if exclude_report else None,
                    script_sha256=sha256(__file__), results=results)
    companion = input_path.with_name('sample.json')
    if companion.exists():
        source_manifest = json.loads(companion.read_text())
        expected = source_manifest.get('hashes', {}).get(input_path.name)
        if expected and expected != manifest['input_sha256']:
            raise ValueError('Input hash does not match the sampling manifest')
        manifest['sampling_provenance'] = source_manifest
    (output_dir / 'report.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    (output_dir / 'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    return manifest



def prepare_lid(directory):
    """Pin model revision and bytes once, then verify cached reuse."""
    from huggingface_hub import HfApi, hf_hub_download
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / 'model.json'
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if sha256(lock['path']) != lock['sha256']:
            raise ValueError('GlotLID model checksum mismatch')
        return lock
    revision = HfApi().model_info('cis-lmu/glotlid').sha
    path = hf_hub_download('cis-lmu/glotlid', 'model_v3.bin', revision=revision,
                           cache_dir=str(directory / 'cache'))
    lock = dict(repo='cis-lmu/glotlid', revision=revision, filename='model_v3.bin',
                path=str(Path(path).resolve()), sha256=sha256(path))
    lock_path.write_text(json.dumps(lock, indent=2)+'\n')
    return lock


def paragraph_spans(text, max_chars=1000):
    """Preserve exact offsets; split long paragraphs at whitespace, not midword."""
    import re
    if max_chars < 1:
        raise ValueError('max_chars must be positive')
    for match in re.finditer(r'\S[\s\S]*?(?=\n\s*\n|\Z)', text):
        start, end = match.span()
        while end > start and text[end-1].isspace():
            end -= 1
        while start < end:
            stop = min(start + max_chars, end)
            if stop < end:
                cuts = [m.start() for m in re.finditer(r'\s+', text[start:stop])]
                if cuts:
                    stop = start + cuts[-1]
                else:
                    # One overlong token: keep it intact rather than fabricate words.
                    tail = re.search(r'\s', text[stop:end])
                    stop = stop + tail.start() if tail else end
            yield start, stop
            start = stop
            while start < end and text[start].isspace():
                start += 1


def language_spans(text, max_chars=1000):
    """Inspect sentence/clause boundaries without merging short neighbouring text."""
    import re
    for start, end in paragraph_spans(text, max_chars):
        cursor = start
        for boundary in re.finditer(r"\s+/\s+|(?<=[.!?;])\s+(?=[A-ZÀ-Ý])|\n", text[start:end]):
            stop = start + boundary.start()
            if text[cursor:stop].strip():
                yield cursor, stop
            # Keep punctuation and separators covered by assigning them to next span.
            cursor = stop
        if text[cursor:end].strip():
            yield cursor, end


def label_document(text, predict, min_chars=80, max_chars=1000, confidence=.8, threshold=.95):
    if not 0 <= confidence <= 1 or not 0 <= threshold <= 1 or min_chars < 1 or max_chars < min_chars:
        raise ValueError('invalid audit thresholds or block sizes')
    blocks = []
    counts = Counter(italian=0, other=0, unknown=0)
    for start, end in language_spans(text, max_chars):
        original = text[start:end]
        normalized = ' '.join(original.split())
        predictions = predict(normalized) if normalized else []
        label, score = predictions[0] if predictions else ('unknown', 0.0)
        short = len(normalized) < min_chars
        required_score = max(confidence, LANGUAGE_POLICY['short_confidence']) if short else confidence
        category = ('unknown' if len(normalized) < LANGUAGE_POLICY['short_min_characters'] or sum(c.isalpha() for c in normalized) < LANGUAGE_POLICY['short_min_letters'] or not math.isfinite(score) or score < required_score
                    or label.split('_')[0] in {'und', 'zxx', 'unknown'} else
                    'italian' if label == 'ita_Latn' else 'other')
        weight = sum(not c.isspace() for c in original)
        counts[category] += weight
        blocks.append(dict(start=start, end=end, text=original, normalized_characters=len(normalized),
                           weight=weight, label=label, confidence=score, category=category,
                           short_block=len(normalized)<min_chars, predictions=predictions))
    total = sum(counts.values())
    return dict(blocks=blocks, character_counts=dict(counts), measured_characters=total,
                italian_share=counts['italian']/total if total else 0,
                other_share=counts['other']/total if total else 0,
                unknown_share=counts['unknown']/total if total else 1,
                would_pass=total > 0 and counts['italian']/total >= threshold)


def audit_language(input_path, output_dir, model_dir, rows=None, min_chars=80, max_chars=1000,
                   confidence=.8, threshold=.95):
    import fasttext
    import importlib.metadata
    lock = prepare_lid(model_dir)
    model = fasttext.load_model(lock['path'])

    def predict(text):
        # The list interface avoids fasttext-wheel's NumPy-2 scalar copy=False bug.
        labels, scores = model.predict([text], k=3)
        return [(label.removeprefix('__label__'), float(score))
                for label, score in zip(labels[0], scores[0])]

    input_path, output_dir = Path(input_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    input_hash = sha256(input_path)
    provenance_path = input_path.with_name('sample.json')
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    if provenance and provenance.get('hashes', {}).get(input_path.name, input_hash) != input_hash:
        raise ValueError('Sample checksum mismatch')
    num_rows = pq.ParquetFile(input_path).metadata.num_rows
    selected = list(range(num_rows)) if rows is None else sorted(set(rows))
    if not selected or any(i < 0 or i >= num_rows for i in selected):
        raise ValueError('Invalid or empty row selection')
    output_dir.mkdir(parents=True)
    results = []
    retained = []
    try:
        from clean_dataset import accepted_chunks, SOURCES
    except ModuleNotFoundError:
        from src.dataset.clean_dataset import accepted_chunks, SOURCES
    index = 0
    for batch in pq.ParquetFile(input_path).iter_batches(batch_size=8):
        for row in batch.to_pylist():
            current = index
            index += 1
            if current not in selected:
                continue
            text = row.get('text') or ''
            metadata = json.loads(row['metadata_json']) if 'metadata_json' in row else row
            result = label_document(text, predict, min_chars, max_chars, confidence, threshold)
            result.update(input_row=current, document_id=row.get('document_id',metadata.get('id')),
                          url=metadata.get('url'), text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                          page_measure=measure(text,metadata,threshold))
            quality_counts = Counter()
            quality_pass = bool(list(accepted_chunks(SOURCES['edu_pdf'], dict(metadata, text=text), quality_counts)))
            reasons = [key for key in quality_counts if key.startswith('rejected_')]
            if not quality_pass and not reasons:
                reasons.append('no_usable_chunks')
            if not result['would_pass']:
                reasons.append('insufficient_italian_coverage')
            result.update(candidate_pass=quality_pass and result['would_pass'], rejection_reasons=reasons,
                          baseline_quality_counts=dict(quality_counts))
            if result['candidate_pass']:
                retained.append(row)
            results.append(result)
            (output_dir / f'document_{current}.txt').write_text(text,encoding='utf-8')
            print(f"row {current}: Italian {result['italian_share']:.1%}; other {result['other_share']:.1%}; unknown {result['unknown_share']:.1%}",flush=True)
    import pyarrow as pa
    source_schema = pq.ParquetFile(input_path).schema_arrow
    pq.write_table(pa.Table.from_pylist(retained, schema=source_schema), output_dir/'accepted.parquet')
    (output_dir/'decisions.jsonl').write_text(''.join(json.dumps({k:r[k] for k in ['input_row','document_id','candidate_pass','rejection_reasons','text_sha256']})+'\n' for r in results))
    counts = Counter(italian=0,other=0,unknown=0)
    for result in results:
        counts.update(result['character_counts'])
    total = sum(counts.values())
    sensitivity = []
    for cutoff in sorted(set([.90,.95,.98,1.0,threshold])):
        kept = [r for r in results if r['measured_characters'] and r['italian_share'] >= cutoff]
        sensitivity.append(dict(threshold=cutoff, documents=len(kept), total_documents=len(results),
                                retained_text_share=sum(r['measured_characters'] for r in kept)/total if total else 0))
    report = dict(mode='audit_and_candidate_selection', accepted_documents=len(retained),
                  accepted_sha256=sha256(output_dir/'accepted.parquet'),
                  quality_script_sha256=sha256(Path(__file__).with_name('clean_dataset.py')), input_sha256=input_hash, input=str(input_path.resolve()),
                  model=lock, script_sha256=sha256(__file__), selected_rows=selected,
                  config=dict(min_chars=min_chars,max_chars=max_chars,confidence=confidence,threshold=threshold,policy=LANGUAGE_POLICY),
                  weighting='non-whitespace characters; short blocks need >=20 characters, >=10 letters and >=0.95 score; uncertain blocks count as unknown',
                  versions={p:importlib.metadata.version(p) for p in ['fasttext-wheel','numpy','pyarrow']},
                  sampling_provenance=provenance, character_counts=dict(counts), sensitivity=sensitivity,
                  results=results)
    (output_dir/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    with (output_dir/'blocks.jsonl').open('w') as f:
        for result in results:
            for block in result['blocks']:
                f.write(json.dumps(dict(document_id=result['document_id'],input_row=result['input_row'],**block),ensure_ascii=False)+'\n')
    lines = ['# GlotLID paragraph audit','', 'Original source text is preserved. accepted.parquet is a candidate pool, not a training release.',
             'Scores are classifier outputs, not verified word-language percentages.',
             'Short blocks require stronger evidence; uncertain blocks stay unknown. Coverage uses non-whitespace characters.',
             '', '| Row | Italian | Other | Unknown | At threshold |', '|---|---:|---:|---:|---|']
    for r in results:
        lines.append(f"| [{r['input_row']}](document_{r['input_row']}.txt) | {r['italian_share']:.2%} | {r['other_share']:.2%} | {r['unknown_share']:.2%} | {r['would_pass']} |")
    lines += ['', '## Hypothetical whole-document retention', '', '| Threshold | Documents | Text retained |','|---|---:|---:|']
    for r in sensitivity:
        lines.append(f"| {r['threshold']:.0%} | {r['documents']}/{r['total_documents']} | {r['retained_text_share']:.2%} |")
    lines += ['', f"Combined language + existing quality rules: {len(retained)}/{len(results)} documents in `accepted.parquet`.",
              'Exclusion decisions: `decisions.jsonl`. No train/validation/test release was modified.', '', 'Detailed text, offsets, three model predictions and decisions: `blocks.jsonl`.',
              'These are sample diagnostics, not corpus-wide retention estimates.']
    (output_dir/'report.md').write_text('\n'.join(lines)+'\n')
    return report


def sample_pdfs(output_dir, count=100, seed=1729, buffer=32, revision=None):
    """Freeze an unfiltered shuffled-prefix pilot; a row budget is not a byte limit."""
    from datasets import load_dataset
    from huggingface_hub import HfApi
    import pyarrow as pa
    if min(count,buffer) <= 0:
        raise ValueError('count and buffer must be positive')
    directory = Path(output_dir)
    if directory.exists():
        raise FileExistsError(directory)
    repo = 'HuggingFaceFW/finepdfs-edu'
    revision = HfApi().dataset_info(repo,revision=revision or 'main').sha
    stream = load_dataset(repo,'ita_Latn',split='train',revision=revision,streaming=True).shuffle(seed=seed,buffer_size=buffer)
    directory.mkdir(parents=True)
    schema=pa.schema([('document_id',pa.string()),('text',pa.string()),('metadata_json',pa.string())])
    seen = 0
    iterator = iter(stream)
    try:
        with pq.ParquetWriter(directory/'sample.parquet',schema) as writer:
            for _ in range(count):
                try:
                    row=next(iterator)
                except StopIteration:
                    break
                text=row.get('text') or ''
                item=dict(document_id=str(row.get('id')),text=text,
                          metadata_json=json.dumps({k:v for k,v in row.items() if k!='text'},default=str))
                writer.write_table(pa.Table.from_pylist([item],schema=schema))
                seen+=1
                if seen % 10 == 0:
                    print(f'Saved {seen}/{count} PDF documents',flush=True)
    finally:
        close=getattr(iterator,'close',None)
        if close: close()
    if seen != count:
        raise ValueError(f'Source exhausted after {seen}/{count} documents')
    manifest=dict(dataset=repo,config='ita_Latn',split='train',revision=revision,rows=seen,
                  seed=seed,buffer=buffer,method='unfiltered seeded shuffled stream prefix; not globally uniform',
                  hashes={'sample.parquet':sha256(directory/'sample.parquet')})
    (directory/'sample.json').write_text(json.dumps(manifest,indent=2)+'\n')


def audit_main(argv):
    p=argparse.ArgumentParser(description='Dataset language checks: all operations are audit-only')
    commands=p.add_subparsers(dest='command',required=True)
    model=commands.add_parser('prepare-lid')
    model.add_argument('--model-dir',type=Path,default=Path(__file__).resolve().parents[2]/'artifacts/models/glotlid')
    sample=commands.add_parser('sample-pdfs')
    sample.add_argument('--output-dir',type=Path,required=True)
    sample.add_argument('--count',type=int,default=100)
    sample.add_argument('--seed',type=int,default=1729)
    sample.add_argument('--buffer',type=int,default=32)
    sample.add_argument('--revision')
    for name in ['audit-pages','audit-language']:
        sub=commands.add_parser(name)
        sub.add_argument('--input',type=Path,required=True)
        sub.add_argument('--output-dir',type=Path,required=True)
        sub.add_argument('--threshold',type=float,default=.95)
        if name=='audit-pages':
            sub.add_argument('--count',type=int,default=3)
            sub.add_argument('--seed',type=int,default=42)
            sub.add_argument('--exclude-report',type=Path)
        else:
            sub.add_argument('--model-dir',type=Path,default=Path(__file__).resolve().parents[2]/'artifacts/models/glotlid')
            sub.add_argument('--rows',type=int,nargs='+')
            sub.add_argument('--min-chars',type=int,default=80)
            sub.add_argument('--max-chars',type=int,default=1000)
            sub.add_argument('--confidence',type=float,default=.8)
    args=p.parse_args(argv)
    if args.command=='prepare-lid':
        print(json.dumps(prepare_lid(args.model_dir),indent=2))
    elif args.command=='sample-pdfs':
        sample_pdfs(args.output_dir,args.count,args.seed,args.buffer,args.revision)
    elif args.command=='audit-pages':
        run(args.input,args.output_dir,args.count,args.seed,args.threshold,args.exclude_report)
    else:
        audit_language(args.input,args.output_dir,args.model_dir,args.rows,args.min_chars,args.max_chars,args.confidence,args.threshold)
