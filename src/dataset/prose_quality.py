"""Conservative PDF prose selection from a verified language-audit report.

Reject whole documents; preserve original text and record all heuristic signals.
This is not a grammar checker, factual verifier, or reading-order reconstruction.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from language_quality import sha256
except ModuleNotFoundError:
    from src.dataset.language_quality import sha256

POLICY = dict(version='prose-v1', min_visible_characters=1000,
              min_prose_paragraph_characters=120, min_prose_paragraph_words=20,
              min_complete_prose_share=.60, min_fragmented_paragraphs=3,
              max_fragmented_paragraph_share=.20, max_table_character_share=.20,
              max_replacement_character_share=.001)


def measure_prose(text: str) -> dict:
    """Visible characters provide consistent denominators for layout signals."""
    visible = sum(not c.isspace() for c in text)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    complete_chars = 0
    prose_count = 0
    fragments = []
    for index, paragraph in enumerate(paragraphs):
        # Tables/lists/headings are not standalone prose evidence.
        value = paragraph.strip('*_# ')
        if (len(value) < POLICY['min_prose_paragraph_characters'] or
                len(value.split()) < POLICY['min_prose_paragraph_words'] or
                value.count('|') >= 2):
            continue
        prose_count += 1
        first_letter = next((c for c in value if c.isalpha()), '')
        starts_lower = bool(first_letter and first_letter.islower())
        open_end = not bool(re.search(r'[.!?][\"\'»”’)*_\]]*$', value))
        if starts_lower or open_end:
            fragments.append(dict(paragraph=index, starts_lower=starts_lower, open_end=open_end))
        else:
            complete_chars += sum(not c.isspace() for c in paragraph)
    table_chars = sum(sum(not c.isspace() for c in line) for line in text.splitlines() if line.count('|') >= 2)
    signals = dict(visible_characters=visible, prose_paragraphs=prose_count,
                   complete_prose_share=complete_chars / visible if visible else 0,
                   fragmented_paragraphs=len(fragments),
                   fragmented_paragraph_share=len(fragments) / prose_count if prose_count else 0,
                   table_character_share=table_chars / visible if visible else 0,
                   replacement_character_share=text.count('\ufffd') / visible if visible else 0,
                   fragments=fragments)
    reasons = []
    if visible < POLICY['min_visible_characters']:
        reasons.append('insufficient_standalone_text')
    if signals['complete_prose_share'] < POLICY['min_complete_prose_share']:
        reasons.append('insufficient_complete_prose')
    if len(fragments) >= POLICY['min_fragmented_paragraphs'] and signals['fragmented_paragraph_share'] >= POLICY['max_fragmented_paragraph_share']:
        reasons.append('fragmented_paragraph_boundaries')
    if signals['table_character_share'] > POLICY['max_table_character_share']:
        reasons.append('table_dominated')
    if signals['replacement_character_share'] > POLICY['max_replacement_character_share']:
        reasons.append('corrupted_characters')
    return dict(passes=not reasons, reasons=reasons, signals=signals)


def audit_prose(report_path: Path, output_dir: Path, total_documents: int | None = None) -> dict:
    previous = json.loads(report_path.read_text())
    source = Path(previous['input'])
    if sha256(source) != previous['input_sha256']:
        raise ValueError('Source checksum mismatch')
    rows = pq.read_table(source)
    indexed = {r['input_row']: r for r in previous['results']}
    if not indexed or len(indexed) != len(previous['results']) or any(i < 0 or i >= len(rows) for i in indexed):
        raise ValueError('Invalid source row selection')
    if total_documents is not None and total_documents < len(indexed):
        raise ValueError('Population must be at least sample size')
    decisions, retained = [], []
    for i, row in enumerate(rows.to_pylist()):
        if i not in indexed:
            continue
        original = indexed[i]
        text = row.get('text') or ''
        if hashlib.sha256(text.encode()).hexdigest() != original['text_sha256']:
            raise ValueError('Document hash mismatch')
        quality = measure_prose(text)
        accepted = original['candidate_pass'] and quality['passes']
        decisions.append(dict(input_row=i, document_id=original['document_id'],
                              language_pass=original['would_pass'], baseline_pass=original['candidate_pass'],
                              accepted=accepted, text_sha256=original['text_sha256'],
                              previous_reasons=original['rejection_reasons'], **quality))
        if accepted:
            retained.append(row)
    output_dir.mkdir(parents=True, exist_ok=False)
    pq.write_table(pa.Table.from_pylist(retained, schema=rows.schema), output_dir/'accepted.parquet')
    counts = dict(sample_documents=len(decisions), language_pass=sum(d['language_pass'] for d in decisions),
                  baseline_pass=sum(d['baseline_pass'] for d in decisions), prose_pass=len(retained))
    projection = None
    if total_documents is not None:
        projection = dict(population=total_documents, estimates={k:round(total_documents*v/len(decisions))
                          for k,v in counts.items() if k != 'sample_documents'},
                          limitation='Rough projection from a shuffled stream prefix, not a representative estimate or confidence interval.')
    report = dict(policy=POLICY, script_sha256=sha256(__file__), language_report=str(report_path.resolve()),
                  language_report_sha256=sha256(report_path), source_sha256=previous['input_sha256'],
                  counts=counts, projection=projection, decisions=decisions,
                  accepted_sha256=sha256(output_dir/'accepted.parquet'),
                  rejection_counts=dict(Counter(reason for d in decisions if d['baseline_pass'] for reason in d['reasons'])))
    (output_dir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (output_dir/'decisions.jsonl').write_text(''.join(json.dumps(d)+'\n' for d in decisions))
    lines=['# PDF extraction and prose audit','', 'Conservative heuristic selection, not a clean-text guarantee. Original source text is unchanged.', '',
           '| Stage | Documents |','|---|---:|']
    lines += [f'| {key} | {value} |' for key,value in counts.items()]
    if projection:
        lines += ['', f'Population: {total_documents:,}.', projection['limitation'], '']
        lines += [f'- {key}: approximately {value:,} documents.' for key,value in projection['estimates'].items()]
    lines += ['', 'Rules and document-level signals: `report.json`; decisions: `decisions.jsonl`.',
              'Retained source documents: `accepted.parquet`. These are not deduplicated train/validation/test releases.']
    (output_dir/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(counts=counts, rejection_counts=report['rejection_counts'],projection=projection),indent=2))
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language-report',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--total-documents',type=int)
    args=parser.parse_args(argv)
    audit_prose(args.language_report,args.output_dir,args.total_documents)
