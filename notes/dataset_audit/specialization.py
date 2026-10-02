"""Attribute paired model loss differences to visible web topic groups.

Topic flags are deliberately broad: legitimate finance/vaping prose is included.
Decoded flagged samples are saved for manual semantic review. This is a
descriptive grouping discovered after looking at losses, not an unbiased test
of filtering effectiveness on a held-out benchmark.
"""
from __future__ import annotations

import json
import re

import numpy as np

from analyze import ROOT, RAW, MODELS, artifacts, write
from inspect_tokens import load_tokenizer


def main():
    arts = artifacts()
    sp = load_tokenizer(ROOT / "artifacts/tokenizers/production_16k/tokenizer.model", next(iter(arts.values())))
    data = {label: json.loads((RAW / f"cross_eval_{label}.json").read_text()) for label in MODELS}
    patterns = {
        "finance": re.compile(r"\b(?:prestit[oi]|finanziament[oi]|noipa)\b|cessione del quinto", re.I),
        "vaping": re.compile(r"sigarett[ae] elettronich[ae]|punt[oi] vendita sigarette", re.I),
    }
    rows = []
    for i, old in enumerate(data['old']['blocks']):
        if old['source'] != 'web':
            continue
        text = sp.decode(arts[old['artifact']].read(old['start_token'], 1024).tolist())
        flags = [key for key, pattern in patterns.items() if pattern.search(text)]
        row = dict(artifact=old['artifact'], start_token=old['start_token'], flags=flags,
                   losses={label: d['blocks'][i]['loss'] for label, d in data.items()})
        if flags:
            row['text'] = text
        rows.append(row)
    summary = dict(method=__doc__, groups={}, flagged_examples=[])
    for name in arts:
        values = [r for r in rows if r['artifact'] == name]
        summary['groups'][name] = {}
        for group in ['finance_or_vaping', 'other']:
            subset = [r for r in values if bool(r['flags']) == (group == 'finance_or_vaping')]
            if not subset:
                continue
            summary['groups'][name][group] = dict(blocks=len(subset), fraction=len(subset)/len(values),
                                                  losses={label: float(np.mean([r['losses'][label] for r in subset])) for label in MODELS})
        summary['groups'][name]['contribution_to_sanitized_web_loss_gain'] = {
            group: s['fraction'] * (s['losses']['old'] - s['losses']['sanitized'])
            for group,s in summary['groups'][name].items()}
    (RAW / 'specialization_blocks.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    for name in ['sanitized/validation','sanitized/test','finesynth/validation','finesynth/test']:
        matches = [r for r in rows if r['artifact']==name and r['flags']]
        summary['flagged_examples'].extend([{k:v for k,v in r.items() if k!='text'} for r in matches])
    write('specialization.json', summary)
    print(json.dumps(summary['groups'],indent=2),flush=True)


if __name__ == '__main__':
    main()
