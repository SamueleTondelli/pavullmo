"""Locate escort mentions in all web chunks and save examples for manual review.

The promotional-text rule is a screening heuristic; its token share is not a
verified adult-site prevalence estimate. Source URLs are unavailable here.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from analyze import ROOT, CACHE, artifacts, write
from inspect_tokens import load_tokenizer


def main():
    result = {"method": __doc__, "mixes": {}}
    patterns = [[427, 621, 11531], [3870, 621, 11531], [427, 621, 1897]]
    signal = re.compile(r"\b(?:massaggi|sensual\w*|incontri|annunci|client[ei]|riservat\w*|discre\w*|transex|transessual\w*|sesso|eros|erotic\w*)\b", re.I)
    for name, artifact in artifacts().items():
        if not name.endswith("train"):
            continue
        mix = name.split('/')[0]
        index = np.load(CACHE / f"{mix}_train_index_v2.npy")
        region = next(r for r in artifact.ranges if r['source'] == 'web')
        web_end = region['start_token'] + region['tokens']
        mentions = set()
        base = 0
        for shard in artifact.maps:
            if base >= web_end:
                break
            part = artifact.read(base, min(len(shard) + 2, artifact.total_tokens - base))
            for pattern in patterns:
                hits = np.flatnonzero(part[:-2] == pattern[0])
                hits = hits[(part[hits + 1] == pattern[1]) & (part[hits + 2] == pattern[2])]
                hits = hits[base + hits < web_end]
                doc_indices = np.searchsorted(index['start'], base + hits, side='right') - 1
                for hit, doc in zip(hits, doc_indices):
                    if doc >= 0 and base + hit < index[doc]['start'] + index[doc]['length']:
                        mentions.add(int(doc))
            base += len(shard)
        sp = load_tokenizer(ROOT / 'artifacts/tokenizers/production_16k/tokenizer.model', artifact)
        flagged = []
        for doc in sorted(mentions):
            row = index[doc]
            start, count = int(row['start']), int(row['length'])
            text = sp.decode(artifact.read(start, count).tolist())
            words = sorted(set(m.lower() for m in signal.findall(text)))
            if len(words) >= 2:
                flagged.append(dict(start_token=start, tokens=count, signals=words,
                                    text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest()))
        result['mixes'][mix] = dict(mention_chunks=len(mentions), screened_promotional_chunks=len(flagged),
                                   screened_chunk_tokens=sum(r['tokens'] for r in flagged),
                                   screened_share_of_web_tokens=sum(r['tokens'] for r in flagged) / region['tokens'],
                                   examples=flagged[:10])
        print(mix, {k:v for k,v in result['mixes'][mix].items() if k != 'examples'}, flush=True)
    write('escort_screen.json', result)


if __name__ == '__main__':
    main()
