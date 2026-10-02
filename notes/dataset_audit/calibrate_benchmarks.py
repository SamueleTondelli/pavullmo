"""Diagnose benchmark target priors with independently scored neutral prefixes.

Subtract mean target-token log P(target | prefix) with fixed coefficient one.
No alternative contexts or task-specific calibration tuning are used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F

from analyze import ROOT, MODELS, write
from cross_eval import load_model


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", action="append", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / "artifacts/tokenizers/production_16k/tokenizer.model"))
    result = {"settings": vars(args), "method": __doc__, "models": {}}
    for label, model_name in MODELS.items():
        reports = {}
        targets = set()
        for path in sorted((ROOT / "artifacts/benchmarks/scores").glob(f"*/{model_name}/*scores.json")):
            data = json.loads(path.read_text())
            reports[f"{path.parent.parent.name}/{path.name}"] = data
            for s in data["sets"]:
                targets.update(tuple(t["token_ids"]) for t in s["targets"])
        targets = sorted(targets, key=lambda t: (len(t), t))
        model, _ = load_model(model_name, device)
        model_result = {}
        for prefix in args.prefix:
            context = [sp.bos_id(), *sp.encode(prefix)]
            priors = {}
            for left in range(0, len(targets), args.batch_size):
                batch = targets[left:left + args.batch_size]
                size = len(context) + max(map(len, batch))
                ids = torch.full((len(batch), size), sp.pad_id(), dtype=torch.long, device=device)
                for i, target in enumerate(batch):
                    ids[i, :len(context)] = torch.tensor(context, device=device)
                    ids[i, len(context):len(context) + len(target)] = torch.tensor(target, device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(ids[:, :-1])
                    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
                                             reduction="none").view(len(batch), -1)
                for i, target in enumerate(batch):
                    priors[target] = -float(losses[i, len(context) - 1:len(context) - 1 + len(target)].mean())
            category_rows = defaultdict(list)
            benchmark_rows = {}
            for key, data in reports.items():
                rows = []
                for s in data["sets"]:
                    a = np.asarray(s["mean_token_logprob_matrix"], dtype=float)
                    baseline = np.array([priors[tuple(t["token_ids"])] for t in s["targets"]])
                    corrected = a - baseline[None, :]
                    mask = ~np.eye(len(a), dtype=bool)
                    context_wins = np.diag(a)[:, None] > a.T
                    target_wins = np.diag(a)[:, None] > a
                    calibrated_wins = np.diag(corrected)[:, None] > corrected
                    rows.append(dict(raw_target=float(target_wins[mask].mean()),
                                     calibrated_target=float(calibrated_wins[mask].mean()),
                                     raw_bidirectional=float((context_wins & target_wins)[mask].mean()),
                                     calibrated_bidirectional=float((context_wins & calibrated_wins)[mask].mean())))
                means = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
                benchmark_rows[key] = means
                category_rows[key.split('/')[0]].append(means)
            model_result[prefix] = dict(categories={c: {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
                                                    for c, rows in category_rows.items()},
                                        benchmarks=benchmark_rows, unique_targets=len(targets))
            print(label, repr(prefix), model_result[prefix]["categories"], flush=True)
        result["models"][label] = model_result
        del model
        torch.cuda.empty_cache()
    write("neutral_calibration.json", result)


if __name__ == "__main__":
    main()
