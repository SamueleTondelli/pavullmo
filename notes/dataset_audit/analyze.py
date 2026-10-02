"""Reproduce metadata, overlap, ordering, benchmark, and TensorBoard audits."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dataset"))
from inspect_tokens import TokenArtifact, load_tokenizer

OUT = Path(__file__).resolve().parent
CACHE = ROOT / "artifacts/dataset_audit"
RAW = CACHE / "raw"
PATHS = {
    "old": {"train": "train_balanced_1b", "validation": "validation", "test": "test"},
    "sanitized": {"train": "balanced_sanitized_v2_16k/train_balanced",
                  "validation": "balanced_sanitized_v2_16k/validation",
                  "test": "balanced_sanitized_v2_16k/test"},
    "finesynth": {"train": "finesynth_mix/train_finesynth_mix",
                 "validation": "finesynth_mix/validation_finesynth",
                 "test": "finesynth_mix/test_finesynth"},
}
MODELS = {"old": "35m_balanced_1b", "sanitized": "35m_newbalsanitized_1b",
          "finesynth": "35m_finesynth_1b"}


def artifacts():
    return {f"{mix}/{split}": TokenArtifact(ROOT / "artifacts/datasets" / path)
            for mix, splits in PATHS.items() for split, path in splits.items()}


def write(name, result):
    raw = (name.startswith("events_") or name.startswith("cross_eval_") and
           name != "cross_eval_summary.json" or
           name in ["decoded_samples.json", "overlap_examples.json"])
    target = RAW if raw else OUT
    target.mkdir(parents=True, exist_ok=True)
    (target / name).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def schedule():
    import torch
    from torch.utils.data import DataLoader, Dataset

    class Indices(Dataset):
        def __len__(self):
            return (1_000_000_000 - 1) // 1024

        def __getitem__(self, i):
            return i

    loader = DataLoader(Indices(), batch_size=16, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(42))
    order = np.concatenate([batch.numpy() for batch in loader])
    steps = len(order) // 16
    decay_steps = math.ceil((steps - 219) * .4)
    decay_start = steps - decay_steps
    result = dict(method="PyTorch DataLoader RandomSampler, same dedicated seed and base-seed draw",
                  sequence_length=1024, batch_size=16, seed=42, steps=steps,
                  decay_start_step_index=decay_start, decay_steps=decay_steps,
                  omitted_blocks=len(Indices()) - len(order), mixes={})
    for name, artifact in artifacts().items():
        if not name.endswith("train"):
            continue
        starts = order * 1024
        labels = np.zeros(len(starts), dtype=int)
        for i, r in enumerate(artifact.ranges):
            labels[(starts >= r["start_token"]) &
                   (starts < r["start_token"] + r["tokens"])] = i
        windows = {"all": labels, "stable": labels[:decay_start * 16],
                   "decay": labels[decay_start * 16:], "last_1m": labels[-976:]}
        result["mixes"][name] = {
            window: {r["source"]: float(np.mean(values == i))
                     for i, r in enumerate(artifact.ranges)}
            for window, values in windows.items()
        }
        result["mixes"][name]["deciles"] = [
            {r["source"]: float(np.mean(v == i)) for i, r in enumerate(artifact.ranges)}
            for v in np.array_split(labels, 10)]
    write("schedule.json", result)
    print("schedule complete", flush=True)


def benchmarks():
    result = {}
    all_sets = {}
    for label, model in {**MODELS, "new_unsanitized": "35m_newbal_1b",
                         "old_1b5": "35m_balanced_1b5",
                         "synth_only": "35m_synth_it_850m_muon_16k_20260929"}.items():
        categories = defaultdict(list)
        all_sets[label] = {}
        for path in sorted((ROOT / "artifacts/benchmarks/scores").glob(f"*/{model}/*scores.json")):
            data = json.loads(path.read_text())
            per_set = []
            for s in data["sets"]:
                a = np.array(s["mean_token_logprob_matrix"], dtype=float)
                n = len(a)
                mask = ~np.eye(n, dtype=bool)
                # Axes here are [matched member, alternative member], so the
                # context comparison uses the transposed score matrix.
                context = np.diag(a)[:, None] > a.T
                target = np.diag(a)[:, None] > a
                centered = a - a.mean(axis=0, keepdims=True)
                centered_target = np.diag(centered)[:, None] > centered
                residual = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
                per_set.append(dict(context=float(context[mask].mean()),
                                    target=float(target[mask].mean()),
                                    bidirectional=float((context & target)[mask].mean()),
                                    centered_target=float(centered_target[mask].mean()),
                                    interaction_lift=float(np.trace(residual) / n),
                                    target_bias_variance=float(a.mean(axis=0).var())))
            means = {k: float(np.mean([s[k] for s in per_set])) for k in per_set[0]}
            for key, stored in [('context', 'macro_context_based_accuracy'),
                                ('target', 'macro_target_based_accuracy'),
                                ('bidirectional', 'macro_bidirectional_accuracy')]:
                if not np.isclose(means[key], data['metrics'][stored], atol=1e-10, rtol=0):
                    raise ValueError(f"recomputed {key} differs from repository scorer: {path}")
            means["benchmark"] = path.name
            means["benchmark_id"] = data["benchmark_id"]
            means["member_sha256"] = hashlib.sha256(json.dumps(
                [s["members"] for s in data["sets"]], sort_keys=True).encode()).hexdigest()
            categories[path.parent.parent.name].append(means)
            all_sets[label][f"{path.parent.parent.name}/{path.name}"] = per_set
        result[label] = {
            "categories": {c: {k: float(np.mean([b[k] for b in bs]))
                               for k in bs[0] if k not in ["benchmark", "benchmark_id", "member_sha256"]}
                           for c, bs in categories.items()},
            "benchmarks": dict(categories),
        }
        result[label]["overall"] = {k: float(np.mean([v[k] for v in result[label]["categories"].values()]))
                                   for k in next(iter(result[label]["categories"].values()))}
    # Paired resampling of contrast sets within each of the five fixed tasks.
    # This interval describes item sampling, not training-seed uncertainty.
    rng = np.random.default_rng(42)
    comparisons = {}
    for baseline, candidate in [("old", "sanitized"), ("sanitized", "finesynth")]:
        comparisons[f"{candidate}-minus-{baseline}"] = {}
        for category in result[baseline]["categories"]:
            bs = []
            for key in all_sets[baseline]:
                if not key.startswith(category + "/"):
                    continue
                # Reject unpaired suites rather than giving a misleading interval.
                bmeta = next(x for x in result[baseline]["benchmarks"][category] if x["benchmark"] == key.split("/")[1])
                cmeta = next(x for x in result[candidate]["benchmarks"][category] if x["benchmark"] == key.split("/")[1])
                if bmeta["member_sha256"] != cmeta["member_sha256"]:
                    raise ValueError("benchmark members changed between models")
                delta = np.array([c["bidirectional"] - b["bidirectional"]
                                  for b, c in zip(all_sets[baseline][key], all_sets[candidate][key], strict=True)])
                bs.append(delta[rng.integers(len(delta), size=(2000, len(delta)))].mean(axis=1))
            means = np.mean(bs, axis=0)
            comparisons[f"{candidate}-minus-{baseline}"][category] = {
                "delta": result[candidate]["categories"][category]["bidirectional"] - result[baseline]["categories"][category]["bidirectional"],
                "paired_item_bootstrap_95pct": np.quantile(means, [.025, .975]).tolist(),
            }
    result["comparisons"] = comparisons
    write("benchmarks.json", result)
    print("benchmarks complete", flush=True)


def events():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    result = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    tags = ["validation/loss", "diagnostics/parameter_norm", "diagnostics/embedding_norm",
            "optimizer/muon/parameter_norm", "train/loss", "train/gradient_clipped",
            "train/tokens_seen", "optimizer/muon/update_to_parameter_norm",
            "optimizer/adamw/update_to_parameter_norm"]
    for label, model in MODELS.items():
        cache = RAW / f"events_{label}.json"
        if cache.exists():
            series = json.loads(cache.read_text())
        else:
            acc = EventAccumulator(str(ROOT / "artifacts/runs" / model), size_guidance={"scalars": 0})
            acc.Reload()
            series = {}
            for tag in tags:
                values = acc.Scalars(tag)
                # Some directories include an early failed attempt; retain the last
                # wall-time event at each step, and report how many were replaced.
                dedup = {v.step: [v.step, v.value] for v in sorted(values, key=lambda v: v.wall_time)}
                series[tag] = dict(values=sorted(dedup.values()), duplicates=len(values) - len(dedup))
            text_event = acc.Tensors("configuration/text_summary")[-1]
            series["configuration"] = text_event.tensor_proto.string_val[0].decode()
            write(cache.name, series)
        result[label] = {}
        for tag in tags:
            values = np.array(series[tag]["values"])
            first, last = values[0], values[-1]
            result[label][tag] = dict(first=first.tolist(), final=last.tolist(),
                                     minimum=values[values[:, 1].argmin()].tolist(),
                                     duplicates=series[tag]["duplicates"])
            if tag in ["validation/loss", "diagnostics/parameter_norm", "diagnostics/embedding_norm", "optimizer/muon/parameter_norm"]:
                ax = axes.flat[["validation/loss", "diagnostics/parameter_norm", "diagnostics/embedding_norm", "optimizer/muon/parameter_norm"].index(tag)]
                ax.plot(values[:, 0] * 16384 / 1e6, values[:, 1], label=label)
                ax.set_title(tag)
                ax.set_xlabel("Training tokens (millions)")
            if tag == "train/loss":
                result[label][tag]["last_1000_step_mean"] = float(values[-1000:, 1].mean())
            if tag == "train/gradient_clipped":
                result[label][tag]["whole_run_fraction"] = float(values[:, 1].mean())
                result[label][tag]["late_fraction"] = float(values[values[:, 0] > 10000, 1].mean())
        result[label]["configuration"] = series["configuration"]
        print("events", label, "complete", flush=True)
    for ax in axes.flat:
        ax.legend()
        ax.axvline(601.4, color="grey", linestyle="--", linewidth=.8)
    fig.suptitle("Validation sets differ; curves are within-run diagnostics")
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=150)
    write("events.json", result)


def overlap():
    arts = artifacts()
    indexes = {}
    for name, artifact in arts.items():
        path = CACHE / (name.replace("/", "_") + "_index_v2.npy")
        CACHE.mkdir(parents=True, exist_ok=True)
        if path.exists():
            index = np.load(path)
        else:
            rows = []
            for start, tokens in artifact.documents():
                rows.append((hashlib.blake2b(tokens[1:-1].tobytes(), digest_size=16).digest(),
                             start, len(tokens)))
            index = np.array(rows, dtype=[("hash", "V16"), ("start", "<u8"), ("length", "<u4")])
            np.save(path, index)
        indexes[name] = index
        print("indexed", name, len(index), flush=True)
    result = {"method": "all complete encoded BOS/EOS chunks, BLAKE2b-128 content-token digest; quota-truncated chunks excluded",
              "chunks": {}, "exact_overlap": {}}
    for name, index in indexes.items():
        _, counts = np.unique(index["hash"], return_counts=True)
        result["chunks"][name] = dict(count=len(index), duplicate_occurrences=int(np.sum(counts - 1)),
                                    token_length_quantiles=np.quantile(index["length"], [.1, .5, .9, .99]).tolist())
        result["exact_overlap"][name] = {}
        for other, train in indexes.items():
            if name == other:
                continue
            mask = np.isin(index["hash"], train["hash"])
            result["exact_overlap"][name][other] = dict(chunks=int(mask.sum()),
                                                       tokens=int(index["length"][mask].sum()),
                                                       token_fraction=float(index["length"][mask].sum() / arts[name].total_tokens))
    write("overlap.json", result)
    # Save matched examples from each evaluation into its own training artifact.
    sp = load_tokenizer(ROOT / "artifacts/tokenizers/production_16k/tokenizer.model", next(iter(arts.values())))
    examples = []
    for mix in PATHS:
        train = indexes[mix + "/train"]
        for split in ["validation", "test"]:
            name = mix + "/" + split
            index = indexes[name]
            mask = np.isin(index["hash"], train["hash"])
            for row in index[mask][:5]:
                examples.append(dict(artifact=name, start=int(row["start"]), tokens=int(row["length"]),
                                     text=sp.decode(arts[name].read(int(row["start"]), int(row["length"])).tolist())))
    write("overlap_examples.json", examples)
    print("overlap complete", flush=True)


def profiles():
    sp = None
    result = {}
    snippets = []
    for name, artifact in artifacts().items():
        if sp is None:
            sp = load_tokenizer(ROOT / "artifacts/tokenizers/production_16k/tokenizer.model", artifact)
        rng = np.random.default_rng(42)
        result[name] = {}
        for region in artifact.ranges:
            # Uniform-token samples, for token-share estimates rather than document-share estimates.
            records = []
            for start in rng.integers(region["start_token"], region["start_token"] + region["tokens"] - 1025, size=400):
                tokens = artifact.read(int(start), 1024)
                text = sp.decode(tokens.tolist())
                words = text.split()
                records.append(dict(chars=len(text), words=len(words), bos=int(np.sum(tokens == 1)),
                                    byte_tokens=int(sum(sp.is_byte(int(t)) for t in tokens)),
                                    question_marks=text.count("?"), arrows=text.count("→"),
                                    finance_or_vaping=bool(__import__('re').search(r"\b(?:prestit[oi]|finanziament[oi]|noipa)\b|cessione del quinto|sigarett[ae] elettronich[ae]", text, __import__('re').I)),
                                    escort=bool(__import__('re').search(r"\b(?:escort|massaggi erotici|accompagnatrice|prostitut[ae])\b", text, __import__('re').I)),
                                    assistant=bool(__import__('re').search(r"\b(?:in conclusione|è importante notare|questa è una questione|tuttavia, è importante|in sintesi)\b", text, __import__('re').I))))
                if len(records) <= 5:
                    snippets.append(dict(artifact=name, source=region["source"], start=int(start), text=text))
            means = {k: float(np.mean([r[k] for r in records])) for k in records[0]}
            means["tokens_per_whitespace_word"] = 1024 / means["words"]
            means["byte_token_share"] = means["byte_tokens"] / 1024
            means["sample_blocks"] = len(records)
            result[name][region["source"]] = means
        print("profiles", name, "complete", flush=True)
    write("profiles.json", result)
    write("decoded_samples.json", snippets)


def passages():
    """A lower bound on long verbatim overlap, not a semantic near-duplicate audit.

    Fingerprint 128-token windows at stride 128 relative to each encoded chunk.
    Different token alignment can hide an overlap; matched windows are exact.
    """
    arts = artifacts()
    evals = {k: v for k, v in arts.items() if not k.endswith("train")}
    lookup = defaultdict(list)
    sizes = {}
    for name, artifact in evals.items():
        count = 0
        for start, tokens in artifact.documents():
            for offset in range(1, len(tokens) - 128, 128):
                digest = hashlib.blake2b(tokens[offset:offset + 128].tobytes(), digest_size=16).digest()
                lookup[digest].append((name, start, start + offset))
                count += 1
        sizes[name] = count
    result = dict(method=passages.__doc__, window_tokens=128, stride=128,
                  evaluation_windows=sizes, matches={}, examples=[])
    for name, artifact in arts.items():
        if not name.endswith("train"):
            continue
        matched = defaultdict(set)
        chunks = defaultdict(set)
        for start, tokens in artifact.documents():
            for offset in range(1, len(tokens) - 128, 128):
                digest = hashlib.blake2b(tokens[offset:offset + 128].tobytes(), digest_size=16).digest()
                for target, chunk_start, window_start in lookup.get(digest, ()):
                    # Explicitly verify hash matches against the underlying tokens.
                    if not np.array_equal(tokens[offset:offset + 128], evals[target].read(window_start, 128)):
                        raise ValueError("fingerprint collision")
                    if window_start not in matched[target] and len(result["examples"]) < 30:
                        result["examples"].append(dict(train=name, train_start=start + offset,
                                                       evaluation=target, evaluation_start=window_start))
                    matched[target].add(window_start)
                    chunks[target].add(chunk_start)
        result["matches"][name] = {
            target: dict(windows=len(matched[target]), chunks=len(chunks[target]),
                         evaluation_token_fraction_lower_bound=len(matched[target]) * 128 / evals[target].total_tokens)
            for target in evals}
        print("passages", name, result["matches"][name], flush=True)
    write("passages.json", result)


def synth():
    import pyarrow.parquet as pq
    exercises = Counter()
    models = Counter()
    seeds = Counter()
    for path in sorted((ROOT / "artifacts/synth_it").glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=10000,
                                                     columns=["exercise", "model", "query_seed_url"]):
            exercises.update(batch.column(0).to_pylist())
            models.update(batch.column(1).to_pylist())
            seeds.update(batch.column(2).to_pylist())
    write("synth.json", dict(rows=sum(exercises.values()), exercises=dict(exercises), models=dict(models),
                             unique_seed_urls=len(seeds), top_seed_urls=seeds.most_common(30),
                             note="Full local Italian SYNTH pool; encoded 200M-token subset may differ"))
    print("synth complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="+", choices=["schedule", "benchmarks", "events", "overlap", "profiles", "passages", "synth"])
    args = parser.parse_args()
    for part in args.parts:
        globals()[part]()
