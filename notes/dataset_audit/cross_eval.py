"""Evaluate saved models on identical source-stratified blocks from all mixes.

This is a diagnostic sample, not a replacement for complete held-out evaluation.
All sample offsets, per-block losses, and effective evaluation settings are saved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import ROOT, OUT, MODELS, artifacts, write
sys.path.insert(0, str(ROOT / "src/model"))
from model import DecoderTransformer


def load_model(name, device):
    checkpoint = torch.load(ROOT / "artifacts/models" / (name + ".pt"),
                            map_location="cpu", weights_only=True)
    h = checkpoint["hyperparameters"]
    model = DecoderTransformer(
        vocab_size=h["VOCAB_SIZE"], n_blocks=h["N_BLOCKS"], embed_dim=h["EMBED_DIM"],
        attn_heads=h["ATTN_HEADS"], ffn_dim=h["FFN_DIM"], dropout=h["DROPOUT"],
        seq_len=h["SEQ_LEN"], rope_base=h["ROPE_BASE"], qk_norm=h["QK_NORM"],
        split_qkv_projections=h.get("SPLIT_QKV_PROJECTIONS", False),
        canon_layers=h.get("CANON_LAYERS", False))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-blocks-per-source", type=int, required=True)
    parser.add_argument("--train-blocks-per-source", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("requires CUDA with BF16 autocast")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    arts = artifacts()
    length = 1024
    samples = []
    rng = np.random.default_rng(args.seed)
    for name, artifact in arts.items():
        count = args.train_blocks_per_source if name.endswith("train") else args.eval_blocks_per_source
        for region in artifact.ranges:
            lo = (region["start_token"] + length - 1) // length
            hi = (region["start_token"] + region["tokens"] - length - 1) // length + 1
            for block in rng.choice(np.arange(lo, hi), size=count, replace=False):
                samples.append(dict(artifact=name, source=region["source"], start_token=int(block) * length))
    write("cross_eval_samples.json", dict(settings=vars(args), sequence_length=length, samples=samples))
    all_results = {}
    for label in args.models:
        model, checkpoint = load_model(MODELS[label], device)
        if checkpoint["hyperparameters"]["SEQ_LEN"] != length:
            raise ValueError("sample context length must match checkpoint")
        rows = []
        started = time.monotonic()
        for left in range(0, len(samples), args.batch_size):
            batch = samples[left:left + args.batch_size]
            tokens = np.stack([arts[s["artifact"]].read(s["start_token"], length + 1) for s in batch]).astype(np.int64)
            ids = torch.from_numpy(tokens).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(ids[:, :-1])
                losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                         ids[:, 1:].reshape(-1), reduction="none").view(len(batch), length)
            for s, loss, target in zip(batch, losses.cpu().numpy(), tokens[:, 1:], strict=True):
                content = (target != 1) & (target != 2)
                rows.append({**s, "loss": float(loss.mean()),
                             "content_loss": float(loss[content].mean()),
                             "bos_eos_targets": int((~content).sum())})
            if left % (args.batch_size * 100) == 0:
                print(label, left, "/", len(samples), "seconds", round(time.monotonic() - started), flush=True)
        groups = {}
        for name, artifact in arts.items():
            groups[name] = {}
            for region in artifact.ranges:
                values = np.array([r["loss"] for r in rows if r["artifact"] == name and r["source"] == region["source"]])
                groups[name][region["source"]] = dict(loss=float(values.mean()),
                                                      block_standard_error=float(values.std(ddof=1) / np.sqrt(len(values))),
                                                      blocks=len(values))
            groups[name]["weighted_loss"] = sum(groups[name][r["source"]]["loss"] * r["tokens"] / artifact.total_tokens
                                                 for r in artifact.ranges)
        output = dict(model=MODELS[label], settings=vars(args), sequence_length=length,
                      global_step=checkpoint["global_step"], hyperparameters=checkpoint["hyperparameters"],
                      seconds=time.monotonic() - started, groups=groups, blocks=rows)
        write(f"cross_eval_{label}.json", output)
        all_results[label] = groups
        print(label, json.dumps(groups), flush=True)
        del model, checkpoint
        torch.cuda.empty_cache()
    write("cross_eval_summary.json", all_results)


if __name__ == "__main__":
    main()
