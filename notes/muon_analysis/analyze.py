"""Reproduce the read-only 907m optimizer audit; run with .venv/bin/python.

Reads complete TensorBoard scalar histories and final checkpoints on CPU.
Writes summary.json and comparison.png beside this script. No training runs,
registry rows, checkpoints, or training code are modified.
"""

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src/pavullmo"))
from pretrain_base_muon import gradient_group_name

RUNS = {
    "Original AdamW": "10m_tuned_tok16k_907m",
    "Retuned AdamW": "10m_tok16k_907m_b20.99_wd0.05",
    "Muon": "10m_16k_907m_muon_lr0.5_mwd0.0",
}
COLORS = ["#94a3b8", "#2563eb", "#d97706"]
torch.set_num_threads(4)
rows = list(csv.DictReader((ROOT / "src/pavullmo/pretrain_runs.csv").open()))
histories, summary = {}, {}
for label, name in RUNS.items():
    matches = [r for r in rows if r["experiment_name"] == name]
    assert len(matches) == 1, (name, len(matches))
    row = matches[0]
    events = EventAccumulator(str(ROOT / "runs" / name), size_guidance={"scalars": 0, "tensors": 0})
    events.Reload()
    s = {tag: np.array([(e.step, e.value, e.wall_time) for e in events.Scalars(tag)])
         for tag in events.Tags()["scalars"]}
    assert all(len(np.unique(a[:, 0])) == len(a) for a in s.values()), name
    histories[label] = s
    config = json.loads(events.Tensors("configuration/text_summary")[0].tensor_proto.string_val[0])
    ck = torch.load(ROOT / "models" / Path(row["model_path"]).name, map_location="cpu", weights_only=True)
    assert ck["config"] == row["config"]
    assert abs(s["validation/loss"][-1, 1] - float(row["validation_loss"])) < 1e-6
    groups, spectra, qkv = defaultdict(list), defaultdict(list), defaultdict(list)
    for pname, w in ck["model_state_dict"].items():
        group = gradient_group_name(pname)
        groups[group].append(w)
        if w.ndim == 2 and pname != "embeddings.weight":
            sv = torch.linalg.svdvals(w)
            spectra[group].append({"name": pname, "spectral_norm": sv[0].item(),
                                   "stable_rank": (sv.square().sum() / sv[0].square()).item()})
        if ".c_attn.weight" in pname:
            for part, piece in zip("QKV", w.chunk(3, 0)):
                qkv[part].append(piece)
    def stats(ws):
        n = sum(w.numel() for w in ws)
        norm = math.sqrt(sum(w.double().square().sum().item() for w in ws))
        return {"count": n, "norm": norm, "rms": norm / math.sqrt(n)}
    gtags = [t for t in s if t.startswith("gradient_groups/")]
    masks = [(s[t][:, 0] >= 8000) & (s[t][:, 0] <= 10440) for t in gtags]
    g = np.array([s[t][mask, 1] for t, mask in zip(gtags, masks)])
    shares = (g*g / (g*g).sum(axis=0)).mean(axis=1)
    windows = {}
    for lo, hi in [(300, 1500), (3000, 6000), (8000, 10440), (12000, 13003)]:
        windows[f"{lo}:{hi}"] = {t: float(a[(a[:, 0] >= lo) & (a[:, 0] <= hi), 1].mean())
                                for t, a in s.items() if np.any((a[:, 0] >= lo) & (a[:, 0] <= hi))}
    summary[label] = {"run": name, "csv": row, "configuration": config,
                      "checkpoint_groups": {g: stats(ws) for g, ws in groups.items()},
                      "checkpoint_qkv": {g: stats(ws) for g, ws in qkv.items()},
                      "checkpoint_spectra": dict(spectra), "window_means": windows,
                      "late_gradient_energy_share": dict(zip(gtags, shares.tolist())),
                      "validation": s["validation/loss"][:, :2].tolist(),
                      "train_wall_seconds_first_to_last_scalar": float(s["train/loss"][-1, 2] - s["train/loss"][0, 2])}
    print(label, "final validation", row["validation_loss"], flush=True)

a = histories["Retuned AdamW"]["validation/loss"]
m = histories["Muon"]["validation/loss"]
assert np.array_equal(a[:, 0], m[:, 0])
summary["sweep_rows"] = [r for r in rows if "muon" in r["experiment_name"]]
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(3, 2, figsize=(13, 12), layout="constrained")
decay_tokens = 10446 * 16384 / 1e6
def curve(ax, s, tag, color, label, smooth=1):
    x = s[tag][:, 0] * 16384 / 1e6
    y = s[tag][:, 1]
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        x, y = np.convolve(x, kernel, "valid"), np.convolve(y, kernel, "valid")
    ax.plot(x, y, color=color, label=label, linewidth=1.6)

for (label, s), color in zip(histories.items(), COLORS):
    curve(axes[0, 0], s, "validation/loss", color, label)
    curve(axes[1, 0], s, "train/gradient_norm", color, label, smooth=201)
    curve(axes[1, 1], s, "diagnostics/update_norm", color, label, smooth=5)
axes[0, 0].set(ylim=(3.65, 5.3), title="Validation loss (early values above 5.3 omitted)", ylabel="Cross-entropy, nats/token")
axes[0, 0].legend(frameon=False)
axes[0, 1].plot(a[:, 0] * 16384 / 1e6, a[:, 1] - m[:, 1], color=COLORS[2])
axes[0, 1].set(title="Muon lead over retuned AdamW", ylabel="AdamW loss − Muon loss")
axes[0, 1].axhline(0, color="#94a3b8", linewidth=0.8)
axes[1, 0].set(title="Raw gradients: Muon is larger", ylabel="Global L2 norm (201-step mean)", ylim=(0, 0.85))
axes[1, 1].set(title="Actual updates: Muon is smaller", ylabel="Global update L2 norm (5 samples mean)")
for ax in axes[:2].flat:
    ax.axvline(decay_tokens, color="#475569", linestyle="--", alpha=.7)
    ax.set_xlabel("Actual training tokens (millions)")
    ax.grid(alpha=.15)

groups = ["embeddings", "attention_qkv", "attention_output", "ffn_input_gate", "ffn_output"]
short = ["Embedding", "QKV", "Attn out", "FFN in/gate", "FFN out"]
x = np.arange(len(groups))
for i, label in enumerate(["Retuned AdamW", "Muon"]):
    vals = [summary[label]["checkpoint_groups"][g]["rms"] for g in groups]
    axes[2, 0].bar(x + (i-.5)*.35, vals, width=.35, color=COLORS[i+1], label=label)
axes[2, 0].set(xticks=x, xticklabels=short, title="Final parameter RMS: total norm hides differences", ylabel="RMS (FFN groups include biases)")
axes[2, 0].legend(frameon=False)
for i, label in enumerate(["Retuned AdamW", "Muon"]):
    vals = [np.mean([v["stable_rank"] for v in summary[label]["checkpoint_spectra"][g]]) for g in groups[1:]]
    axes[2, 1].bar(np.arange(4)+(i-.5)*.35, vals, width=.35, color=COLORS[i+1])
axes[2, 1].set(xticks=np.arange(4), xticklabels=short[1:], title="Muon weights have broader singular spectra", ylabel="Mean stable rank = ||W||²F / ||W||²₂")
fig.suptitle("907m optimizer audit · 13,003 steps · 213.041M actual tokens\nDashed line: start of the final WSD decay", fontsize=15)
fig.savefig(OUT / "comparison.png", dpi=150)
plt.close(fig)
