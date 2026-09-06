"""Audit all twelve 421m Muon runs without changing training artifacts.

Run from the project root: .venv/bin/python notes/muon_analysis/analyze_sweep.py
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

torch.set_num_threads(4)
histories, summary = {}, {}
for row in csv.DictReader((ROOT / "src/pavullmo/pretrain_runs.csv").open()):
    name = row["experiment_name"]
    if "421m_muon" not in name:
        continue
    events = EventAccumulator(str(ROOT / "runs" / name), size_guidance={"scalars": 0, "tensors": 0})
    events.Reload()
    s = {t: np.array([(e.step, e.value) for e in events.Scalars(t)]) for t in events.Tags()["scalars"]}
    assert all(len(np.unique(a[:, 0])) == len(a) for a in s.values()), name
    c = json.loads(events.Tensors("configuration/text_summary")[0].tensor_proto.string_val[0])
    key = (c["muon_lr_multiplier"], c["muon_weight_decay"])
    assert key not in histories
    histories[key] = s
    checkpoint = torch.load(ROOT / "models" / Path(row["model_path"]).name, weights_only=True, map_location="cpu")
    assert checkpoint["config"] == row["config"]
    assert abs(float(row["validation_loss"]) - s["validation/loss"][-1, 1]) < 1e-6
    groups = defaultdict(list)
    for pname, w in checkpoint["model_state_dict"].items():
        groups[gradient_group_name(pname)].append(w)
    weights = {}
    for g, ws in groups.items():
        count = sum(w.numel() for w in ws)
        norm = math.sqrt(sum(w.double().square().sum().item() for w in ws))
        weights[g] = {"count": count, "norm": norm, "rms": norm / math.sqrt(count)}
    windows = {}
    for lo, hi in [(300, 1500), (1800, 3000), (3600, 4800), (5400, 6035)]:
        windows[f"{lo}:{hi}"] = {t: float(a[(a[:, 0] >= lo) & (a[:, 0] <= hi), 1].mean())
                                for t, a in s.items() if np.any((a[:, 0] >= lo) & (a[:, 0] <= hi))}
    gtags = [t for t in s if t.startswith("gradient_groups/")]
    gs = np.array([a[(a[:, 0] >= 3600) & (a[:, 0] <= 4800), 1] for a in [s[t] for t in gtags]])
    shares = (gs**2 / (gs**2).sum(axis=0)).mean(axis=1)
    scaled_gradients = {}
    for g in ["attention_qkv", "attention_output", "ffn_input_gate", "ffn_output"]:
        wnorms = dict(s[f"optimizer/muon_groups/{g}/parameter_norm"])
        scaled_gradients[g] = float(np.mean([v*wnorms[st] for st, v in s[f"gradient_groups/{g}"] if 3600 <= st <= 4800]))
    summary[name] = {"row": row, "configuration": c, "checkpoint_groups": weights,
                     "window_means": windows, "late_gradient_energy_shares": dict(zip(gtags, shares.tolist())),
                     "late_mean_gradient_norm_times_weight_norm": scaled_gradients,
                     "final_clip_fraction": float(s["train/clip_fraction"][-1, 1]),
                     "decay_only_retention": float(np.exp(np.log1p(-s["train/muon_learning_rate"][:, 1]*key[1]).sum())),
                     "validation": s["validation/loss"].tolist()}
    print(name, flush=True)

assert len(summary) == 12
configs = [dict(part.split("=", 1) for part in d["row"]["config"].split(";")) for d in summary.values()]
varying = {k for k in configs[0] if len({c[k] for c in configs}) > 1}
assert varying == {"MUON_LR_MULTIPLIER", "MUON_WEIGHT_DECAY"}, varying
(OUT / "sweep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(3, 2, figsize=(13, 12), layout="constrained")
def plot(axis, array, label, color, smooth=1):
    x, y = array[:, 0] * 16384 / 1e6, array[:, 1]
    if smooth > 1:
        kernel = np.ones(smooth)/smooth
        x, y = np.convolve(x, kernel, "valid"), np.convolve(y, kernel, "valid")
    axis.plot(x, y, label=label, color=color, linewidth=1.5)

for mult, color in zip([.25, .5, .75, 1., 1.5, 2.], plt.cm.viridis(np.linspace(.05, .9, 6))):
    s = histories[mult, .05]
    plot(ax[0, 0], s["train/gradient_norm"], f"LR ×{mult:g}", color, 151)
    norm = dict(s["optimizer/muon_groups/attention_qkv/parameter_norm"])
    product = np.array([(st, v*norm[st]) for st, v in s["gradient_groups/attention_qkv"]])
    plot(ax[0, 1], product, f"LR ×{mult:g}", color, 9)
ax[0, 0].set(title="At WD=0.05, lower LR produces larger gradients", ylabel="Global gradient L2 norm", ylim=(0, 1.8))
ax[0, 0].legend(frameon=False, ncols=2)
ax[0, 1].set(title="Much of the QKV difference is parameter scale", ylabel="QKV gradient norm × weight norm", ylim=(0, 32))

for wd, color in zip([0, .05, .1], ["#2563eb", "#d97706", "#b91c1c"]):
    s = histories[2., wd]
    plot(ax[1, 0], s["optimizer/muon/parameter_norm"], f"WD={wd:g}", color)
    a = s["optimizer/muon/angular_step"].copy()
    a[:, 1] *= 1000
    plot(ax[1, 1], a, f"WD={wd:g}", color, 5)
    plot(ax[2, 0], s["validation/loss"], f"WD={wd:g}", color)
ax[1, 0].set(title="At LR ×2, decay substantially limits weight growth", ylabel="Muon weight L2 norm")
ax[1, 0].legend(frameon=False)
ax[1, 1].set(title="Smaller weights mean larger angular steps", ylabel="Muon angular step (milliradians)", ylim=(0, 17))
ax[2, 0].set(title="At LR ×2, WD ranking changes during cooldown", ylabel="Validation cross-entropy", xlim=(45, 100), ylim=(3.86, 4.28))
ax[2, 0].legend(frameon=False)

mults, wds = [.25, .5, .75, 1., 1.5, 2.], [0, .05, .1]
losses = np.full((len(mults), len(wds)), np.nan)
for i, lr in enumerate(mults):
    for j, wd in enumerate(wds):
        if (lr, wd) in histories:
            losses[i, j] = histories[lr, wd]["validation/loss"][-1, 1]
ax[2, 1].imshow(losses, cmap="YlOrRd", vmin=3.88, vmax=3.91, aspect="auto")
for i in range(len(mults)):
    for j in range(len(wds)):
        ax[2, 1].text(j, i, f"{losses[i,j]:.5f}" if np.isfinite(losses[i,j]) else "—", ha="center", va="center", color="white" if losses[i,j] > 3.9 else "black")
ax[2, 1].set(title="Final validation losses · one seed", xticks=np.arange(3), xticklabels=wds,
              yticks=np.arange(6), yticklabels=mults, xlabel="Muon weight decay", ylabel="Muon LR multiplier")
for axis in [*ax[0], *ax[1], ax[2, 0]]:
    axis.axvline(4871*16384/1e6, color="#64748b", linestyle="--", alpha=.6)
    axis.set_xlabel("Actual training tokens (millions)")
    axis.grid(alpha=.15)
fig.suptitle("Muon LR / weight-decay sweep · 12 runs · 6,035 steps each\nDashed line: WSD decay starts at approximately 79.8M actual tokens", fontsize=15)
fig.savefig(OUT / "sweep_comparison.png", dpi=150)
plt.close(fig)
