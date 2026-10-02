"""Render measured dataset-audit summaries without rerunning inference."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import OUT


def main():
    bench = json.loads((OUT / "benchmarks.json").read_text())
    loss = json.loads((OUT / "cross_eval_summary.json").read_text())
    specialization = json.loads((OUT / "specialization.json").read_text())
    labels = ["old", "sanitized", "finesynth"]
    colors = ["#3478aa", "#e28a2f", "#378556"]
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    ax = fig.add_subplot(grid[0, :])
    categories = ["icl", "knowledge", "language-manipulation", "math", "math-languages", "reasoning"]
    x = np.arange(len(categories))
    for i, label in enumerate(labels):
        values = [bench[label]["categories"][c]["bidirectional"] * 100 for c in categories]
        bars = ax.bar(x + (i - 1) * .24, values, .24, color=colors[i], label=label)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
    ax.set_xticks(x, ["ICL", "Knowledge", "Language\nmanipulation", "Math", "Math /\nlanguage", "Reasoning"])
    ax.set_ylabel("Bidirectional accuracy (%)")
    ax.set_ylim(0, 87)
    ax.set_title("Same benchmark items, different strengths")
    ax.legend(loc="upper right", ncol=3)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 0])
    values = np.array([[loss[m][f"{ds}/test"]["weighted_loss"] for ds in labels] for m in labels])
    ax.imshow(values, cmap="YlOrRd", vmin=2.9, vmax=3.8, aspect="auto")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{values[i,j]:.3f}", ha="center", va="center",
                    color="white" if values[i,j] > 3.6 else "#222222", fontsize=14)
    ax.set_xticks(range(3), ["Old", "Sanitized", "FineSynth\n(natural)"])
    ax.set_yticks(range(3), ["Old model", "Sanitized model", "FineSynth model"])
    ax.set_title("Sampled test loss (nats/token)\nCompare models within each column")
    ax.set_xlabel("Evaluation corpus; 128 blocks per source")

    ax = fig.add_subplot(grid[1, 1])
    contrib = [specialization["groups"][f"sanitized/{s}"]["contribution_to_sanitized_web_loss_gain"]
               for s in ["validation", "test"]]
    finance = np.array([c["finance_or_vaping"] for c in contrib])
    other = np.array([c["other"] for c in contrib])
    ax.bar([0, 1], finance, .55, color="#b65a56", label="Finance / vaping topics")
    ax.bar([0, 1], other, .55, bottom=finance, color="#8caaa2", label="Other web text")
    for i in range(2):
        ax.text(i, finance[i] / 2, f"{finance[i]:.3f}", ha="center", va="center", color="white")
        ax.text(i, finance[i] + other[i] / 2, f"{other[i]:.3f}", ha="center", va="center")
    ax.set_xticks([0, 1], ["Sanitized\nvalidation", "Sanitized\ntest"])
    ax.set_ylabel("Old loss minus sanitized loss (nats/token)")
    ax.set_title("Where the web loss improvement comes from")
    ax.set_ylim(0, .53)
    ax.legend(loc="upper center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Corpus specialization can lower loss without improving every skill", fontsize=15)
    fig.savefig(OUT / "summary.png", dpi=160)


if __name__ == "__main__":
    main()
