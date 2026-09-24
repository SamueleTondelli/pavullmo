"""Analyze whether benchmark accuracy tracks the context-minus-target gap.

Reads saved benchmark reports; no checkpoint or GPU is needed.

Examples:
    uv run python src/evaluation/analyze_accuracy_gap.py
    uv run python src/evaluation/analyze_accuracy_gap.py --model 35m_balanced_1b
    uv run python src/evaluation/analyze_accuracy_gap.py --all-models
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SCORES = Path(__file__).resolve().parent / "benchmarks" / "scores"


@dataclass(frozen=True)
class Benchmark:
    category: str
    name: str
    context: float
    target: float

    @property
    def accuracy(self) -> float:
        return (self.context + self.target) / 2

    @property
    def gap(self) -> float:
        return self.context - self.target


def load_model(scores_dir: Path, model: str) -> list[Benchmark]:
    rows = []
    for path in sorted(scores_dir.glob(f"*/{model}/*-bench-scores.json")):
        with path.open(encoding="utf-8") as file:
            metrics = json.load(file)["metrics"]
        context = float(metrics["macro_context_based_accuracy"])
        target = float(metrics["macro_target_based_accuracy"])
        if not (math.isfinite(context) and math.isfinite(target)):
            raise ValueError(f"Non-finite accuracy in {path}")
        if not (0 <= context <= 1 and 0 <= target <= 1):
            raise ValueError(f"Accuracy outside [0, 1] in {path}")
        rows.append(Benchmark(path.parent.parent.name, path.stem, context, target))
    if not rows:
        raise FileNotFoundError(
            f"No benchmark reports found for {model} in {scores_dir}"
        )
    return rows


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 3:
        return math.nan
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    centered_x = [x - x_mean for x in xs]
    centered_y = [y - y_mean for y in ys]
    denominator = math.sqrt(
        sum(x * x for x in centered_x) * sum(y * y for y in centered_y)
    )
    if denominator == 0:
        return math.nan
    return sum(x * y for x, y in zip(centered_x, centered_y, strict=True)) / denominator


def ranks(values: list[float]) -> list[float]:
    """Average ranks for ties."""
    result = [0.0] * len(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2 + 1
        for index in order[start:end]:
            result[index] = rank
        start = end
    return result


def permutation_p(
    xs: list[float], ys: list[float], *, permutations: int, seed: int,
    groups: list[str] | None = None,
) -> float:
    """Two-sided permutation p-value, shuffling gaps within groups if given."""
    observed = pearson(xs, ys)
    if not math.isfinite(observed):
        return math.nan
    indices_by_group: dict[str, list[int]] = {}
    if groups is None:
        indices_by_group["all"] = list(range(len(xs)))
    else:
        for index, group in enumerate(groups):
            indices_by_group.setdefault(group, []).append(index)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(permutations):
        shuffled = ys.copy()
        for indices in indices_by_group.values():
            values = [ys[index] for index in indices]
            rng.shuffle(values)
            for index, value in zip(indices, values, strict=True):
                shuffled[index] = value
        extreme += abs(pearson(xs, shuffled)) >= abs(observed) - 1e-12
    return (extreme + 1) / (permutations + 1)


def fmt(value: float) -> str:
    return f"{value:+.3f}" if math.isfinite(value) else "undefined"


def analyze(rows: list[Benchmark], model: str, permutations: int, seed: int) -> None:
    accuracies = [row.accuracy for row in rows]
    gaps = [row.gap for row in rows]
    categories = sorted({row.category for row in rows})
    groups = [row.category for row in rows]
    accuracy_means = {
        category: sum(row.accuracy for row in rows if row.category == category)
        / sum(row.category == category for row in rows)
        for category in categories
    }
    gap_means = {
        category: sum(row.gap for row in rows if row.category == category)
        / sum(row.category == category for row in rows)
        for category in categories
    }
    centered_accuracies = [
        row.accuracy - accuracy_means[row.category] for row in rows
    ]
    centered_gaps = [row.gap - gap_means[row.category] for row in rows]

    overall_r = pearson(accuracies, gaps)
    overall_p = permutation_p(accuracies, gaps, permutations=permutations, seed=seed)
    spearman_r = pearson(ranks(accuracies), ranks(gaps))
    spearman_p = permutation_p(
        ranks(accuracies), ranks(gaps), permutations=permutations, seed=seed + 1
    )
    within_r = pearson(centered_accuracies, centered_gaps)
    within_p = permutation_p(
        centered_accuracies, centered_gaps,
        permutations=permutations, seed=seed + 2, groups=groups,
    )

    print(f"\n{model}: {len(rows)} benchmarks across {len(categories)} categories")
    print("Accuracy = (context + target) / 2; gap = context - target")
    print(f"Benchmark Pearson r={fmt(overall_r)}, permutation p={overall_p:.4f}")
    print(f"Benchmark Spearman rho={fmt(spearman_r)}, permutation p={spearman_p:.4f}")
    print(f"Within-category Pearson r={fmt(within_r)}, permutation p={within_p:.4f}")
    print("Category means (descriptive; five benchmarks per category):")
    print("  category                 n   accuracy       gap")
    for category in categories:
        count = sum(row.category == category for row in rows)
        print(
            f"  {category:23} {count:2d}   "
            f"{accuracy_means[category]:.3f}      {gap_means[category]:+.3f}"
        )
    print(
        "  Category-mean Pearson r="
        f"{fmt(pearson(list(accuracy_means.values()), list(gap_means.values())))}"
    )

    print("Chance split, using each benchmark's mean accuracy:")
    for label, subset in (
        ("below 0.5", [row for row in rows if row.accuracy < 0.5]),
        ("at/above 0.5", [row for row in rows if row.accuracy >= 0.5]),
    ):
        if subset:
            mean_gap = sum(row.gap for row in subset) / len(subset)
            positive = sum(row.gap > 0 for row in subset)
            negative = sum(row.gap < 0 for row in subset)
            print(
                f"  {label:12} n={len(subset):2d}, mean gap={mean_gap:+.3f}, "
                f"positive={positive}, negative={negative}"
            )
        else:
            print(f"  {label:12} n=0")

    # A=(C+T)/2 and G=C-T imply Cov(A,G)=(Var(C)-Var(T))/2.
    # This identity shows how a positive correlation can arise from unequal
    # context and target variation, even without a separate quality effect.
    context_mean = sum(row.context for row in rows) / len(rows)
    target_mean = sum(row.target for row in rows) / len(rows)
    context_variance = (
        sum((row.context - context_mean) ** 2 for row in rows) / len(rows)
    )
    target_variance = sum((row.target - target_mean) ** 2 for row in rows) / len(rows)
    print(
        f"Context variance={context_variance:.4f}; "
        f"target variance={target_variance:.4f}"
    )
    print(
        "Interpretation: mean accuracy and gap share the same two scores; "
        "their correlation is descriptive, not independent evidence that "
        "higher model quality causes a larger gap."
    )
    print(
        "Permutation p-values describe this collection of benchmarks and "
        "assume exchangeable benchmarks within each shuffle group."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--model", default="35m_balanced_1b5")
    selection.add_argument("--all-models", action="store_true")
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.permutations < 1:
        parser.error("--permutations must be positive")
    models = (
        sorted({path.name for path in args.scores_dir.glob("*/*") if path.is_dir()})
        if args.all_models else [args.model]
    )
    if not models:
        parser.error(f"No model score directories found in {args.scores_dir}")
    for model in models:
        analyze(load_model(args.scores_dir, model), model, args.permutations, args.seed)


if __name__ == "__main__":
    main()
