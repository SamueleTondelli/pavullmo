"""Inspect context and target sensitivity in saved benchmark score matrices.

Rows of each matrix are contexts; columns are targets. All variance and margin
calculations use mean log probability per target token, so targets of different
lengths are on a comparable scale. The original scorer's context ranking uses
total log probability, which has the same ranking within each target column.
Column-centering subtracts each target's average over all contexts in the set;
it uses the full contrast set and is a diagnostic, not a deployable score for
an isolated prompt.

Examples:
    python src/evaluation/analyze_score_matrices.py
    python src/evaluation/analyze_score_matrices.py --all-models
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median


DEFAULT_SCORES = Path(__file__).resolve().parent / "benchmarks" / "scores"


def variance(values: list[float]) -> float:
    average = sum(values) / len(values)
    return sum((value - average) ** 2 for value in values) / len(values)


@dataclass
class SetStats:
    category: str
    benchmark: str
    size: int
    across_contexts: float
    across_targets: float
    context_main_effect: float
    target_main_effect: float
    interaction: float
    diagonal_lift: float
    positive_lift: bool
    collapsed: bool
    correct_top_targets: float
    centered_correct_top_targets: float
    first_top_targets: float
    tied_top_rows: int
    context_wins: int
    target_wins: int
    centered_target_wins: int
    comparisons: int
    context_absolute_margins: list[float]
    target_absolute_margins: list[float]


def analyze_set(raw: dict, category: str, benchmark: str) -> SetStats:
    matrix = raw["mean_token_logprob_matrix"]
    size = len(matrix)
    if size < 2 or any(len(row) != size for row in matrix):
        raise ValueError(f"Expected a square matrix of size >= 2 in {benchmark}")
    if any(not math.isfinite(value) for row in matrix for value in row):
        raise ValueError(f"Non-finite score in {benchmark}")

    row_means = [sum(row) / size for row in matrix]
    column_means = [
        sum(matrix[row][column] for row in range(size)) / size
        for column in range(size)
    ]
    grand_mean = sum(row_means) / size
    residuals = [
        [
            matrix[row][column] - row_means[row] - column_means[column]
            + grand_mean
            for column in range(size)
        ]
        for row in range(size)
    ]
    context_main_effect = variance(row_means)
    target_main_effect = variance(column_means)
    interaction = sum(
        value**2 for row in residuals for value in row
    ) / size**2
    across_contexts = sum(
        variance([matrix[row][column] for row in range(size)])
        for column in range(size)
    ) / size
    across_targets = sum(variance(row) for row in matrix) / size

    # Two-way additive decomposition: differences between the two conditional
    # variances reflect main effects, while interaction contributes to both.
    if not math.isclose(
        across_contexts, context_main_effect + interaction,
        rel_tol=1e-8, abs_tol=1e-10,
    ) or not math.isclose(
        across_targets, target_main_effect + interaction,
        rel_tol=1e-8, abs_tol=1e-10,
    ):
        raise ValueError(f"Variance decomposition failed in {benchmark}")

    top_targets = [
        [column for column, score in enumerate(row) if score == max(row)]
        for row in matrix
    ]
    centered = [
        [matrix[row][column] - column_means[column] for column in range(size)]
        for row in range(size)
    ]
    centered_top_targets = [
        [column for column, score in enumerate(row) if score == max(row)]
        for row in centered
    ]
    context_margins = [
        abs(matrix[index][index] - matrix[other][index])
        for index in range(size) for other in range(size) if other != index
    ]
    target_margins = [
        abs(matrix[index][index] - matrix[index][other])
        for index in range(size) for other in range(size) if other != index
    ]
    metrics = raw["metrics"]
    diagonal_lift = sum(residuals[index][index] for index in range(size)) / size
    return SetStats(
        category=category,
        benchmark=benchmark,
        size=size,
        across_contexts=across_contexts,
        across_targets=across_targets,
        context_main_effect=context_main_effect,
        target_main_effect=target_main_effect,
        interaction=interaction,
        diagonal_lift=diagonal_lift,
        positive_lift=diagonal_lift > 0,
        collapsed=(
            all(len(top) == 1 for top in top_targets)
            and len({top[0] for top in top_targets}) == 1
        ),
        correct_top_targets=sum(
            (1 / len(top) if index in top else 0)
            for index, top in enumerate(top_targets)
        ),
        centered_correct_top_targets=sum(
            (1 / len(top) if index in top else 0)
            for index, top in enumerate(centered_top_targets)
        ),
        first_top_targets=sum(1 / len(top) for top in top_targets if 0 in top),
        tied_top_rows=sum(len(top) > 1 for top in top_targets),
        context_wins=metrics["context_based_wins"],
        target_wins=metrics["target_based_wins"],
        centered_target_wins=sum(
            centered[index][index] > centered[index][other]
            for index in range(size)
            for other in range(size)
            if other != index
        ),
        comparisons=metrics["context_based_comparisons"],
        context_absolute_margins=context_margins,
        target_absolute_margins=target_margins,
    )


def load_model(scores_dir: Path, model: str) -> list[SetStats]:
    stats = []
    for path in sorted(scores_dir.glob(f"*/{model}/*-bench-scores.json")):
        with path.open(encoding="utf-8") as file:
            report = json.load(file)
        stats.extend(
            analyze_set(raw, path.parent.parent.name, path.stem)
            for raw in report["sets"]
        )
    if not stats:
        raise FileNotFoundError(
            f"No detailed reports found for {model} in {scores_dir}"
        )
    return stats


def mean_attribute(rows: list[SetStats], name: str) -> float:
    return sum(getattr(row, name) for row in rows) / len(rows)


def print_summary(label: str, rows: list[SetStats]) -> None:
    context_var = mean_attribute(rows, "across_contexts")
    target_var = mean_attribute(rows, "across_targets")
    ratio = target_var / context_var if context_var else math.inf
    higher_target_var = sum(
        row.across_targets > row.across_contexts for row in rows
    ) / len(rows)
    row_effect = mean_attribute(rows, "context_main_effect")
    column_effect = mean_attribute(rows, "target_main_effect")
    interaction = mean_attribute(rows, "interaction")
    total = row_effect + column_effect + interaction
    collapsed = sum(row.collapsed for row in rows) / len(rows)
    members = sum(row.size for row in rows)
    correct_top = sum(row.correct_top_targets for row in rows) / members
    centered_correct_top = (
        sum(row.centered_correct_top_targets for row in rows) / members
    )
    first_top = sum(row.first_top_targets for row in rows) / members
    ties = sum(row.tied_top_rows for row in rows) / members
    chance = len(rows) / members  # one correct target and one index 0 per set
    context_accuracy = (
        sum(row.context_wins / row.comparisons for row in rows) / len(rows)
    )
    target_accuracy = (
        sum(row.target_wins / row.comparisons for row in rows) / len(rows)
    )
    centered_target_accuracy = (
        sum(row.centered_target_wins / row.comparisons for row in rows)
        / len(rows)
    )
    context_margin = median(
        margin for row in rows for margin in row.context_absolute_margins
    )
    target_margin = median(
        margin for row in rows for margin in row.target_absolute_margins
    )

    print(f"\n{label}: {len(rows)} sets")
    print(
        f"  variance across contexts={context_var:.4f}; "
        f"across targets={target_var:.4f}; target/context={ratio:.2f}x"
    )
    print(f"  sets with higher across-target variance: {higher_target_var:.1%}")
    print(
        f"  variance parts: context baseline {row_effect / total:.1%}, "
        f"target baseline {column_effect / total:.1%}, "
        f"context-target interaction {interaction / total:.1%}"
    )
    lift = mean_attribute(rows, "diagonal_lift")
    positive_lift = mean_attribute(rows, "positive_lift")
    print(
        f"  matched-pair interaction lift={lift:+.4f}; "
        f"positive in {positive_lift:.1%} of sets"
    )
    print(
        f"  median absolute matched-vs-other margin: "
        f"fixed target {context_margin:.4f}, fixed context {target_margin:.4f}"
    )
    print(
        f"  pairwise accuracy (set mean): context {context_accuracy:.1%}, "
        f"target {target_accuracy:.1%} (chance 50%)"
    )
    print(
        f"  column-centered target accuracy: pairwise {centered_target_accuracy:.1%}, "
        f"top target {centered_correct_top:.1%} (diagnostic only)"
    )
    print(
        f"  same top target in every context: {collapsed:.1%} of sets; "
        f"correct top target {correct_top:.1%} (chance {chance:.1%})"
    )
    print(
        f"  target index 0 ranked top: {first_top:.1%} of contexts "
        f"(uniform index baseline {chance:.1%}); top ties {ties:.1%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--model", default="35m_balanced_1b5")
    selection.add_argument("--all-models", action="store_true")
    args = parser.parse_args()
    models = (
        sorted({path.name for path in args.scores_dir.glob("*/*") if path.is_dir()})
        if args.all_models else [args.model]
    )
    if not models:
        parser.error(f"No model score directories found in {args.scores_dir}")
    print("Variances use mean log probability per target token (population variance).")
    print(
        "Target index means saved contrast-set order, "
        "not a choice list in the prompt."
    )
    print(
        "Some benchmarks fix member order in their templates, so index 0 "
        "frequency alone does not prove a position bias."
    )
    for model in models:
        rows = load_model(args.scores_dir, model)
        benchmarks: dict[str, list[SetStats]] = defaultdict(list)
        categories: dict[str, list[SetStats]] = defaultdict(list)
        for row in rows:
            benchmarks[f"{row.category}/{row.benchmark}"].append(row)
            categories[row.category].append(row)
        print(f"\n=== {model}: {len(benchmarks)} benchmarks ===")
        print_summary("Overall", rows)
        higher_target_variance = sum(
            mean_attribute(group, "across_targets")
            > mean_attribute(group, "across_contexts")
            for group in benchmarks.values()
        )
        print(
            "Benchmarks with higher across-target variance: "
            f"{higher_target_variance}/{len(benchmarks)}"
        )
        for category, group in sorted(categories.items()):
            print_summary(category, group)
    print(
        "\nInterpretation: across-target minus across-context variance equals "
        "target-baseline minus context-baseline variance. A large ratio "
        "shows stable target preferences, not that target rankings are random."
    )


if __name__ == "__main__":
    main()
