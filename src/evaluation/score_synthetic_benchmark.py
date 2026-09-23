"""Score a generated synthetic benchmark with a PavuLLMo checkpoint.

For every contrast set this script computes the complete matrix
S[i,j] = log P(target[j] | context[i]) and derives context-based,
target-based, and bidirectional metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch


PAVULLMO_DIR = Path(__file__).resolve().parent.parent / "pavullmo"
sys.path.insert(0, str(PAVULLMO_DIR))

from generate import load_model


class BenchmarkError(ValueError):
    """The generated benchmark cannot be scored."""


@dataclass(frozen=True)
class Pair:
    set_index: int
    context_index: int
    target_index: int
    input_ids: list[int]
    target_ids: list[int]
    target_start: int


@dataclass(frozen=True)
class PairScore:
    logprob: float
    mean_token_logprob: float
    token_logprobs: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="path to the SentencePiece tokenizer model",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="path to a generated benchmark JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="path for the score report JSON",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="number of context-target pairs per model pass (default: 32)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="evaluation device (default: auto)",
    )
    return parser.parse_args()


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkError(f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise BenchmarkError(f"{label} must be a list")
    return value


def load_benchmark(
    path: Path,
) -> tuple[dict[str, Any], list[list[dict[str, str]]]]:
    with path.open(encoding="utf-8") as file:
        document = json.load(file)
    root = require_object(document, "benchmark")
    if root.get("version") != 1:
        raise BenchmarkError("benchmark version must be 1")
    if not isinstance(root.get("id"), str) or not root["id"]:
        raise BenchmarkError("benchmark id must be a nonempty string")

    raw_sets = require_list(root.get("sets"), "benchmark.sets")
    if not raw_sets:
        raise BenchmarkError("benchmark.sets must not be empty")

    sets: list[list[dict[str, str]]] = []
    for set_index, raw_set in enumerate(raw_sets):
        set_object = require_object(raw_set, f"benchmark.sets[{set_index}]")
        raw_members = require_list(
            set_object.get("members"), f"benchmark.sets[{set_index}].members"
        )
        if len(raw_members) < 2:
            raise BenchmarkError(
                f"benchmark.sets[{set_index}] must contain at least two members"
            )
        members: list[dict[str, str]] = []
        for member_index, raw_member in enumerate(raw_members):
            member = require_object(
                raw_member,
                f"benchmark.sets[{set_index}].members[{member_index}]",
            )
            context = member.get("context")
            target = member.get("target")
            if not isinstance(context, str) or not isinstance(target, str):
                raise BenchmarkError(
                    f"benchmark.sets[{set_index}].members[{member_index}] must "
                    "contain string context and target values"
                )
            if not target:
                raise BenchmarkError(
                    f"benchmark.sets[{set_index}].members[{member_index}].target "
                    "must not be empty"
                )
            members.append({"context": context, "target": target})
        sets.append(members)
    return root, sets


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise BenchmarkError("CUDA was requested but is not available")
    return torch.device(requested)


def encode_pairs(
    sets: list[list[dict[str, str]]],
    tokenizer: spm.SentencePieceProcessor,
    sequence_length: int,
) -> tuple[list[Pair], list[list[list[int]]]]:
    pairs: list[Pair] = []
    all_target_ids: list[list[list[int]]] = []

    for set_index, members in enumerate(sets):
        target_ids_for_set: list[list[int]] = []
        for target_index, member in enumerate(members):
            target_ids = tokenizer.encode(member["target"], out_type=int)
            if not target_ids:
                raise BenchmarkError(
                    f"set {set_index}, target {target_index} tokenizes to no tokens"
                )
            target_ids_for_set.append(target_ids)
        all_target_ids.append(target_ids_for_set)

        for context_index, context_member in enumerate(members):
            context = context_member["context"]
            context_ids = tokenizer.encode(context, out_type=int)
            for target_index, target_member in enumerate(members):
                target = target_member["target"]
                target_ids = target_ids_for_set[target_index]
                combined_ids = tokenizer.encode(context + target, out_type=int)
                expected_ids = [*context_ids, *target_ids]
                if combined_ids != expected_ids:
                    raise BenchmarkError(
                        f"set {set_index}, context {context_index}, target "
                        f"{target_index} is not tokenization-separable; make the "
                        "context-target boundary explicit, usually with leading "
                        "target whitespace"
                    )

                input_ids = [
                    tokenizer.bos_id(),
                    *context_ids,
                    *target_ids[:-1],
                ]
                if len(input_ids) > sequence_length:
                    raise BenchmarkError(
                        f"set {set_index}, context {context_index}, target "
                        f"{target_index} needs {len(input_ids)} model positions, "
                        f"but the checkpoint limit is {sequence_length}"
                    )
                pairs.append(
                    Pair(
                        set_index=set_index,
                        context_index=context_index,
                        target_index=target_index,
                        input_ids=input_ids,
                        target_ids=target_ids,
                        target_start=len(context_ids),
                    )
                )
    return pairs, all_target_ids


@torch.inference_mode()
def score_pairs(
    model: torch.nn.Module,
    pairs: list[Pair],
    *,
    batch_size: int,
    pad_id: int,
    device: torch.device,
) -> dict[tuple[int, int, int], PairScore]:
    scores: dict[tuple[int, int, int], PairScore] = {}

    for batch_start in range(0, len(pairs), batch_size):
        batch = pairs[batch_start : batch_start + batch_size]
        maximum_length = max(len(pair.input_ids) for pair in batch)
        input_ids = torch.full(
            (len(batch), maximum_length),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        for row, pair in enumerate(batch):
            input_ids[row, : len(pair.input_ids)] = torch.tensor(
                pair.input_ids,
                dtype=torch.long,
                device=device,
            )

        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else nullcontext()
        )
        with autocast:
            logits = model(input_ids)

        for row, pair in enumerate(batch):
            target_count = len(pair.target_ids)
            target_logits = logits[
                row,
                pair.target_start : pair.target_start + target_count,
            ].float()
            target_tensor = torch.tensor(
                pair.target_ids,
                dtype=torch.long,
                device=device,
            )
            selected_logits = target_logits.gather(
                dim=-1, index=target_tensor[:, None]
            ).squeeze(-1)
            token_logprobs = selected_logits - torch.logsumexp(
                target_logits, dim=-1
            )
            token_values = token_logprobs.cpu().tolist()
            total = float(sum(token_values))
            scores[(pair.set_index, pair.context_index, pair.target_index)] = (
                PairScore(
                    logprob=total,
                    mean_token_logprob=total / target_count,
                    token_logprobs=token_values,
                )
            )
    return scores


def comparison_credit(correct: float, alternative: float) -> float:
    if correct > alternative:
        return 1.0
    if correct == alternative:
        return 0.5
    return 0.0


def set_metrics(
    score_matrix: list[list[float]],
    mean_token_logprob_matrix: list[list[float]],
    target_token_counts: list[int],
) -> dict[str, Any]:
    size = len(score_matrix)
    context_wins = 0
    context_ties = 0
    target_wins = 0
    target_ties = 0
    bidirectional_wins = 0
    bidirectional_credit = 0.0
    comparisons = size * (size - 1)
    retrieval_credits: list[float] = []
    reciprocal_ranks: list[float] = []
    margins: list[float] = []
    per_token_margins: list[float] = []
    context_perfect = True
    target_perfect = True
    bidirectional_perfect = True

    for target_index in range(size):
        diagonal = score_matrix[target_index][target_index]
        distractors = [
            score_matrix[context_index][target_index]
            for context_index in range(size)
            if context_index != target_index
        ]

        for other_index in range(size):
            if other_index == target_index:
                continue

            context_credit = comparison_credit(
                diagonal,
                score_matrix[other_index][target_index],
            )
            target_credit = comparison_credit(
                mean_token_logprob_matrix[target_index][target_index],
                mean_token_logprob_matrix[target_index][other_index],
            )
            joint_credit = context_credit * target_credit

            if context_credit == 1.0:
                context_wins += 1
            elif context_credit == 0.5:
                context_ties += 1
            if target_credit == 1.0:
                target_wins += 1
            elif target_credit == 0.5:
                target_ties += 1
            if joint_credit == 1.0:
                bidirectional_wins += 1

            context_perfect &= context_credit == 1.0
            target_perfect &= target_credit == 1.0
            bidirectional_perfect &= joint_credit == 1.0
            bidirectional_credit += joint_credit

        greater = sum(value > diagonal for value in distractors)
        equal = sum(value == diagonal for value in distractors)
        retrieval_credits.append(1.0 / (equal + 1) if greater == 0 else 0.0)
        average_rank = 1.0 + greater + equal / 2.0
        reciprocal_ranks.append(1.0 / average_rank)

        margin = diagonal - max(distractors)
        margins.append(margin)
        per_token_margins.append(margin / target_token_counts[target_index])

    context_credit = context_wins + 0.5 * context_ties
    target_credit = target_wins + 0.5 * target_ties
    return {
        "context_based_accuracy": context_wins / comparisons,
        "context_based_wins": context_wins,
        "context_based_ties": context_ties,
        "context_based_comparisons": comparisons,
        "context_based_tie_adjusted_accuracy": context_credit / comparisons,
        "target_based_accuracy": target_wins / comparisons,
        "target_based_wins": target_wins,
        "target_based_ties": target_ties,
        "target_based_comparisons": comparisons,
        "target_based_tie_adjusted_accuracy": target_credit / comparisons,
        "bidirectional_accuracy": bidirectional_wins / comparisons,
        "bidirectional_wins": bidirectional_wins,
        "bidirectional_credit": bidirectional_credit,
        "bidirectional_comparisons": comparisons,
        "bidirectional_tie_adjusted_accuracy": (
            bidirectional_credit / comparisons
        ),
        # Backwards-compatible aliases for the original context comparison,
        # which awards half credit to ties.
        "pairwise_accuracy": context_credit / comparisons,
        "pairwise_wins": context_wins,
        "pairwise_ties": context_ties,
        "pairwise_comparisons": comparisons,
        "context_retrieval_accuracy": sum(retrieval_credits) / size,
        "mean_reciprocal_rank": sum(reciprocal_ranks) / size,
        "mean_worst_distractor_margin": sum(margins) / size,
        "mean_worst_distractor_margin_per_token": (
            sum(per_token_margins) / size
        ),
        "context_perfect_set": context_perfect,
        "target_perfect_set": target_perfect,
        "bidirectional_perfect_set": bidirectional_perfect,
        # Backwards-compatible alias for context_perfect_set.
        "perfect_set": context_perfect,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def aggregate_metrics(set_results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [result["metrics"] for result in set_results]
    context_wins = sum(metric["context_based_wins"] for metric in metrics)
    context_ties = sum(metric["context_based_ties"] for metric in metrics)
    target_wins = sum(metric["target_based_wins"] for metric in metrics)
    target_ties = sum(metric["target_based_ties"] for metric in metrics)
    bidirectional_wins = sum(
        metric["bidirectional_wins"] for metric in metrics
    )
    bidirectional_credit = sum(
        metric["bidirectional_credit"] for metric in metrics
    )
    total_comparisons = sum(metric["pairwise_comparisons"] for metric in metrics)
    macro_context_accuracy = mean(
        [metric["context_based_accuracy"] for metric in metrics]
    )
    micro_context_accuracy = context_wins / total_comparisons
    macro_pairwise_accuracy = mean(
        [metric["pairwise_accuracy"] for metric in metrics]
    )
    micro_pairwise_accuracy = (
        context_wins + 0.5 * context_ties
    ) / total_comparisons
    return {
        # Preserve the original primary metric and names for compatibility.
        "primary_metric": "macro_pairwise_accuracy",
        "macro_context_based_accuracy": macro_context_accuracy,
        "micro_context_based_accuracy": micro_context_accuracy,
        "context_based_wins": context_wins,
        "context_based_ties": context_ties,
        "context_based_comparisons": total_comparisons,
        "macro_context_based_tie_adjusted_accuracy": mean(
            [
                metric["context_based_tie_adjusted_accuracy"]
                for metric in metrics
            ]
        ),
        "micro_context_based_tie_adjusted_accuracy": micro_pairwise_accuracy,
        "macro_target_based_accuracy": mean(
            [metric["target_based_accuracy"] for metric in metrics]
        ),
        "micro_target_based_accuracy": target_wins / total_comparisons,
        "target_based_wins": target_wins,
        "target_based_ties": target_ties,
        "target_based_comparisons": total_comparisons,
        "macro_target_based_tie_adjusted_accuracy": mean(
            [
                metric["target_based_tie_adjusted_accuracy"]
                for metric in metrics
            ]
        ),
        "micro_target_based_tie_adjusted_accuracy": (
            target_wins + 0.5 * target_ties
        ) / total_comparisons,
        "macro_bidirectional_accuracy": mean(
            [metric["bidirectional_accuracy"] for metric in metrics]
        ),
        "micro_bidirectional_accuracy": bidirectional_wins / total_comparisons,
        "bidirectional_wins": bidirectional_wins,
        "bidirectional_credit": bidirectional_credit,
        "bidirectional_comparisons": total_comparisons,
        "macro_bidirectional_tie_adjusted_accuracy": mean(
            [
                metric["bidirectional_tie_adjusted_accuracy"]
                for metric in metrics
            ]
        ),
        "micro_bidirectional_tie_adjusted_accuracy": (
            bidirectional_credit / total_comparisons
        ),
        # Backwards-compatible names for the original context comparison,
        # which awards half credit to ties.
        "macro_pairwise_accuracy": macro_pairwise_accuracy,
        "micro_pairwise_accuracy": micro_pairwise_accuracy,
        "pairwise_wins": context_wins,
        "pairwise_ties": context_ties,
        "pairwise_comparisons": total_comparisons,
        "macro_context_retrieval_accuracy": mean(
            [metric["context_retrieval_accuracy"] for metric in metrics]
        ),
        "macro_mean_reciprocal_rank": mean(
            [metric["mean_reciprocal_rank"] for metric in metrics]
        ),
        "macro_mean_worst_distractor_margin": mean(
            [metric["mean_worst_distractor_margin"] for metric in metrics]
        ),
        "macro_mean_worst_distractor_margin_per_token": mean(
            [
                metric["mean_worst_distractor_margin_per_token"]
                for metric in metrics
            ]
        ),
        "perfect_set_accuracy": mean(
            [float(metric["perfect_set"]) for metric in metrics]
        ),
        "context_perfect_set_accuracy": mean(
            [float(metric["context_perfect_set"]) for metric in metrics]
        ),
        "target_perfect_set_accuracy": mean(
            [float(metric["target_perfect_set"]) for metric in metrics]
        ),
        "bidirectional_perfect_set_accuracy": mean(
            [float(metric["bidirectional_perfect_set"]) for metric in metrics]
        ),
    }


def build_set_results(
    sets: list[list[dict[str, str]]],
    all_target_ids: list[list[list[int]]],
    pair_scores: dict[tuple[int, int, int], PairScore],
    tokenizer: spm.SentencePieceProcessor,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for set_index, members in enumerate(sets):
        size = len(members)
        score_matrix = [
            [
                pair_scores[(set_index, context_index, target_index)].logprob
                for target_index in range(size)
            ]
            for context_index in range(size)
        ]
        mean_matrix = [
            [
                pair_scores[
                    (set_index, context_index, target_index)
                ].mean_token_logprob
                for target_index in range(size)
            ]
            for context_index in range(size)
        ]
        token_logprobs = [
            [
                pair_scores[
                    (set_index, context_index, target_index)
                ].token_logprobs
                for target_index in range(size)
            ]
            for context_index in range(size)
        ]
        target_ids = all_target_ids[set_index]
        target_token_counts = [len(ids) for ids in target_ids]
        results.append(
            {
                "members": members,
                "targets": [
                    {
                        "token_ids": ids,
                        "token_pieces": [tokenizer.id_to_piece(value) for value in ids],
                    }
                    for ids in target_ids
                ],
                "score_matrix": score_matrix,
                "mean_token_logprob_matrix": mean_matrix,
                "token_logprobs": token_logprobs,
                "metrics": set_metrics(
                    score_matrix,
                    mean_matrix,
                    target_token_counts,
                ),
            }
        )
    return results


def build_score_report(
    benchmark: dict[str, Any],
    set_results: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    checkpoint: Path,
    tokenizer: Path,
    device: torch.device,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "version": 1,
        "benchmark_id": benchmark["id"],
        "checkpoint": str(checkpoint),
        "tokenizer": str(tokenizer),
        "device": str(device),
        "scoring": {
            "score_matrix_axes": "rows=contexts, columns=targets",
            "pair_score": "sum of target-token conditional log-probabilities",
            "context_based_comparison": (
                "full-target log-probability; matched context versus "
                "alternative context for the same target"
            ),
            "target_based_comparison": (
                "mean target-token log-probability; matched target versus "
                "alternative target for the same context"
            ),
            "bidirectional_credit": (
                "context comparison credit multiplied by target comparison "
                "credit for each ordered contrast"
            ),
            "includes_eos": False,
            "accuracy_ties": "strict accuracies count ties as failures",
            "tie_adjusted_credit": 0.5,
        },
        "metrics": metrics,
        "sets": set_results,
    }
    if "language" in benchmark:
        report["language"] = benchmark["language"]
    return report


def write_score_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("error: --batch-size must be positive")

    input_paths = {
        args.checkpoint.resolve(),
        args.tokenizer.resolve(),
        args.benchmark.resolve(),
    }
    if args.output.resolve() in input_paths:
        raise SystemExit("error: --output must not overwrite an input file")

    try:
        benchmark, sets = load_benchmark(args.benchmark)
        device = choose_device(args.device)
        tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
        model, vocab_size, sequence_length = load_model(args.checkpoint, device)
        if tokenizer.vocab_size() != vocab_size:
            raise BenchmarkError(
                f"tokenizer vocabulary size {tokenizer.vocab_size()} does not "
                f"match checkpoint vocabulary size {vocab_size}"
            )
        if tokenizer.bos_id() < 0:
            raise BenchmarkError("tokenizer must define a BOS token")

        pairs, all_target_ids = encode_pairs(sets, tokenizer, sequence_length)
        pad_id = tokenizer.pad_id() if tokenizer.pad_id() >= 0 else 0
        pair_scores = score_pairs(
            model,
            pairs,
            batch_size=args.batch_size,
            pad_id=pad_id,
            device=device,
        )
        set_results = build_set_results(
            sets,
            all_target_ids,
            pair_scores,
            tokenizer,
        )
        metrics = aggregate_metrics(set_results)
        report = build_score_report(
            benchmark,
            set_results,
            metrics,
            checkpoint=args.checkpoint,
            tokenizer=args.tokenizer,
            device=device,
        )
        write_score_report(report, args.output)
    except (
        OSError,
        json.JSONDecodeError,
        BenchmarkError,
        RuntimeError,
        TypeError,
        KeyError,
    ) as error:
        raise SystemExit(f"error: {error}") from error

    print(f"Benchmark: {benchmark['id']} ({len(set_results)} sets)")
    print(
        "Context-based accuracy: "
        f"{metrics['macro_context_based_accuracy']:.6f}"
    )
    print(
        "Target-based accuracy: "
        f"{metrics['macro_target_based_accuracy']:.6f}"
    )
    print(
        "Bidirectional accuracy: "
        f"{metrics['macro_bidirectional_accuracy']:.6f}"
    )
    print(
        "Context retrieval accuracy: "
        f"{metrics['macro_context_retrieval_accuracy']:.6f}"
    )
    print(f"Mean reciprocal rank: {metrics['macro_mean_reciprocal_rank']:.6f}")
    print(
        "Context perfect-set accuracy: "
        f"{metrics['context_perfect_set_accuracy']:.6f}"
    )
    print(
        "Target perfect-set accuracy: "
        f"{metrics['target_perfect_set_accuracy']:.6f}"
    )
    print(
        "Bidirectional perfect-set accuracy: "
        f"{metrics['bidirectional_perfect_set_accuracy']:.6f}"
    )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
