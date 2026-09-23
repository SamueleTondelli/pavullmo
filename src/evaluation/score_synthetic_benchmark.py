"""Score a generated synthetic benchmark with a PavuLLMo checkpoint.

For every contrast set this script computes the complete matrix
S[i,j] = log P(target[j] | context[i]) and derives column-wise metrics.
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


def set_metrics(
    score_matrix: list[list[float]], target_token_counts: list[int]
) -> dict[str, Any]:
    size = len(score_matrix)
    wins = 0
    ties = 0
    comparisons = size * (size - 1)
    retrieval_credits: list[float] = []
    reciprocal_ranks: list[float] = []
    margins: list[float] = []
    per_token_margins: list[float] = []
    perfect = True

    for target_index in range(size):
        diagonal = score_matrix[target_index][target_index]
        distractors = [
            score_matrix[context_index][target_index]
            for context_index in range(size)
            if context_index != target_index
        ]
        for distractor in distractors:
            if diagonal > distractor:
                wins += 1
            elif diagonal == distractor:
                ties += 1
                perfect = False
            else:
                perfect = False

        greater = sum(value > diagonal for value in distractors)
        equal = sum(value == diagonal for value in distractors)
        retrieval_credits.append(1.0 / (equal + 1) if greater == 0 else 0.0)
        average_rank = 1.0 + greater + equal / 2.0
        reciprocal_ranks.append(1.0 / average_rank)

        margin = diagonal - max(distractors)
        margins.append(margin)
        per_token_margins.append(margin / target_token_counts[target_index])

    pairwise_credit = wins + 0.5 * ties
    return {
        "pairwise_accuracy": pairwise_credit / comparisons,
        "pairwise_wins": wins,
        "pairwise_ties": ties,
        "pairwise_comparisons": comparisons,
        "context_retrieval_accuracy": sum(retrieval_credits) / size,
        "mean_reciprocal_rank": sum(reciprocal_ranks) / size,
        "mean_worst_distractor_margin": sum(margins) / size,
        "mean_worst_distractor_margin_per_token": (
            sum(per_token_margins) / size
        ),
        "perfect_set": perfect,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def aggregate_metrics(set_results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [result["metrics"] for result in set_results]
    total_wins = sum(metric["pairwise_wins"] for metric in metrics)
    total_ties = sum(metric["pairwise_ties"] for metric in metrics)
    total_comparisons = sum(metric["pairwise_comparisons"] for metric in metrics)
    return {
        "primary_metric": "macro_pairwise_accuracy",
        "macro_pairwise_accuracy": mean(
            [metric["pairwise_accuracy"] for metric in metrics]
        ),
        "micro_pairwise_accuracy": (
            total_wins + 0.5 * total_ties
        ) / total_comparisons,
        "pairwise_wins": total_wins,
        "pairwise_ties": total_ties,
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
                "metrics": set_metrics(score_matrix, target_token_counts),
            }
        )
    return results


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
        report: dict[str, Any] = {
            "version": 1,
            "benchmark_id": benchmark["id"],
            "checkpoint": str(args.checkpoint),
            "tokenizer": str(args.tokenizer),
            "device": str(device),
            "scoring": {
                "score_matrix_axes": "rows=contexts, columns=targets",
                "pair_score": "sum of target-token conditional log-probabilities",
                "target_normalization": "none",
                "includes_eos": False,
                "tie_credit": 0.5,
            },
            "metrics": metrics,
            "sets": set_results,
        }
        if "language" in benchmark:
            report["language"] = benchmark["language"]

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
            file.write("\n")
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
    print(f"Macro pairwise accuracy: {metrics['macro_pairwise_accuracy']:.6f}")
    print(
        "Context retrieval accuracy: "
        f"{metrics['macro_context_retrieval_accuracy']:.6f}"
    )
    print(f"Mean reciprocal rank: {metrics['macro_mean_reciprocal_rank']:.6f}")
    print(f"Perfect-set accuracy: {metrics['perfect_set_accuracy']:.6f}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
