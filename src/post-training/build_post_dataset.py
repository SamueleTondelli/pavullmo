"""Build deterministic supervised fine-tuning splits from JSONL conversations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch

from sft_data import encode_example, validate_messages


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_SOURCE = SCRIPT_DIR / "data" / "sft_examples.jsonl"
DEFAULT_TOKENIZER = PROJECT_DIR / "src" / "tokenizer" / "16k" / "tokenizer.model"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "dataset"
SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize JSONL conversations and build train/validation/test SFT splits."
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--source",
        type=Path,
        help="one JSONL file to split deterministically (defaults to the trial data)",
    )
    inputs.add_argument(
        "--split-source-dir",
        type=Path,
        help="directory containing train.jsonl, validation.jsonl, and test.jsonl",
    )
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--validation-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-length", type=int, default=1024)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_examples(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on {path}:{line_number}: {error}") from error
            example_id = example.get("id")
            messages = example.get("messages")
            if not isinstance(example_id, str) or not example_id.strip():
                raise ValueError(f"{path}:{line_number} requires a non-empty string id")
            if example_id in seen_ids:
                raise ValueError(f"duplicate example id {example_id!r}")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{example_id!r} requires a non-empty messages list")
            validate_messages(example_id, messages)
            seen_ids.add(example_id)
            examples.append(example)
    if not examples:
        raise ValueError(f"no examples found in {path}")
    return examples


def split_examples(
    examples: list[dict[str, Any]], train_count: int, validation_count: int, seed: int
) -> dict[str, list[dict[str, Any]]]:
    if train_count <= 0 or validation_count <= 0:
        raise ValueError("train and validation counts must be positive")
    if train_count + validation_count >= len(examples):
        raise ValueError("at least one example must remain for the test split")
    ordered = sorted(
        examples,
        key=lambda item: hashlib.sha256(
            f"{seed}:{item['id']}".encode("utf-8")
        ).digest(),
    )
    return {
        "train": ordered[:train_count],
        "validation": ordered[train_count : train_count + validation_count],
        "test": ordered[train_count + validation_count :],
    }


def read_publisher_splits(directory: Path) -> dict[str, list[dict[str, Any]]]:
    splits = {name: read_examples(directory / f"{name}.jsonl") for name in SPLITS}
    seen: dict[str, str] = {}
    for split_name, examples in splits.items():
        for example in examples:
            example_id = example["id"]
            if example_id in seen:
                raise ValueError(
                    f"duplicate id {example_id!r} across {seen[example_id]} and {split_name}"
                )
            seen[example_id] = split_name
    return splits


def main() -> None:
    args = parse_args()
    source = (args.source or DEFAULT_SOURCE).expanduser().resolve()
    split_source_dir = (
        args.split_source_dir.expanduser().resolve()
        if args.split_source_dir is not None
        else None
    )
    tokenizer_path = args.tokenizer.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.context_length <= 1:
        raise ValueError("--context-length must be greater than one")
    if split_source_dir is None and not source.is_file():
        raise FileNotFoundError(source)
    if split_source_dir is not None and not split_source_dir.is_dir():
        raise FileNotFoundError(split_source_dir)
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)

    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if min(tokenizer.bos_id(), tokenizer.eos_id(), tokenizer.pad_id()) < 0:
        raise ValueError("tokenizer must define BOS, EOS, and PAD tokens")

    if split_source_dir is not None:
        splits = read_publisher_splits(split_source_dir)
        split_strategy = "publisher_provided"
        source_files = {
            name: {
                "path": str(split_source_dir / f"{name}.jsonl"),
                "sha256": sha256_file(split_source_dir / f"{name}.jsonl"),
            }
            for name in SPLITS
        }
        preparation_metadata_path = split_source_dir / "preparation_metadata.json"
        source_preparation = (
            json.loads(preparation_metadata_path.read_text(encoding="utf-8"))
            if preparation_metadata_path.is_file()
            else None
        )
    else:
        raw_examples = read_examples(source)
        splits = split_examples(
            raw_examples, args.train_count, args.validation_count, args.seed
        )
        split_strategy = "deterministic_hash"
        source_files = {"all": {"path": str(source), "sha256": sha256_file(source)}}
        source_preparation = None
    output_dir.mkdir(parents=True, exist_ok=True)
    split_summaries: dict[str, Any] = {}
    for split_name, split_examples_list in splits.items():
        encoded = [
            encode_example(example, tokenizer, args.context_length)
            for example in split_examples_list
        ]
        artifact = {
            "format_version": 1,
            "split": split_name,
            "examples": encoded,
        }
        artifact_path = output_dir / f"{split_name}.pt"
        torch.save(artifact, artifact_path)
        split_summaries[split_name] = {
            "examples": len(encoded),
            "tokens": sum(item["token_ids"].numel() for item in encoded),
            "assistant_targets": sum(item["loss_mask"].sum().item() for item in encoded),
            "minimum_tokens": min(item["token_ids"].numel() for item in encoded),
            "maximum_tokens": max(item["token_ids"].numel() for item in encoded),
            "ids": [item["id"] for item in encoded],
            "file": artifact_path.name,
        }

    metadata = {
        "format_version": 1,
        "source_files": source_files,
        "source_preparation": source_preparation,
        "split_strategy": split_strategy,
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "vocab_size": tokenizer.vocab_size(),
        "special_token_ids": {
            "unk": tokenizer.unk_id(),
            "bos": tokenizer.bos_id(),
            "eos": tokenizer.eos_id(),
            "pad": tokenizer.pad_id(),
            "system": tokenizer.piece_to_id("<system>"),
            "user": tokenizer.piece_to_id("<user>"),
            "assistant": tokenizer.piece_to_id("<assistant>"),
        },
        "context_length": args.context_length,
        "seed": args.seed,
        "splits": split_summaries,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    printable_metadata = {
        **metadata,
        "splits": {
            name: {key: value for key, value in summary.items() if key != "ids"}
            for name, summary in split_summaries.items()
        },
    }
    print(json.dumps(printable_metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
