"""Build deterministic supervised fine-tuning splits from JSONL conversations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_SOURCE = SCRIPT_DIR / "data" / "sft_examples.jsonl"
DEFAULT_TOKENIZER = PROJECT_DIR / "src" / "tokenizer" / "16k" / "tokenizer.model"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "dataset"
ROLES = ("system", "user", "assistant")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize JSONL conversations and build train/validation/test SFT splits."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
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


def validate_messages(example_id: str, messages: list[object]) -> None:
    previous_role: str | None = None
    assistant_count = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{example_id!r} message {index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in ROLES:
            raise ValueError(f"{example_id!r} message {index} has invalid role {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{example_id!r} message {index} has empty content")
        if role == "system" and index != 0:
            raise ValueError(f"{example_id!r} system message must be first")
        if role == "user" and previous_role not in {None, "system", "assistant"}:
            raise ValueError(f"{example_id!r} has consecutive user messages")
        if role == "assistant" and previous_role != "user":
            raise ValueError(f"{example_id!r} assistant message must follow a user message")
        assistant_count += role == "assistant"
        previous_role = role
    if previous_role != "assistant" or assistant_count == 0:
        raise ValueError(f"{example_id!r} must end with an assistant response")


def encode_example(
    example: dict[str, Any],
    tokenizer: spm.SentencePieceProcessor,
    context_length: int,
) -> dict[str, Any]:
    token_ids = [tokenizer.bos_id()]
    loss_mask = [False]
    text_parts: list[str] = []

    for message in example["messages"]:
        role = message["role"]
        content = message["content"].strip()
        role_id = tokenizer.piece_to_id(f"<{role}>")
        if role_id == tokenizer.unk_id():
            raise ValueError(f"tokenizer does not define <{role}>")
        content_ids = tokenizer.encode(f"\n{content}\n", out_type=int)
        learns_response = role == "assistant"
        token_ids.extend([role_id, *content_ids])
        loss_mask.extend([learns_response] * (1 + len(content_ids)))
        text_parts.append(f"<{role}>\n{content}")

    token_ids.append(tokenizer.eos_id())
    loss_mask.append(True)
    if len(token_ids) > context_length:
        raise ValueError(
            f"{example['id']!r} uses {len(token_ids)} tokens, exceeding "
            f"context length {context_length}"
        )
    if sum(loss_mask) < 2:
        raise ValueError(f"{example['id']!r} has no assistant targets")

    return {
        "id": example["id"],
        "category": example.get("category", "unspecified"),
        "text": "\n".join(text_parts),
        "token_ids": torch.tensor(token_ids, dtype=torch.int32),
        "loss_mask": torch.tensor(loss_mask, dtype=torch.bool),
    }


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


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    tokenizer_path = args.tokenizer.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.context_length <= 1:
        raise ValueError("--context-length must be greater than one")
    if not source.is_file():
        raise FileNotFoundError(source)
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)

    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if min(tokenizer.bos_id(), tokenizer.eos_id(), tokenizer.pad_id()) < 0:
        raise ValueError("tokenizer must define BOS, EOS, and PAD tokens")

    raw_examples = read_examples(source)
    splits = split_examples(
        raw_examples, args.train_count, args.validation_count, args.seed
    )
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
            "ids": [item["id"] for item in encoded],
            "file": artifact_path.name,
        }

    metadata = {
        "format_version": 1,
        "source": str(source),
        "source_sha256": sha256_file(source),
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
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
