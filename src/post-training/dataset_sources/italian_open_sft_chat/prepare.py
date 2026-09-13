"""Download and normalize SerFabio89's Italian Open SFT Chat dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import sentencepiece as spm
from datasets import load_dataset
from huggingface_hub import HfApi


SCRIPT_DIR = Path(__file__).resolve().parent
POST_TRAINING_DIR = SCRIPT_DIR.parent.parent
PROJECT_DIR = POST_TRAINING_DIR.parent.parent
sys.path.insert(0, str(POST_TRAINING_DIR))

from sft_data import encode_example, validate_messages


DATASET_NAME = "SerFabio89/italian-open-sft-chat-dataset"
DEFAULT_OUTPUT_DIR = POST_TRAINING_DIR / "data" / "italian_open_sft_chat"
DEFAULT_TOKENIZER = PROJECT_DIR / "src" / "tokenizer" / "16k" / "tokenizer.model"
SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare filtered publisher splits as PavuLLMo-compatible JSONL."
    )
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--revision", help="optional Hugging Face revision or commit")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--length-band", action="append", dest="length_bands")
    parser.add_argument("--source", action="append", dest="sources")
    parser.add_argument(
        "--conversation-mode",
        choices=("single-turn", "any"),
        default="single-turn",
    )
    parser.add_argument(
        "--max-per-split",
        type=int,
        default=0,
        help="optional deterministic prefix limit for development; zero keeps all",
    )
    return parser.parse_args()


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "category": record.get("category"),
        "difficulty": record.get("difficulty"),
        "teacher_role": record.get("teacher_role"),
        "teacher_model": record.get("teacher_model"),
        "teacher_reasoning_used": record.get("teacher_reasoning_used"),
        "length_band": record.get("length_band"),
        "source": record.get("source"),
        "messages": [
            {"role": message["role"], "content": message["content"].strip()}
            for message in record["messages"]
        ],
    }


def main() -> None:
    args = parse_args()
    length_bands = set(args.length_bands or ["short"])
    sources = set(args.sources or ["magpie", "meta"])
    if args.context_length <= 1 or args.max_per_split < 0:
        raise ValueError("context length must exceed one and max-per-split cannot be negative")

    tokenizer_path = args.tokenizer.expanduser().resolve()
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    dataset_info = HfApi().dataset_info(
        repo_id=args.dataset_name,
        revision=args.revision,
    )
    resolved_revision = dataset_info.sha
    dataset = load_dataset(args.dataset_name, revision=resolved_revision)
    missing_splits = [name for name in SPLITS if name not in dataset]
    if missing_splits:
        raise ValueError("dataset is missing splits: " + ", ".join(missing_splits))

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_ids: set[str] = set()
    summaries: dict[str, Any] = {}

    for split_name in SPLITS:
        counters: Counter[str] = Counter()
        kept_tokens = 0
        output_path = output_dir / f"{split_name}.jsonl"
        with output_path.open("w", encoding="utf-8") as output:
            for source_record in dataset[split_name]:
                counters["seen"] += 1
                if source_record.get("length_band") not in length_bands:
                    counters["filtered_length_band"] += 1
                    continue
                if source_record.get("source") not in sources:
                    counters["filtered_source"] += 1
                    continue
                messages = source_record.get("messages")
                if args.conversation_mode == "single-turn" and (
                    not isinstance(messages, list)
                    or [message.get("role") for message in messages]
                    != ["system", "user", "assistant"]
                ):
                    counters["filtered_conversation_shape"] += 1
                    continue
                try:
                    record = normalize_record(dict(source_record))
                    validate_messages(record["id"], record["messages"])
                    encoded = encode_example(record, tokenizer, args.context_length)
                except (KeyError, TypeError, ValueError):
                    counters["filtered_invalid_or_overlength"] += 1
                    continue
                if record["id"] in all_ids:
                    raise ValueError(f"duplicate id across publisher splits: {record['id']!r}")
                all_ids.add(record["id"])
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                counters["kept"] += 1
                kept_tokens += int(encoded["token_ids"].numel())
                if args.max_per_split and counters["kept"] >= args.max_per_split:
                    counters["stopped_at_limit"] = 1
                    break
        summaries[split_name] = {
            **dict(counters),
            "tokens": kept_tokens,
            "path": str(output_path),
        }

    metadata = {
        "dataset": args.dataset_name,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "filters": {
            "length_bands": sorted(length_bands),
            "sources": sorted(sources),
            "conversation_mode": args.conversation_mode,
            "context_length": args.context_length,
            "max_per_split": args.max_per_split,
        },
        "tokenizer": str(tokenizer_path),
        "splits": summaries,
    }
    metadata_path = output_dir / "preparation_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
