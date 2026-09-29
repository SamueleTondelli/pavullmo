"""Build 16-bit token artifacts from the local Italian SYNTH document pool.

Each row becomes one instruction example::

    [BOS, <user>, query, <assistant>, synthetic_answer, EOS]

The English ``synthetic_reasoning`` field is deliberately excluded.  A stable
hash of ``synth_id`` assigns examples to validation before either split is
truncated to its exact token budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
import sentencepiece as spm
from tqdm import tqdm

from build_dataset import ArtifactBuilder, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCUMENTS_DIR = PROJECT_ROOT / "artifacts" / "documents" / "synth_it"
DEFAULT_TOKENIZER = (
    PROJECT_ROOT / "artifacts" / "tokenizers" / "production_16k" / "tokenizer.model"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "datasets"
DEFAULT_TRAIN_TOKENS = 1_000_000_000
DEFAULT_VALIDATION_TOKENS = 10_000_000
DEFAULT_SHARD_TOKENS = 50_000_000
VALIDATION_BUCKETS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents-dir", type=Path, default=DEFAULT_DOCUMENTS_DIR)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-name", default="train_16k_synth_it")
    parser.add_argument("--validation-name", default="validation_16k")
    parser.add_argument("--train-tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    parser.add_argument(
        "--validation-tokens", type=int, default=DEFAULT_VALIDATION_TOKENS
    )
    parser.add_argument("--shard-tokens", type=int, default=DEFAULT_SHARD_TOKENS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def rows(documents_dir: Path) -> Iterator[dict[str, str]]:
    manifest_path = documents_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "PleIAs/SYNTH" or manifest.get("language") == "en":
        raise ValueError(f"unexpected SYNTH manifest: {manifest_path}")
    expected = [record["file"] for record in manifest["files"]]
    actual = sorted(path.name for path in documents_dir.glob("part-*.parquet"))
    if actual != sorted(expected):
        raise ValueError("SYNTH Parquet inventory does not match manifest.json")

    columns = ["synth_id", "language", "query", "synthetic_answer"]
    for name in expected:
        parquet = pq.ParquetFile(documents_dir / name)
        for batch in parquet.iter_batches(batch_size=4096, columns=columns):
            yield from batch.to_pylist()


def is_validation(synth_id: str) -> bool:
    digest = hashlib.sha256(synth_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % VALIDATION_BUCKETS == 0


def encode_example(
    tokenizer: spm.SentencePieceProcessor,
    row: dict[str, str],
    user_id: int,
    assistant_id: int,
) -> list[int]:
    query = (row.get("query") or "").strip()
    answer = (row.get("synthetic_answer") or "").strip()
    if row.get("language") != "it" or not query or not answer:
        return []
    return [
        tokenizer.bos_id(),
        user_id,
        *tokenizer.encode(query, out_type=int),
        assistant_id,
        *tokenizer.encode(answer, out_type=int),
        tokenizer.eos_id(),
    ]


def main() -> None:
    args = parse_args()
    if min(args.train_tokens, args.validation_tokens, args.shard_tokens) <= 0:
        raise ValueError("token and shard budgets must be positive")

    tokenizer_path = args.tokenizer.resolve()
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if tokenizer.vocab_size() != 16_384:
        raise ValueError(
            f"expected a 16,384-token tokenizer, got {tokenizer.vocab_size():,}"
        )
    user_id = tokenizer.piece_to_id("<user>")
    assistant_id = tokenizer.piece_to_id("<assistant>")
    if min(user_id, assistant_id) < 0:
        raise ValueError("tokenizer is missing <user> or <assistant>")

    builders = {
        "train": ArtifactBuilder(
            args.output_dir, args.train_name, args.shard_tokens, args.overwrite
        ),
        "validation": ArtifactBuilder(
            args.output_dir, args.validation_name, args.shard_tokens, args.overwrite
        ),
    }
    targets = {"train": args.train_tokens, "validation": args.validation_tokens}
    documents = {"train": 0, "validation": 0}
    skipped = 0
    progress = tqdm(total=sum(targets.values()), unit="tok", unit_scale=True)

    for row in rows(args.documents_dir.resolve()):
        split = "validation" if is_validation(row["synth_id"]) else "train"
        writer = builders[split].writer
        if writer.total_tokens >= targets[split]:
            if all(
                builders[name].writer.total_tokens >= target
                for name, target in targets.items()
            ):
                break
            continue
        token_ids = encode_example(tokenizer, row, user_id, assistant_id)
        if not token_ids:
            skipped += 1
            continue
        remaining = targets[split] - writer.total_tokens
        take = min(remaining, len(token_ids))
        writer.write(token_ids[:take])
        documents[split] += 1
        progress.update(take)
    progress.close()

    for split, target in targets.items():
        builder = builders[split]
        if builder.writer.total_tokens != target:
            raise RuntimeError(
                f"{split} supplied {builder.writer.total_tokens:,} tokens; "
                f"needed {target:,}"
            )
        builder.finish(
            {
                "dataset": {
                    "name": "PleIAs/SYNTH",
                    "revision": "main",
                    "language": "it",
                    "local_documents": str(args.documents_dir.resolve()),
                    "split_assignment": (
                        "sha256(synth_id) modulo 100; bucket 0 is validation"
                    ),
                },
                "tokenizer": {
                    "file": str(tokenizer_path),
                    "sha256": sha256_file(tokenizer_path),
                    "vocab_size": tokenizer.vocab_size(),
                    "special_token_ids": {
                        "unk": tokenizer.unk_id(),
                        "bos": tokenizer.bos_id(),
                        "eos": tokenizer.eos_id(),
                        "pad": tokenizer.pad_id(),
                        "user": user_id,
                        "assistant": assistant_id,
                    },
                },
                "document_encoding": [
                    "BOS",
                    "<user>",
                    "query",
                    "<assistant>",
                    "synthetic_answer",
                    "EOS",
                ],
                "excluded_fields": ["synthetic_reasoning"],
                "split": split,
                "target_token_count": target,
                "documents_written": documents[split],
                "rows_skipped": skipped,
                "ends_at_document_boundary": False,
            }
        )
        print(f"Built {builder.final_dir}: {target:,} tokens")


if __name__ == "__main__":
    main()
