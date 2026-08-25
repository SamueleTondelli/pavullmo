"""Build local, pre-tokenized clean_mc4_it datasets.

The training artifacts are deterministic prefixes of the ``tiny`` training
split with source-text budgets of 10M, 100M, and 1B UTF-8 bytes. The document
that reaches or exceeds a budget is included in full, so artifacts always end
at a document boundary and may exceed their byte budget by one document. Every
source document is encoded as ``[BOS, *content_tokens, EOS]`` before it is
appended to the token stream.

The complete ``tiny`` validation split is encoded with the same document
boundary policy. Tokens are stored in sharded, little-endian uint16 files so
they can later be memory-mapped and packed at any context length.
"""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Iterable, Sequence

from datasets import load_dataset
import sentencepiece as spm
from tqdm import tqdm


DATASET_NAME = "gsarti/clean_mc4_it"
DATASET_VARIANT = "tiny"
TRAIN_BYTE_BUDGETS = {
    "42m": 42_100_000,  # 10M
    "90m": 90_000_000,
    "195m": 195_000_000,
    "421m": 421_000_000,  # 100M
    "907m": 907_000_000,
    "1b9": 1_954_000_000,
    "4b": 4_210_000_000,  # 1B
}
DEFAULT_SHARD_TOKENS = 50_000_000  # 100 MB per full uint16 shard.
UINT16_MAX = 65_535

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER = PROJECT_ROOT / "src" / "tokenizer" / "tokenizer.model"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "ds"
DATASET_PREFIX_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-tokenize 10M, 100M, and 1B-byte training prefixes and the "
            "complete tiny validation split of clean_mc4_it."
        )
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_TOKENIZER,
        help=f"SentencePiece model (default: {DEFAULT_TOKENIZER})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"artifact directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dataset-prefix",
        default="",
        help=(
            "prefix used to distinguish tokenizer-specific artifacts; for "
            "example, '4k' creates train_4k_10m and validation_4k"
        ),
    )
    parser.add_argument(
        "--shard-tokens",
        type=int,
        default=DEFAULT_SHARD_TOKENS,
        help=(f"maximum tokens per binary shard (default: {DEFAULT_SHARD_TOKENS:,})"),
    )
    parser.add_argument(
        "--dataset-revision",
        default="main",
        help="Hugging Face dataset revision (default: main)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing artifacts inside --output-dir",
    )
    return parser.parse_args()


def validate_dataset_prefix(prefix: str) -> None:
    if prefix and not DATASET_PREFIX_PATTERN.fullmatch(prefix):
        raise ValueError(
            "--dataset-prefix must start with an ASCII letter or digit and "
            "contain only letters, digits, '.', '_', and '-'"
        )


def training_artifact_name(prefix: str, size: str) -> str:
    return f"train_{prefix}_{size}" if prefix else f"train_{size}"


def validation_artifact_name(prefix: str) -> str:
    return f"validation_{prefix}" if prefix else "validation"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def uint16_bytes(token_ids: Sequence[int]) -> bytes:
    """Encode token IDs using the artifact's explicit little-endian format."""

    values = array("H", token_ids)
    if values.itemsize != 2:
        raise RuntimeError("this platform does not provide a 16-bit unsigned short")
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


class ShardedTokenWriter:
    """Write a flat uint16 token stream into fixed-size shards."""

    def __init__(self, directory: Path, shard_tokens: int) -> None:
        self.directory = directory
        self.shard_tokens = shard_tokens
        self.total_tokens = 0
        self._file = None
        self._shard_index = 0
        self._shard_token_count = 0
        self._shard_digest = None
        self._shard_path = None
        self.shards: list[dict[str, int | str]] = []

    def _open_shard(self) -> None:
        self._shard_path = self.directory / f"tokens-{self._shard_index:05d}.bin"
        self._file = self._shard_path.open("wb", buffering=1024 * 1024)
        self._shard_token_count = 0
        self._shard_digest = hashlib.sha256()

    def _close_shard(self) -> None:
        if self._file is None:
            return

        self._file.close()
        assert self._shard_path is not None
        assert self._shard_digest is not None
        self.shards.append(
            {
                "file": self._shard_path.name,
                "tokens": self._shard_token_count,
                "bytes": self._shard_token_count * 2,
                "sha256": self._shard_digest.hexdigest(),
            }
        )
        self._file = None
        self._shard_index += 1

    def write(self, token_ids: Sequence[int]) -> None:
        position = 0
        while position < len(token_ids):
            if self._file is None:
                self._open_shard()

            capacity = self.shard_tokens - self._shard_token_count
            chunk = token_ids[position : position + capacity]
            data = uint16_bytes(chunk)
            self._file.write(data)
            self._shard_digest.update(data)

            written = len(chunk)
            position += written
            self.total_tokens += written
            self._shard_token_count += written

            if self._shard_token_count == self.shard_tokens:
                self._close_shard()

    def close(self) -> None:
        self._close_shard()


class ArtifactBuilder:
    """Own the temporary directory and metadata for one dataset artifact."""

    def __init__(
        self,
        output_root: Path,
        name: str,
        shard_tokens: int,
        overwrite: bool,
    ) -> None:
        self.output_root = output_root.resolve()
        self.final_dir = (output_root / name).resolve()
        self.temporary_dir = (output_root / f".{name}.tmp").resolve()
        self._validate_child(self.final_dir)
        self._validate_child(self.temporary_dir)

        self.overwrite = overwrite
        if self.final_dir.exists() and not overwrite:
            raise FileExistsError(
                f"{self.final_dir} already exists; pass --overwrite to replace it"
            )
        if self.temporary_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{self.temporary_dir} already exists; pass --overwrite to replace it"
                )
            shutil.rmtree(self.temporary_dir)

        self.temporary_dir.mkdir(parents=True)
        self.writer = ShardedTokenWriter(self.temporary_dir, shard_tokens)

    def _validate_child(self, path: Path) -> None:
        if path.parent != self.output_root:
            raise ValueError(
                f"refusing to manage path outside output directory: {path}"
            )

    def finish(self, metadata: dict[str, object]) -> None:
        self.writer.close()
        metadata = {
            "format_version": 1,
            "storage": {
                "dtype": "uint16",
                "endianness": "little",
                "layout": "flat_token_stream",
                "shard_token_limit": self.writer.shard_tokens,
            },
            "token_count": self.writer.total_tokens,
            "shards": self.writer.shards,
            **metadata,
        }
        metadata_path = self.temporary_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if self.final_dir.exists():
            if not self.overwrite:
                raise FileExistsError(
                    f"artifact appeared while building: {self.final_dir}"
                )
            shutil.rmtree(self.final_dir)
        self.temporary_dir.rename(self.final_dir)


def validate_tokenizer(tokenizer: spm.SentencePieceProcessor) -> None:
    if tokenizer.bos_id() != 1:
        raise ValueError(f"expected BOS=1, found BOS={tokenizer.bos_id()}")
    if tokenizer.eos_id() != 2:
        raise ValueError(f"expected EOS=2, found EOS={tokenizer.eos_id()}")
    if tokenizer.pad_id() != 3:
        raise ValueError(f"expected PAD=3, found PAD={tokenizer.pad_id()}")
    if tokenizer.vocab_size() > UINT16_MAX + 1:
        raise ValueError(
            f"vocabulary of {tokenizer.vocab_size():,} tokens does not fit in uint16"
        )


def encode_document(tokenizer: spm.SentencePieceProcessor, text: str) -> list[int]:
    content = tokenizer.encode(text.strip(), out_type=int)
    return [tokenizer.bos_id(), *content, tokenizer.eos_id()]


def dataset_stream(split: str, revision: str) -> Iterable[dict[str, object]]:
    return load_dataset(
        DATASET_NAME,
        DATASET_VARIANT,
        split=split,
        streaming=True,
        revision=revision,
        trust_remote_code=True,
    )


def common_metadata(
    tokenizer: spm.SentencePieceProcessor,
    tokenizer_path: Path,
    tokenizer_sha256: str,
    split: str,
    revision: str,
) -> dict[str, object]:
    return {
        "dataset": {
            "name": DATASET_NAME,
            "variant": DATASET_VARIANT,
            "split": split,
            "revision": revision,
        },
        "tokenizer": {
            "file": str(tokenizer_path.resolve()),
            "sha256": tokenizer_sha256,
            "vocab_size": tokenizer.vocab_size(),
            "special_token_ids": {
                "unk": tokenizer.unk_id(),
                "bos": tokenizer.bos_id(),
                "eos": tokenizer.eos_id(),
                "pad": tokenizer.pad_id(),
            },
        },
        "document_encoding": ["BOS", "content", "EOS"],
    }


def build_training_artifacts(
    tokenizer: spm.SentencePieceProcessor,
    tokenizer_path: Path,
    tokenizer_sha256: str,
    output_dir: Path,
    shard_tokens: int,
    revision: str,
    overwrite: bool,
    dataset_prefix: str = "",
) -> dict[str, int]:
    builders = {
        name: ArtifactBuilder(
            output_dir,
            training_artifact_name(dataset_prefix, name),
            shard_tokens,
            overwrite,
        )
        for name in TRAIN_BYTE_BUDGETS
    }
    source_bytes_at_completion = {name: 0 for name in TRAIN_BYTE_BUDGETS}
    documents_seen_at_completion = {name: 0 for name in TRAIN_BYTE_BUDGETS}
    nonempty_seen_at_completion = {name: 0 for name in TRAIN_BYTE_BUDGETS}
    documents_seen = 0
    nonempty_documents_seen = 0
    source_bytes_seen = 0
    largest_name = max(TRAIN_BYTE_BUDGETS, key=TRAIN_BYTE_BUDGETS.get)
    largest_budget = TRAIN_BYTE_BUDGETS[largest_name]

    progress = tqdm(
        total=largest_budget,
        desc="Reading tiny train",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    )
    for document in dataset_stream("train", revision):
        documents_seen += 1
        text = document.get("text")
        if not isinstance(text, str) or not text.strip():
            continue

        text = text.strip()
        nonempty_documents_seen += 1
        document_bytes = len(text.encode("utf-8"))
        tokens = encode_document(tokenizer, text)
        previous_source_bytes = source_bytes_seen
        source_bytes_seen += document_bytes

        for name, byte_budget in TRAIN_BYTE_BUDGETS.items():
            if source_bytes_at_completion[name] != 0:
                continue

            builders[name].writer.write(tokens)
            if source_bytes_seen >= byte_budget:
                source_bytes_at_completion[name] = source_bytes_seen
                documents_seen_at_completion[name] = documents_seen
                nonempty_seen_at_completion[name] = nonempty_documents_seen

        progress.update(
            min(source_bytes_seen, largest_budget)
            - min(previous_source_bytes, largest_budget)
        )
        if source_bytes_at_completion[largest_name]:
            break
    progress.close()

    if source_bytes_seen < largest_budget:
        raise RuntimeError(
            "training stream ended after "
            f"{source_bytes_seen:,} source bytes, before the "
            f"{largest_budget:,}-byte budget"
        )

    token_counts: dict[str, int] = {}
    for name, byte_budget in TRAIN_BYTE_BUDGETS.items():
        builder = builders[name]
        source_byte_count = source_bytes_at_completion[name]
        builder.finish(
            {
                **common_metadata(
                    tokenizer,
                    tokenizer_path,
                    tokenizer_sha256,
                    "train",
                    revision,
                ),
                "dataset_prefix": dataset_prefix,
                "source_text_byte_budget": byte_budget,
                "source_text_byte_count": source_byte_count,
                "source_text_byte_overshoot": source_byte_count - byte_budget,
                "source_documents_seen": documents_seen_at_completion[name],
                "nonempty_source_documents_seen": nonempty_seen_at_completion[name],
                "ends_at_document_boundary": True,
                "is_prefix_of": (
                    training_artifact_name(dataset_prefix, largest_name)
                    if name != largest_name
                    else None
                ),
            }
        )
        print(
            f"Built {builder.final_dir}: {builder.writer.total_tokens:,} tokens "
            f"from {source_byte_count:,} source bytes "
            f"(budget: {byte_budget:,})"
        )
        token_counts[builder.final_dir.name] = builder.writer.total_tokens

    return token_counts


def build_validation_artifact(
    tokenizer: spm.SentencePieceProcessor,
    tokenizer_path: Path,
    tokenizer_sha256: str,
    output_dir: Path,
    shard_tokens: int,
    revision: str,
    overwrite: bool,
    dataset_prefix: str = "",
) -> int:
    builder = ArtifactBuilder(
        output_dir,
        validation_artifact_name(dataset_prefix),
        shard_tokens,
        overwrite,
    )
    documents_seen = 0
    documents_written = 0
    source_bytes_written = 0

    progress = tqdm(
        desc="Tokenizing tiny validation",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    )
    for document in dataset_stream("validation", revision):
        documents_seen += 1
        text = document.get("text")
        if not isinstance(text, str) or not text.strip():
            continue

        text = text.strip()
        document_bytes = len(text.encode("utf-8"))
        builder.writer.write(encode_document(tokenizer, text))
        documents_written += 1
        source_bytes_written += document_bytes
        progress.update(document_bytes)
    progress.close()

    builder.finish(
        {
            **common_metadata(
                tokenizer,
                tokenizer_path,
                tokenizer_sha256,
                "validation",
                revision,
            ),
            "dataset_prefix": dataset_prefix,
            "source_text_byte_count": source_bytes_written,
            "source_documents_seen": documents_seen,
            "documents_written": documents_written,
            "ends_at_document_boundary": True,
        }
    )
    print(
        f"Built {builder.final_dir}: {builder.writer.total_tokens:,} tokens "
        f"from {documents_written:,} documents"
    )
    return builder.writer.total_tokens


def main() -> None:
    args = parse_args()
    validate_dataset_prefix(args.dataset_prefix)
    if args.shard_tokens <= 0:
        raise ValueError("--shard-tokens must be greater than zero")
    if not args.tokenizer.is_file():
        raise FileNotFoundError(f"tokenizer model not found: {args.tokenizer}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    validate_tokenizer(tokenizer)
    tokenizer_sha256 = sha256_file(args.tokenizer)

    token_counts = build_training_artifacts(
        tokenizer=tokenizer,
        tokenizer_path=args.tokenizer,
        tokenizer_sha256=tokenizer_sha256,
        output_dir=args.output_dir,
        shard_tokens=args.shard_tokens,
        revision=args.dataset_revision,
        overwrite=args.overwrite,
        dataset_prefix=args.dataset_prefix,
    )
    validation_name = validation_artifact_name(args.dataset_prefix)
    token_counts[validation_name] = build_validation_artifact(
        tokenizer=tokenizer,
        tokenizer_path=args.tokenizer,
        tokenizer_sha256=tokenizer_sha256,
        output_dir=args.output_dir,
        shard_tokens=args.shard_tokens,
        revision=args.dataset_revision,
        overwrite=args.overwrite,
        dataset_prefix=args.dataset_prefix,
    )

    print("\nToken counts by split:")
    for split, token_count in token_counts.items():
        print(f"  {split}: {token_count:,} tokens")


if __name__ == "__main__":
    main()
