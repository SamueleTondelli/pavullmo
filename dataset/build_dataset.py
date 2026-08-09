"""Build local, pre-tokenized clean_mc4_it datasets.

The training artifacts are deterministic prefixes of the ``tiny`` training
split containing exactly 10M, 100M, and 1B tokens. Every source document is
encoded as ``[BOS, *content_tokens, EOS]`` before it is appended to the token
stream. A size boundary can therefore cut through the final document; this is
an artifact boundary, not a document boundary, and is recorded in metadata.

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
import shutil
import sys
from typing import Iterable, Sequence

from datasets import load_dataset
import sentencepiece as spm
from tqdm import tqdm


DATASET_NAME = "gsarti/clean_mc4_it"
DATASET_VARIANT = "tiny"
TRAIN_TARGETS = {
    "10m": 10_000_000,
    "100m": 100_000_000,
    "1b": 1_000_000_000,
}
DEFAULT_SHARD_TOKENS = 50_000_000  # 100 MB per full uint16 shard.
UINT16_MAX = 65_535

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER = PROJECT_ROOT / "src" / "tokenizer" / "tokenizer.model"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "ds"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-tokenize 10M, 100M, and 1B-token training prefixes and the "
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
        "--shard-tokens",
        type=int,
        default=DEFAULT_SHARD_TOKENS,
        help=(
            "maximum tokens per binary shard "
            f"(default: {DEFAULT_SHARD_TOKENS:,})"
        ),
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
            raise ValueError(f"refusing to manage path outside output directory: {path}")

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
                raise FileExistsError(f"artifact appeared while building: {self.final_dir}")
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


def encode_document(
    tokenizer: spm.SentencePieceProcessor, text: str
) -> list[int]:
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
) -> None:
    builders = {
        name: ArtifactBuilder(output_dir, f"train_{name}", shard_tokens, overwrite)
        for name in TRAIN_TARGETS
    }
    completed_at_boundary = {name: False for name in TRAIN_TARGETS}
    documents_seen_at_completion = {name: 0 for name in TRAIN_TARGETS}
    nonempty_seen_at_completion = {name: 0 for name in TRAIN_TARGETS}
    documents_seen = 0
    nonempty_documents_seen = 0

    progress = tqdm(
        total=max(TRAIN_TARGETS.values()),
        desc="Tokenizing tiny train",
        unit="tok",
        unit_scale=True,
    )
    for document in dataset_stream("train", revision):
        documents_seen += 1
        text = document.get("text")
        if not isinstance(text, str) or not text.strip():
            continue

        nonempty_documents_seen += 1
        tokens = encode_document(tokenizer, text)
        previous_largest_count = builders["1b"].writer.total_tokens

        for name, target in TRAIN_TARGETS.items():
            builder = builders[name]
            remaining = target - builder.writer.total_tokens
            if remaining <= 0:
                continue

            take = min(remaining, len(tokens))
            builder.writer.write(tokens[:take])
            if builder.writer.total_tokens == target:
                completed_at_boundary[name] = take == len(tokens)
                documents_seen_at_completion[name] = documents_seen
                nonempty_seen_at_completion[name] = nonempty_documents_seen

        progress.update(builders["1b"].writer.total_tokens - previous_largest_count)
        if builders["1b"].writer.total_tokens == TRAIN_TARGETS["1b"]:
            break
    progress.close()

    largest_count = builders["1b"].writer.total_tokens
    if largest_count != TRAIN_TARGETS["1b"]:
        raise RuntimeError(
            "training stream ended after "
            f"{largest_count:,} tokens, before the 1,000,000,000-token target"
        )

    for name, target in TRAIN_TARGETS.items():
        builder = builders[name]
        builder.finish(
            {
                **common_metadata(
                    tokenizer,
                    tokenizer_path,
                    tokenizer_sha256,
                    "train",
                    revision,
                ),
                "target_token_count": target,
                "source_documents_seen": documents_seen_at_completion[name],
                "nonempty_source_documents_seen": nonempty_seen_at_completion[name],
                "ends_at_document_boundary": completed_at_boundary[name],
                "is_prefix_of": "train_1b" if name != "1b" else None,
            }
        )
        print(f"Built {builder.final_dir}: {target:,} tokens")


def build_validation_artifact(
    tokenizer: spm.SentencePieceProcessor,
    tokenizer_path: Path,
    tokenizer_sha256: str,
    output_dir: Path,
    shard_tokens: int,
    revision: str,
    overwrite: bool,
) -> None:
    builder = ArtifactBuilder(output_dir, "validation", shard_tokens, overwrite)
    documents_seen = 0
    documents_written = 0

    for document in tqdm(
        dataset_stream("validation", revision),
        desc="Tokenizing tiny validation",
        unit="docs",
    ):
        documents_seen += 1
        text = document.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        builder.writer.write(encode_document(tokenizer, text))
        documents_written += 1

    builder.finish(
        {
            **common_metadata(
                tokenizer,
                tokenizer_path,
                tokenizer_sha256,
                "validation",
                revision,
            ),
            "source_documents_seen": documents_seen,
            "documents_written": documents_written,
            "ends_at_document_boundary": True,
        }
    )
    print(
        f"Built {builder.final_dir}: {builder.writer.total_tokens:,} tokens "
        f"from {documents_written:,} documents"
    )


def main() -> None:
    args = parse_args()
    if args.shard_tokens <= 0:
        raise ValueError("--shard-tokens must be greater than zero")
    if not args.tokenizer.is_file():
        raise FileNotFoundError(f"tokenizer model not found: {args.tokenizer}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    validate_tokenizer(tokenizer)
    tokenizer_sha256 = sha256_file(args.tokenizer)

    build_training_artifacts(
        tokenizer=tokenizer,
        tokenizer_path=args.tokenizer,
        tokenizer_sha256=tokenizer_sha256,
        output_dir=args.output_dir,
        shard_tokens=args.shard_tokens,
        revision=args.dataset_revision,
        overwrite=args.overwrite,
    )
    build_validation_artifact(
        tokenizer=tokenizer,
        tokenizer_path=args.tokenizer,
        tokenizer_sha256=tokenizer_sha256,
        output_dir=args.output_dir,
        shard_tokens=args.shard_tokens,
        revision=args.dataset_revision,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
