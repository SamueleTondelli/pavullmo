"""Sample source documents and build local, pre-tokenized datasets.

FineWeb2 Italian is the default source. Only documents with a GlotLID language
score of at least 0.98 are used. The legacy ``clean_mc4_it`` source remains
available through ``--source clean_mc4_it``.

``--sample-only`` writes an inspectable Parquet sample before tokenization. In
artifact-building mode, the training artifacts contain exactly 10M, 100M, and
1B tokens. Every document is encoded as ``[BOS, *content_tokens, EOS]``. A
size boundary can therefore cut through the final document; this is recorded
in the artifact metadata. Tokens are stored in sharded little-endian uint16
files so they can later be memory-mapped at any context length.
"""

from __future__ import annotations

import argparse
from array import array
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Iterable, Iterator, Mapping, Sequence

from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq
import sentencepiece as spm
from tqdm import tqdm


DATASET_NAME = "gsarti/clean_mc4_it"
DATASET_VARIANT = "tiny"
FINEWEB2_DATASET_NAME = "HuggingFaceFW/fineweb-2"
FINEWEB2_DATASET_VARIANT = "ita_Latn"
DEFAULT_SOURCE = "fineweb2"
DEFAULT_LANGUAGE_SCORE = 0.98
DEFAULT_SAMPLE_DOCUMENTS = 10_000
DEFAULT_SHUFFLE_SEED = 42
DEFAULT_SHUFFLE_BUFFER = 10_000
DEFAULT_SAMPLE_SOURCE_SHARDS = 85
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
DEFAULT_SAMPLE_OUTPUT = Path(__file__).resolve().parent / "fineweb2_sample.parquet"


@dataclass(frozen=True)
class SourceConfig:
    """Hugging Face source details and its split naming."""

    name: str
    variant: str
    train_split: str
    validation_split: str
    language: str | None = None
    language_score_field: str | None = None


SOURCES = {
    "fineweb2": SourceConfig(
        name=FINEWEB2_DATASET_NAME,
        variant=FINEWEB2_DATASET_VARIANT,
        train_split="train",
        validation_split="test",
        language="ita",
        language_score_field="language_score",
    ),
    "clean_mc4_it": SourceConfig(
        name=DATASET_NAME,
        variant=DATASET_VARIANT,
        train_split="train",
        validation_split="validation",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample source documents as Parquet or pre-tokenize 10M, 100M, "
            "and 1B-token training datasets plus validation."
        )
    )
    parser.add_argument(
        "--source",
        choices=sorted(SOURCES),
        default=DEFAULT_SOURCE,
        help=(
            f"source dataset (default: {DEFAULT_SOURCE}); clean_mc4_it keeps "
            "the previous source available"
        ),
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
        "--min-language-score",
        type=float,
        default=DEFAULT_LANGUAGE_SCORE,
        help=(
            "minimum source language confidence for sources that provide it "
            f"(default: {DEFAULT_LANGUAGE_SCORE})"
        ),
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=DEFAULT_SHUFFLE_SEED,
        help=f"seed used for streaming shuffle (default: {DEFAULT_SHUFFLE_SEED})",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=DEFAULT_SHUFFLE_BUFFER,
        help=(
            "documents held by the approximate streaming shuffle "
            f"(default: {DEFAULT_SHUFFLE_BUFFER:,})"
        ),
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="write an inspectable Parquet sample instead of token artifacts",
    )
    parser.add_argument(
        "--sample-documents",
        type=int,
        default=DEFAULT_SAMPLE_DOCUMENTS,
        help=(
            "accepted documents in the Parquet sample "
            f"(default: {DEFAULT_SAMPLE_DOCUMENTS:,})"
        ),
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=DEFAULT_SAMPLE_OUTPUT,
        help=f"Parquet sample path (default: {DEFAULT_SAMPLE_OUTPUT})",
    )
    parser.add_argument(
        "--sample-source-shards",
        type=int,
        default=DEFAULT_SAMPLE_SOURCE_SHARDS,
        help=(
            "source files represented in a Parquet sample "
            f"(default: {DEFAULT_SAMPLE_SOURCE_SHARDS})"
        ),
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
    """Return the legacy clean_mc4_it stream.

    This function intentionally retains its original signature and behaviour.
    New code should use :func:`source_stream`.
    """
    return load_dataset(
        DATASET_NAME,
        DATASET_VARIANT,
        split=split,
        streaming=True,
        revision=revision,
        trust_remote_code=True,
    )


def source_stream(
    source: SourceConfig,
    split: str,
    revision: str,
    *,
    shuffle_seed: int | None = None,
    shuffle_buffer: int = DEFAULT_SHUFFLE_BUFFER,
) -> Iterable[dict[str, object]]:
    """Stream a configured source, optionally with an approximate shuffle."""

    dataset = load_dataset(
        source.name,
        source.variant,
        split=split,
        streaming=True,
        revision=revision,
        trust_remote_code=True,
    )
    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer)
    return dataset


def iter_accepted_documents(
    documents: Iterable[Mapping[str, object]],
    source: SourceConfig,
    min_language_score: float | None,
    counters: dict[str, int] | None = None,
) -> Iterator[dict[str, object]]:
    """Yield nonempty, target-language documents passing source confidence."""

    stats = counters if counters is not None else {}
    for original in documents:
        stats["documents_seen"] = stats.get("documents_seen", 0) + 1
        text = original.get("text")
        if not isinstance(text, str) or not text.strip():
            stats["rejected_empty"] = stats.get("rejected_empty", 0) + 1
            continue

        if source.language is not None and original.get("language") != source.language:
            stats["rejected_language"] = stats.get("rejected_language", 0) + 1
            continue

        if source.language_score_field is not None and min_language_score is not None:
            score = original.get(source.language_score_field)
            if not isinstance(score, (int, float)) or score < min_language_score:
                stats["rejected_language_score"] = (
                    stats.get("rejected_language_score", 0) + 1
                )
                continue

        stats["documents_accepted"] = stats.get("documents_accepted", 0) + 1
        yield dict(original)


def close_iterator(iterator: object) -> None:
    """Close a streaming iterator when its implementation supports cleanup."""

    close = getattr(iterator, "close", None)
    if callable(close):
        close()


def export_parquet_sample(
    source: SourceConfig,
    revision: str,
    output_path: Path,
    document_count: int,
    min_language_score: float | None,
    shuffle_seed: int,
    shuffle_buffer: int,
    overwrite: bool,
    sample_source_shards: int = DEFAULT_SAMPLE_SOURCE_SHARDS,
) -> None:
    """Write a deterministic streaming sample while preserving source fields."""

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    if temporary_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{temporary_path} already exists; pass --overwrite to replace it"
            )
        temporary_path.unlink()

    counters: dict[str, int] = {}
    documents = source_stream(source, source.train_split, revision)
    available_shards = getattr(documents, "num_shards", 1)
    represented_shards = min(
        available_shards, sample_source_shards, document_count
    )
    shard_indices = list(range(available_shards))
    random.Random(shuffle_seed).shuffle(shard_indices)
    shard_indices = shard_indices[:represented_shards]
    base_quota, extra = divmod(document_count, represented_shards)

    rows: list[dict[str, object]] = []
    progress = tqdm(total=document_count, desc="Sampling documents", unit="docs")
    try:
        for position, shard_index in enumerate(shard_indices):
            quota = base_quota + (position < extra)
            shard = documents.shard(available_shards, shard_index).shuffle(
                seed=shuffle_seed + shard_index,
                buffer_size=shuffle_buffer,
            )
            shard_iterator = iter(shard)
            accepted_documents = iter_accepted_documents(
                shard_iterator, source, min_language_score, counters
            )
            accepted_from_shard = 0
            try:
                for document in accepted_documents:
                    rows.append(
                        {
                            **document,
                            "source_dataset": source.name,
                            "source_variant": source.variant,
                            "source_revision": revision,
                            "sample_source_shard": shard_index,
                        }
                    )
                    accepted_from_shard += 1
                    progress.update()
                    if accepted_from_shard == quota:
                        break
            finally:
                close_iterator(accepted_documents)
                close_iterator(shard_iterator)
    finally:
        progress.close()
        del documents
        # Hugging Face streaming reads Parquet in background C++ threads. Make
        # their Python wrappers collectible before interpreter finalization.
        gc.collect()

    if len(rows) != document_count:
        raise RuntimeError(
            f"source ended after {len(rows):,} accepted documents; "
            f"expected {document_count:,}"
        )

    table = pa.Table.from_pylist(rows)
    metadata = {
        **(table.schema.metadata or {}),
        b"pavullmo_sample": json.dumps(
            {
                "format_version": 1,
                "dataset": source.name,
                "variant": source.variant,
                "split": source.train_split,
                "revision": revision,
                "sample_documents": document_count,
                "min_language_score": min_language_score,
                "shuffle_seed": shuffle_seed,
                "shuffle_buffer": shuffle_buffer,
                "sampling": "equal_quota_across_source_shards_then_buffer_shuffle",
                "available_source_shards": available_shards,
                "represented_source_shards": represented_shards,
                "source_shard_indices": shard_indices,
                "counters": counters,
            },
            sort_keys=True,
        ).encode("utf-8"),
    }
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, temporary_path, compression="zstd")
    if output_path.exists():
        output_path.unlink()
    temporary_path.rename(output_path)
    print(
        f"Built {output_path}: {document_count:,} documents "
        f"({counters.get('rejected_language_score', 0):,} below language threshold)"
    )


def common_metadata(
    tokenizer: spm.SentencePieceProcessor,
    tokenizer_path: Path,
    tokenizer_sha256: str,
    split: str,
    revision: str,
    source: SourceConfig | None = None,
    min_language_score: float | None = None,
) -> dict[str, object]:
    source = source or SOURCES["clean_mc4_it"]
    return {
        "dataset": {
            "name": source.name,
            "variant": source.variant,
            "split": split,
            "revision": revision,
            "language": source.language,
            "min_language_score": (
                min_language_score if source.language_score_field else None
            ),
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
    source: SourceConfig | None = None,
    min_language_score: float | None = None,
    shuffle_seed: int | None = None,
    shuffle_buffer: int = DEFAULT_SHUFFLE_BUFFER,
) -> None:
    source = source or SOURCES["clean_mc4_it"]
    builders = {
        name: ArtifactBuilder(output_dir, f"train_{name}", shard_tokens, overwrite)
        for name in TRAIN_TARGETS
    }
    completed_at_boundary = {name: False for name in TRAIN_TARGETS}
    documents_seen_at_completion = {name: 0 for name in TRAIN_TARGETS}
    nonempty_seen_at_completion = {name: 0 for name in TRAIN_TARGETS}
    source_counters_at_completion: dict[str, dict[str, int]] = {}
    nonempty_documents_seen = 0

    progress = tqdm(
        total=max(TRAIN_TARGETS.values()),
        desc=f"Tokenizing {source.variant} train",
        unit="tok",
        unit_scale=True,
    )
    source_counters: dict[str, int] = {}
    documents = source_stream(
        source,
        source.train_split,
        revision,
        shuffle_seed=shuffle_seed,
        shuffle_buffer=shuffle_buffer,
    )
    for document in iter_accepted_documents(
        documents, source, min_language_score, source_counters
    ):
        text = document.get("text")
        assert isinstance(text, str) and text.strip()

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
                documents_seen_at_completion[name] = source_counters["documents_seen"]
                nonempty_seen_at_completion[name] = nonempty_documents_seen
                source_counters_at_completion[name] = dict(source_counters)

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
                    source,
                    min_language_score,
                ),
                "target_token_count": target,
                "source_documents_seen": documents_seen_at_completion[name],
                "nonempty_source_documents_seen": nonempty_seen_at_completion[name],
                "ends_at_document_boundary": completed_at_boundary[name],
                "is_prefix_of": "train_1b" if name != "1b" else None,
                "stream_shuffle": {
                    "seed": shuffle_seed,
                    "buffer_size": shuffle_buffer if shuffle_seed is not None else None,
                },
                "source_filter_counters": source_counters_at_completion[name],
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
    source: SourceConfig | None = None,
    min_language_score: float | None = None,
) -> None:
    source = source or SOURCES["clean_mc4_it"]
    builder = ArtifactBuilder(output_dir, "validation", shard_tokens, overwrite)
    documents_written = 0
    source_counters: dict[str, int] = {}

    for document in tqdm(
        iter_accepted_documents(
            source_stream(source, source.validation_split, revision),
            source,
            min_language_score,
            source_counters,
        ),
        desc="Tokenizing validation",
        unit="docs",
    ):
        text = document.get("text")
        assert isinstance(text, str) and text.strip()
        builder.writer.write(encode_document(tokenizer, text))
        documents_written += 1

    builder.finish(
        {
            **common_metadata(
                tokenizer,
                tokenizer_path,
                tokenizer_sha256,
                source.validation_split,
                revision,
                source,
                min_language_score,
            ),
            "source_documents_seen": source_counters.get("documents_seen", 0),
            "documents_written": documents_written,
            "ends_at_document_boundary": True,
            "source_filter_counters": source_counters,
        }
    )
    print(
        f"Built {builder.final_dir}: {builder.writer.total_tokens:,} tokens "
        f"from {documents_written:,} documents"
    )


def main() -> None:
    # The production multi-source pipeline is now the default interface. Keep
    # the former sampling and single-source modes available for exploration and
    # backwards compatibility when their explicit flags are supplied.
    legacy_mode = "--sample-only" in sys.argv or "--source" in sys.argv
    if not legacy_mode:
        try:
            from production_pipeline import main as production_main
        except ModuleNotFoundError:
            from dataset.production_pipeline import main as production_main

        production_main()
        return

    args = parse_args()
    if args.shard_tokens <= 0:
        raise ValueError("--shard-tokens must be greater than zero")
    if args.sample_documents <= 0:
        raise ValueError("--sample-documents must be greater than zero")
    if args.shuffle_buffer <= 0:
        raise ValueError("--shuffle-buffer must be greater than zero")
    if args.sample_source_shards <= 0:
        raise ValueError("--sample-source-shards must be greater than zero")
    if not 0.0 <= args.min_language_score <= 1.0:
        raise ValueError("--min-language-score must be between zero and one")

    source = SOURCES[args.source]
    source_min_language_score = (
        args.min_language_score if source.language_score_field is not None else None
    )
    if args.sample_only:
        export_parquet_sample(
            source=source,
            revision=args.dataset_revision,
            output_path=args.sample_output,
            document_count=args.sample_documents,
            min_language_score=source_min_language_score,
            shuffle_seed=args.shuffle_seed,
            shuffle_buffer=args.shuffle_buffer,
            overwrite=args.overwrite,
            sample_source_shards=args.sample_source_shards,
        )
        return

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
        source=source,
        min_language_score=source_min_language_score,
        shuffle_seed=args.shuffle_seed,
        shuffle_buffer=args.shuffle_buffer,
    )
    build_validation_artifact(
        tokenizer=tokenizer,
        tokenizer_path=args.tokenizer,
        tokenizer_sha256=tokenizer_sha256,
        output_dir=args.output_dir,
        shard_tokens=args.shard_tokens,
        revision=args.dataset_revision,
        overwrite=args.overwrite,
        source=source,
        min_language_score=source_min_language_score,
    )


if __name__ == "__main__":
    main()
    # datasets 3.x can leave an Arrow callback racing with CPython 3.12
    # finalization after a remote Parquet stream is stopped early. All output
    # files are closed and atomically renamed before reaching this point, so a
    # direct successful process exit avoids that upstream shutdown-only abort.
    # Exceptions from main() still propagate normally and return a failure.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
