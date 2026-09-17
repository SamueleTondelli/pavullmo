"""Build three comparable Italian pretraining mixtures and shared evaluation sets.

This is an intentionally conservative production baseline. It streams four provenance-clean
source families, applies the same transparent text checks to each, reserves
whole source documents for a deterministic shared test set, and writes token
artifacts compatible with ``src/pavullmo/pretrain_base.py``.

The source families are general web, Wikipedia, public-domain books, and
educational PDFs.  ``edu_pdf`` is deliberately not called "academic": a PDF
quality/education classifier does not prove that a document is a paper.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Iterable, Iterator, Mapping

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / "tmp" / "cache" / "huggingface"))

from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq
import sentencepiece as spm
from tqdm import tqdm

try:
    # Running a script inside ``dataset/`` places this directory on sys.path.
    from build_dataset import (
        ArtifactBuilder,
        DEFAULT_SHARD_TOKENS,
        DEFAULT_TOKENIZER,
        encode_document,
        sha256_file,
        validate_tokenizer,
    )
except ModuleNotFoundError:
    # Also support importing the module from the repository root and test code.
    from dataset.build_dataset import (
        ArtifactBuilder,
        DEFAULT_SHARD_TOKENS,
        DEFAULT_TOKENIZER,
        encode_document,
        sha256_file,
        validate_tokenizer,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "tmp" / "datasets"
DEFAULT_TRAIN_TOKENS = 1_000_000_000
DEFAULT_VALIDATION_TOKENS = 10_000_000
DEFAULT_TEST_TOKENS = 10_000_000
DEFAULT_SHUFFLE_SEED = 42
DEFAULT_SHUFFLE_BUFFER = 10_000
DEFAULT_HOLDOUT_PERMILLE = 20
DEFAULT_DOCUMENTS_DIR = Path(__file__).resolve().parents[1] / "tmp" / "documents"
DEFAULT_DOCUMENT_BYTES_PER_TOKEN = 5.0
DEFAULT_DOCUMENT_SHARD_BYTES = 256 * 1024 * 1024

MIN_CHARACTERS = 200
MIN_ALPHABETIC_RATIO = 0.55
MAX_REPEATED_LINE_RATIO = 0.30
MAX_CHUNK_CHARACTERS = 16_000
MAX_CHUNKS_PER_DOCUMENT = 8

WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class SourceSpec:
    key: str
    dataset: str
    config: str | None
    split: str
    revision: str
    description: str
    homepage: str
    shuffle_buffer_cap: int


SOURCES = {
    "web": SourceSpec(
        key="web",
        dataset="HuggingFaceFW/fineweb-2",
        config="ita_Latn",
        split="train",
        revision="main",
        description="general Italian web text",
        homepage="https://huggingface.co/datasets/HuggingFaceFW/fineweb-2",
        shuffle_buffer_cap=256,
    ),
    "wiki": SourceSpec(
        key="wiki",
        dataset="wikimedia/wikipedia",
        config="20231101.it",
        split="train",
        revision="main",
        description="cleaned Italian Wikipedia articles",
        homepage="https://huggingface.co/datasets/wikimedia/wikipedia",
        shuffle_buffer_cap=64,
    ),
    "books": SourceSpec(
        key="books",
        dataset="PleIAs/Italian-PD",
        config=None,
        split="train",
        revision="main",
        description="OCR text from Italian public-domain books",
        homepage="https://huggingface.co/datasets/PleIAs/Italian-PD",
        shuffle_buffer_cap=8,
    ),
    "edu_pdf": SourceSpec(
        key="edu_pdf",
        dataset="HuggingFaceFW/finepdfs-edu",
        config="ita_Latn",
        split="train",
        revision="main",
        description="Italian PDFs selected for educational content",
        homepage="https://huggingface.co/datasets/HuggingFaceFW/finepdfs-edu",
        shuffle_buffer_cap=8,
    ),
}

# Each mapping is a token allocation, not a document allocation.
MIXES: dict[str, dict[str, float]] = {
    "web": {"web": 0.70, "wiki": 0.10, "books": 0.10, "edu_pdf": 0.10},
    "balanced": {"web": 0.40, "wiki": 0.20, "books": 0.20, "edu_pdf": 0.20},
    "knowledge": {"web": 0.20, "wiki": 0.25, "books": 0.25, "edu_pdf": 0.30},
}
TEST_MIX = {"web": 0.40, "wiki": 0.20, "books": 0.20, "edu_pdf": 0.20}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build three 1B-token production mixtures plus shared validation and test sets."
    )
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--documents-only",
        action="store_true",
        help="materialize cleaned Parquet documents without tokenizing them",
    )
    parser.add_argument(
        "--documents-dir",
        type=Path,
        help="read a previously materialized corpus instead of remote sources",
    )
    parser.add_argument(
        "--documents-output-dir",
        type=Path,
        default=DEFAULT_DOCUMENTS_DIR,
        help=f"output for --documents-only (default: {DEFAULT_DOCUMENTS_DIR})",
    )
    parser.add_argument(
        "--document-bytes-per-token",
        type=float,
        default=DEFAULT_DOCUMENT_BYTES_PER_TOKEN,
        help=(
            "raw UTF-8 byte allowance used only to size tokenizer-independent "
            f"document pools (default: {DEFAULT_DOCUMENT_BYTES_PER_TOKEN})"
        ),
    )
    parser.add_argument(
        "--document-shard-bytes",
        type=int,
        default=DEFAULT_DOCUMENT_SHARD_BYTES,
        help="approximate uncompressed bytes per materialized Parquet shard",
    )
    parser.add_argument("--train-tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    parser.add_argument(
        "--validation-tokens", type=int, default=DEFAULT_VALIDATION_TOKENS
    )
    parser.add_argument("--test-tokens", type=int, default=DEFAULT_TEST_TOKENS)
    parser.add_argument("--shard-tokens", type=int, default=DEFAULT_SHARD_TOKENS)
    parser.add_argument("--shuffle-seed", type=int, default=DEFAULT_SHUFFLE_SEED)
    parser.add_argument("--shuffle-buffer", type=int, default=DEFAULT_SHUFFLE_BUFFER)
    parser.add_argument(
        "--holdout-permille",
        type=int,
        default=DEFAULT_HOLDOUT_PERMILLE,
        help=(
            "stable fraction reserved for each of validation and test "
            "(default: 20 each)"
        ),
    )
    parser.add_argument(
        "--mix",
        choices=["all", *MIXES],
        default="all",
        help=(
            "build all training mixtures or only one "
            "(shared validation and test are always built)"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_document_id(source: SourceSpec, row: Mapping[str, object]) -> str:
    for field in ("id", "identifier", "url", "title"):
        value = row.get(field)
        if value is not None and str(value).strip():
            return f"{source.key}:{field}:{value}"
    text = str(row.get("text", ""))
    return f"{source.key}:text:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def stable_bucket(value: str, modulus: int = 1_000) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def normalized_text_hash(text: str) -> str:
    normalized = " ".join(text.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def repeated_line_ratio(text: str) -> float:
    lines = [" ".join(line.casefold().split()) for line in text.splitlines()]
    lines = [line for line in lines if line]
    return (len(lines) - len(set(lines))) / len(lines) if lines else 0.0


def alphabetic_ratio(text: str) -> float:
    visible = sum(not char.isspace() for char in text)
    return sum(char.isalpha() for char in text) / visible if visible else 0.0


def source_metadata_passes(
    source: SourceSpec, row: Mapping[str, object], counters: Counter[str]
) -> bool:
    if source.key == "web":
        score = row.get("language_score")
        if row.get("language") != "ita" or not isinstance(score, (int, float)):
            counters["rejected_language"] += 1
            return False
        if float(score) < 0.98:
            counters["rejected_language"] += 1
            return False
    elif source.key == "edu_pdf":
        score = row.get("full_doc_lid_score")
        if row.get("full_doc_lid") != "ita_Latn" or not isinstance(
            score, (int, float)
        ):
            counters["rejected_language"] += 1
            return False
        if float(score) < 0.90:
            counters["rejected_language"] += 1
            return False
    return True


def paragraph_chunks(text: str, max_characters: int) -> list[str]:
    paragraphs = [" ".join(part.split()) for part in re.split(r"\n\s*\n", text)]
    paragraphs = [part for part in paragraphs if part]
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    def flush() -> None:
        nonlocal current, current_size
        if current:
            chunks.append("\n\n".join(current))
            current = []
            current_size = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_characters:
            flush()
            for start in range(0, len(paragraph), max_characters):
                chunks.append(paragraph[start : start + max_characters])
            continue
        added = len(paragraph) + (2 if current else 0)
        if current and current_size + added > max_characters:
            flush()
        current.append(paragraph)
        current_size += added
    flush()
    return chunks


def evenly_spaced(items: list[str], limit: int) -> list[str]:
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[len(items) // 2]]
    indices = [round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)]
    return [items[index] for index in indices]


def accepted_chunks(
    source: SourceSpec,
    row: Mapping[str, object],
    counters: Counter[str],
) -> Iterator[str]:
    counters["documents_seen"] += 1
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        counters["rejected_empty"] += 1
        return
    if not source_metadata_passes(source, row, counters):
        return
    text = text.strip()
    if len(text) < MIN_CHARACTERS:
        counters["rejected_short"] += 1
        return
    if alphabetic_ratio(text) < MIN_ALPHABETIC_RATIO:
        counters["rejected_low_alphabetic_ratio"] += 1
        return
    if repeated_line_ratio(text) > MAX_REPEATED_LINE_RATIO:
        counters["rejected_repeated_lines"] += 1
        return

    chunks = paragraph_chunks(text, MAX_CHUNK_CHARACTERS)
    chunks = [chunk for chunk in chunks if len(chunk) >= MIN_CHARACTERS]
    if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
        counters["chunks_omitted_by_document_cap"] += (
            len(chunks) - MAX_CHUNKS_PER_DOCUMENT
        )
        chunks = evenly_spaced(chunks, MAX_CHUNKS_PER_DOCUMENT)
    counters["documents_accepted"] += 1
    counters["chunks_accepted"] += len(chunks)
    yield from chunks


def source_rows(source: SourceSpec, seed: int, buffer_size: int) -> Iterable[dict]:
    dataset = load_dataset(
        source.dataset,
        source.config,
        split=source.split,
        streaming=True,
        revision=source.revision,
        trust_remote_code=True,
    )
    # Full books and PDFs can be hundreds of thousands of characters long.
    # Holding 10,000 of them in a shuffle reservoir can exhaust local memory;
    # datasets still shuffles source shards as well as this bounded row buffer.
    effective_buffer = min(buffer_size, source.shuffle_buffer_cap)
    return dataset.shuffle(seed=seed, buffer_size=effective_buffer)


def iter_partition_chunks(
    source: SourceSpec,
    *,
    partition: str,
    holdout_permille: int,
    seed: int,
    buffer_size: int,
    counters: Counter[str],
) -> Iterator[tuple[str, str]]:
    rows = source_rows(source, seed, buffer_size)
    iterator = iter(rows)
    try:
        for row in iterator:
            document_id = stable_document_id(source, row)
            bucket = stable_bucket(document_id)
            assigned_partition = (
                "test"
                if bucket < holdout_permille
                else "validation"
                if bucket < 2 * holdout_permille
                else "train"
            )
            if assigned_partition != partition:
                counters["documents_wrong_partition"] += 1
                continue
            for chunk in accepted_chunks(source, row, counters):
                yield document_id, chunk
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()


MATERIALIZED_SCHEMA = pa.schema(
    [
        ("document_id", pa.string()),
        ("chunk_index", pa.int32()),
        ("source", pa.string()),
        ("text_sha256", pa.string()),
        ("text", pa.string()),
    ]
)


class MaterializedParquetWriter:
    """Write bounded Parquet shards while tracking raw text bytes."""

    def __init__(self, directory: Path, shard_bytes: int) -> None:
        self.directory = directory
        self.shard_bytes = shard_bytes
        self.directory.mkdir(parents=True)
        self.total_text_bytes = 0
        self.total_chunks = 0
        self.files: list[dict[str, object]] = []
        self._index = 0
        self._writer: pq.ParquetWriter | None = None
        self._path: Path | None = None
        self._shard_text_bytes = 0
        self._shard_chunks = 0
        self._pending: list[dict[str, object]] = []

    def _open(self) -> None:
        self._path = self.directory / f"part-{self._index:05d}.parquet"
        self._writer = pq.ParquetWriter(
            self._path, MATERIALIZED_SCHEMA, compression="zstd"
        )
        self._shard_text_bytes = 0
        self._shard_chunks = 0

    def _flush_pending(self) -> None:
        if not self._pending:
            return
        assert self._writer is not None
        self._writer.write_table(
            pa.Table.from_pylist(self._pending, schema=MATERIALIZED_SCHEMA)
        )
        self._pending.clear()

    def _close(self) -> None:
        if self._writer is None:
            return
        self._flush_pending()
        self._writer.close()
        assert self._path is not None
        self.files.append(
            {
                "file": self._path.name,
                "chunks": self._shard_chunks,
                "text_bytes": self._shard_text_bytes,
                "parquet_bytes": self._path.stat().st_size,
                "sha256": sha256_file(self._path),
            }
        )
        self._writer = None
        self._path = None
        self._index += 1

    def write(self, row: dict[str, object], text_bytes: int) -> None:
        if self._writer is None:
            self._open()
        self._pending.append(row)
        self._shard_text_bytes += text_bytes
        self._shard_chunks += 1
        self.total_text_bytes += text_bytes
        self.total_chunks += 1
        if len(self._pending) >= 1_000:
            self._flush_pending()
        if self._shard_text_bytes >= self.shard_bytes:
            self._close()

    def close(self) -> None:
        self._close()


def document_byte_targets(
    train_tokens: int,
    validation_tokens: int,
    test_tokens: int,
    selected_mixes: Mapping[str, Mapping[str, float]],
    bytes_per_token: float,
) -> dict[str, dict[str, int]]:
    max_train_weights = {
        key: max(weights[key] for weights in selected_mixes.values())
        for key in SOURCES
    }
    token_targets = {
        "train": {
            key: math.ceil(train_tokens * max_train_weights[key]) for key in SOURCES
        },
        "validation": token_allocations(validation_tokens, TEST_MIX),
        "test": token_allocations(test_tokens, TEST_MIX),
    }
    return {
        partition: {
            key: math.ceil(tokens * bytes_per_token)
            for key, tokens in source_targets.items()
        }
        for partition, source_targets in token_targets.items()
    }


def safe_replace_directory(output_dir: Path, temporary_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(output_dir)
    temporary_dir.rename(output_dir)


def materialize_documents(
    *,
    output_dir: Path,
    byte_targets: Mapping[str, Mapping[str, int]],
    shard_bytes: int,
    holdout_permille: int,
    seed: int,
    buffer_size: int,
    overwrite: bool,
    available_mixes: Mapping[str, Mapping[str, float]],
) -> None:
    output_dir = output_dir.resolve()
    if output_dir in {Path("/"), PROJECT_ROOT.resolve(), output_dir.parent}:
        raise ValueError(f"refusing unsafe documents output directory: {output_dir}")
    temporary_dir = output_dir.with_name(f".{output_dir.name}.tmp")
    if temporary_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{temporary_dir} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)

    manifest_sources: dict[str, dict[str, object]] = {}
    for partition, targets in byte_targets.items():
        manifest_sources[partition] = {}
        for index, (key, target_bytes) in enumerate(targets.items()):
            counters: Counter[str] = Counter()
            writer = MaterializedParquetWriter(
                temporary_dir / partition / key, shard_bytes
            )
            hashes: set[str] = set()
            chunks = iter_partition_chunks(
                SOURCES[key],
                partition=partition,
                holdout_permille=holdout_permille,
                seed=seed + index * 1_009,
                buffer_size=buffer_size,
                counters=counters,
            )
            progress = tqdm(
                total=target_bytes,
                desc=f"Materializing {partition}/{key}",
                unit="B",
                unit_scale=True,
            )
            try:
                for document_id, text in chunks:
                    text_hash = normalized_text_hash(text)
                    if text_hash in hashes:
                        counters["rejected_exact_duplicate"] += 1
                        continue
                    hashes.add(text_hash)
                    encoded_bytes = len(text.encode("utf-8"))
                    writer.write(
                        {
                            "document_id": document_id,
                            "chunk_index": counters["materialized_chunks"],
                            "source": key,
                            "text_sha256": text_hash,
                            "text": text,
                        },
                        encoded_bytes,
                    )
                    counters["materialized_chunks"] += 1
                    progress.update(encoded_bytes)
                    if writer.total_text_bytes >= target_bytes:
                        break
            finally:
                progress.close()
                chunks.close()
                writer.close()
            if writer.total_text_bytes < target_bytes:
                raise RuntimeError(
                    f"source {partition}/{key} ended after "
                    f"{writer.total_text_bytes:,} text bytes; expected at least "
                    f"{target_bytes:,}"
                )
            manifest_sources[partition][key] = {
                "target_text_bytes": target_bytes,
                "text_bytes": writer.total_text_bytes,
                "chunks": writer.total_chunks,
                "files": writer.files,
                "filter_counters": dict(counters),
            }

    manifest = {
        "format_version": 1,
        "description": "cleaned tokenizer-independent Pavullmo document corpus",
        "sources": {
            key: {
                "dataset": source.dataset,
                "config": source.config,
                "split": source.split,
                "revision": source.revision,
                "homepage": source.homepage,
            }
            for key, source in SOURCES.items()
        },
        "mixtures": available_mixes,
        "evaluation_mixture": TEST_MIX,
        "partitioning": {
            "method": "sha256(source_key + stable source document id)",
            "validation_buckets_per_1000": holdout_permille,
            "test_buckets_per_1000": holdout_permille,
        },
        "filtering": {
            "min_characters": MIN_CHARACTERS,
            "min_alphabetic_ratio": MIN_ALPHABETIC_RATIO,
            "max_repeated_line_ratio": MAX_REPEATED_LINE_RATIO,
            "max_chunk_characters": MAX_CHUNK_CHARACTERS,
            "max_chunks_per_source_document": MAX_CHUNKS_PER_DOCUMENT,
            "web_min_language_score": 0.98,
            "edu_pdf_min_full_doc_language_score": 0.90,
        },
        "partitions": manifest_sources,
    }
    (temporary_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    safe_replace_directory(output_dir, temporary_dir, overwrite)
    print(f"Materialized cleaned documents in {output_dir}")


def iter_materialized_chunks(
    documents_dir: Path,
    partition: str,
    source_key: str,
    counters: Counter[str],
) -> Iterator[tuple[str, str]]:
    source_dir = documents_dir / partition / source_key
    files = sorted(source_dir.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no materialized Parquet shards in {source_dir}")
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["document_id", "text"]):
            columns = batch.to_pydict()
            for document_id, text in zip(
                columns["document_id"], columns["text"], strict=True
            ):
                counters["materialized_chunks_read"] += 1
                yield document_id, text


def token_allocations(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    if not math.isclose(sum(weights.values()), 1.0):
        raise ValueError(f"mixture weights must sum to one: {weights}")
    keys = list(weights)
    allocations = {key: int(total * weights[key]) for key in keys}
    allocations[keys[-1]] += total - sum(allocations.values())
    return allocations


def tokenizer_metadata(
    tokenizer: spm.SentencePieceProcessor, path: Path, digest: str
) -> dict[str, object]:
    return {
        "file": str(path.resolve()),
        "sha256": digest,
        "vocab_size": tokenizer.vocab_size(),
        "special_token_ids": {
            "unk": tokenizer.unk_id(),
            "bos": tokenizer.bos_id(),
            "eos": tokenizer.eos_id(),
            "pad": tokenizer.pad_id(),
        },
    }


def build_artifact(
    *,
    name: str,
    weights: Mapping[str, float],
    total_tokens: int,
    partition: str,
    heldout_text_hashes: set[str],
    tokenizer: spm.SentencePieceProcessor,
    tokenizer_path: Path,
    tokenizer_digest: str,
    output_dir: Path,
    shard_tokens: int,
    overwrite: bool,
    holdout_permille: int,
    seed: int,
    buffer_size: int,
    documents_dir: Path | None,
) -> set[str]:
    target_by_source = token_allocations(total_tokens, weights)
    written_by_source = Counter[str]()
    counters = {key: Counter() for key in weights}
    exact_hashes: set[str] = set()
    test_hashes = set(heldout_text_hashes)
    builder = ArtifactBuilder(output_dir, name, shard_tokens, overwrite)
    progress = tqdm(total=total_tokens, desc=f"Building {name}", unit="tok", unit_scale=True)
    ends_at_boundary = True
    try:
        # Only one remote stream is kept open at a time. The training DataLoader
        # subsequently shuffles fixed-size token blocks across the artifact.
        for index, key in enumerate(weights):
            iterator = (
                iter_materialized_chunks(
                    documents_dir, partition, key, counters[key]
                )
                if documents_dir is not None
                else iter_partition_chunks(
                    SOURCES[key],
                    partition=partition,
                    holdout_permille=holdout_permille,
                    seed=seed + index * 1_009,
                    buffer_size=buffer_size,
                    counters=counters[key],
                )
            )
            try:
                while written_by_source[key] < target_by_source[key]:
                    try:
                        _document_id, text = next(iterator)
                    except StopIteration as error:
                        raise RuntimeError(
                            f"source {key!r} ended before its "
                            f"{target_by_source[key]:,}-token quota"
                        ) from error

                    text_hash = normalized_text_hash(text)
                    if text_hash in exact_hashes:
                        counters[key]["rejected_exact_duplicate"] += 1
                        continue
                    if text_hash in test_hashes:
                        counters[key]["rejected_evaluation_overlap"] += 1
                        continue
                    exact_hashes.add(text_hash)

                    tokens = encode_document(tokenizer, text)
                    remaining_source = target_by_source[key] - written_by_source[key]
                    take = min(len(tokens), remaining_source)
                    builder.writer.write(tokens[:take])
                    written_by_source[key] += take
                    progress.update(take)
                    if take < len(tokens):
                        ends_at_boundary = False
            finally:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()
    finally:
        progress.close()
        # Do not force collection here. Hugging Face streaming may still have
        # Arrow callbacks in background C++ threads; collecting their wrappers
        # between artifacts can trigger a CPython 3.12 thread-state abort.

    builder.finish(
        {
            "dataset": {
                "name": "pavullmo_production_mixture",
                "split": partition,
                "mixture": (
                    name.removeprefix("train_")
                    if partition == "train"
                    else "shared_evaluation"
                ),
                "weights": dict(weights),
                "target_tokens_by_source": target_by_source,
                "tokens_by_source": dict(written_by_source),
                "sources": {
                    key: {
                        "dataset": SOURCES[key].dataset,
                        "config": SOURCES[key].config,
                        "split": SOURCES[key].split,
                        "revision": SOURCES[key].revision,
                        "description": SOURCES[key].description,
                        "homepage": SOURCES[key].homepage,
                        "shuffle_buffer_cap": SOURCES[key].shuffle_buffer_cap,
                    }
                    for key in weights
                },
            },
            "tokenizer": tokenizer_metadata(tokenizer, tokenizer_path, tokenizer_digest),
            "document_encoding": ["BOS", "content", "EOS"],
            "filtering": {
                "min_characters": MIN_CHARACTERS,
                "min_alphabetic_ratio": MIN_ALPHABETIC_RATIO,
                "max_repeated_line_ratio": MAX_REPEATED_LINE_RATIO,
                "max_chunk_characters": MAX_CHUNK_CHARACTERS,
                "max_chunks_per_source_document": MAX_CHUNKS_PER_DOCUMENT,
                "web_min_language_score": 0.98,
                "edu_pdf_min_full_doc_language_score": 0.90,
                "exact_deduplication": "within artifact after whitespace normalization",
                "near_deduplication": "upstream source processing only",
            },
            "partitioning": {
                "method": "sha256(source_key + stable source document id)",
                "validation_buckets_per_1000": holdout_permille,
                "test_buckets_per_1000": holdout_permille,
                "whole_source_documents_kept_in_one_partition": True,
                "exact_evaluation_text_excluded": bool(test_hashes),
            },
            "source_filter_counters": {
                key: dict(value) for key, value in counters.items()
            },
            "source_unique_chunks": {
                "all_sources": len(exact_hashes),
            },
            "ends_at_document_boundary": ends_at_boundary,
            "stream_shuffle": {
                "seed": seed,
                "artifact_source_order": list(weights),
                "training_loader_shuffles_token_blocks": partition == "train",
                "requested_buffer_size": buffer_size,
                "effective_buffer_size_by_source": {
                    key: min(buffer_size, SOURCES[key].shuffle_buffer_cap)
                    for key in weights
                },
            },
            "materialized_documents": (
                str(documents_dir.resolve()) if documents_dir is not None else None
            ),
        }
    )
    print(f"Built {builder.final_dir}: {builder.writer.total_tokens:,} tokens")
    return exact_hashes if partition != "train" else test_hashes


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "train_tokens",
        "validation_tokens",
        "test_tokens",
        "shard_tokens",
        "shuffle_buffer",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.document_bytes_per_token <= 0:
        raise ValueError("--document-bytes-per-token must be positive")
    if args.document_shard_bytes <= 0:
        raise ValueError("--document-shard-bytes must be positive")
    if not 1 <= args.holdout_permille < 500:
        raise ValueError("--holdout-permille must be between 1 and 499")
    if args.documents_only and args.documents_dir is not None:
        raise ValueError("--documents-only and --documents-dir cannot be combined")
    if not args.documents_only and not args.tokenizer.is_file():
        raise FileNotFoundError(f"tokenizer model not found: {args.tokenizer}")
    if args.documents_dir is not None:
        manifest = args.documents_dir / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"materialized corpus manifest not found: {manifest}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    selected = MIXES if args.mix == "all" else {args.mix: MIXES[args.mix]}
    if args.documents_only:
        targets = document_byte_targets(
            args.train_tokens,
            args.validation_tokens,
            args.test_tokens,
            selected,
            args.document_bytes_per_token,
        )
        materialize_documents(
            output_dir=args.documents_output_dir,
            byte_targets=targets,
            shard_bytes=args.document_shard_bytes,
            holdout_permille=args.holdout_permille,
            seed=args.shuffle_seed,
            buffer_size=args.shuffle_buffer,
            overwrite=args.overwrite,
            available_mixes=selected,
        )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    validate_tokenizer(tokenizer)
    tokenizer_digest = sha256_file(args.tokenizer)

    validation_hashes = build_artifact(
        name="validation",
        weights=TEST_MIX,
        total_tokens=args.validation_tokens,
        partition="validation",
        heldout_text_hashes=set(),
        tokenizer=tokenizer,
        tokenizer_path=args.tokenizer,
        tokenizer_digest=tokenizer_digest,
        output_dir=args.output_dir,
        shard_tokens=args.shard_tokens,
        overwrite=args.overwrite,
        holdout_permille=args.holdout_permille,
        seed=args.shuffle_seed,
        buffer_size=args.shuffle_buffer,
        documents_dir=args.documents_dir,
    )
    test_hashes = build_artifact(
        name="test",
        weights=TEST_MIX,
        total_tokens=args.test_tokens,
        partition="test",
        heldout_text_hashes=validation_hashes,
        tokenizer=tokenizer,
        tokenizer_path=args.tokenizer,
        tokenizer_digest=tokenizer_digest,
        output_dir=args.output_dir,
        shard_tokens=args.shard_tokens,
        overwrite=args.overwrite,
        holdout_permille=args.holdout_permille,
        seed=args.shuffle_seed,
        buffer_size=args.shuffle_buffer,
        documents_dir=args.documents_dir,
    )
    evaluation_hashes = validation_hashes | test_hashes
    for mix_name, weights in selected.items():
        build_artifact(
            name=f"train_{mix_name}",
            weights=weights,
            total_tokens=args.train_tokens,
            partition="train",
            heldout_text_hashes=evaluation_hashes,
            tokenizer=tokenizer,
            tokenizer_path=args.tokenizer,
            tokenizer_digest=tokenizer_digest,
            output_dir=args.output_dir,
            shard_tokens=args.shard_tokens,
            overwrite=args.overwrite,
            holdout_permille=args.holdout_permille,
            seed=args.shuffle_seed,
            buffer_size=args.shuffle_buffer,
            documents_dir=args.documents_dir,
        )


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
