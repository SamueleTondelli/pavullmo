"""Build three comparable Italian pretraining mixtures and shared evaluation sets.

This is an intentionally conservative production baseline. It streams three provenance-clean
source families, applies the same transparent text checks to each, reserves
whole source documents for a deterministic shared test set, and writes token
artifacts compatible with ``src/pavullmo/pretrain_base.py``.

The source families are general web, Wikipedia, and
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

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / "artifacts" / "cache" / "huggingface"))

from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq
import sentencepiece as spm
from tqdm import tqdm

try:
    # Running a script inside ``src/dataset/`` places this directory on sys.path.
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
    from src.dataset.build_dataset import (
        ArtifactBuilder,
        DEFAULT_SHARD_TOKENS,
        DEFAULT_TOKENIZER,
        encode_document,
        sha256_file,
        validate_tokenizer,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "datasets"
DEFAULT_TRAIN_TOKENS = 1_000_000_000
DEFAULT_VALIDATION_TOKENS = 10_000_000
DEFAULT_TEST_TOKENS = 10_000_000
DEFAULT_SHUFFLE_SEED = 42
DEFAULT_SHUFFLE_BUFFER = 10_000
DEFAULT_HOLDOUT_PERMILLE = 20
DEFAULT_DOCUMENTS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "documents"
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
# Italian-PD removed: preserve relative weights of the three remaining sources.
MIXES: dict[str, dict[str, float]] = {
    "web": {"web": 7 / 9, "wiki": 1 / 9, "edu_pdf": 1 / 9},
    "balanced": {"web": 0.50, "wiki": 0.25, "edu_pdf": 0.25},
    "knowledge": {"web": 4 / 15, "wiki": 5 / 15, "edu_pdf": 6 / 15},
}
TEST_MIX = {"web": 0.50, "wiki": 0.25, "edu_pdf": 0.25}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build three 1B-token production mixtures plus shared validation and test sets.",
        epilog="Commands: prepare-lid, audit-pages, audit-language, audit-prose, sample-pdfs, split-experiment. "
               "Use clean_dataset.py COMMAND --help for options. Audits do not filter production data.",
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
        choices=["all", "selected", *MIXES],
        default="all",
        help=(
            "build all training mixtures or only one "
            "(shared validation and test are always built)"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--raw-documents-dir", type=Path, help="reuse SOURCE/sample.parquet raw pools")
    parser.add_argument("--max-source-documents", type=int, default=100_000,
                        help="maximum downloaded candidate documents per source (not accepted quota)")
    parser.add_argument("--source-document-limits", type=json.loads, default=None,
                        help='per-source candidate overrides as JSON, e.g. {"web": 200000}')
    parser.add_argument("--workers", type=int, default=1, help="parallel quality workers; each loads a ~1.6 GB model")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--language-threshold", type=float, default=.95)
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "artifacts/models/glotlid")
    parser.add_argument("--allow-underfilled-documents", action="store_true",
                        help="allow study pools below requested budgets; token builds still enforce quotas")
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
    # Full PDFs can be hundreds of thousands of characters long.
    # Holding 10,000 of them in a shuffle reservoir can exhaust local memory;
    # datasets still shuffles source shards as well as this bounded row buffer.
    effective_buffer = min(buffer_size, source.shuffle_buffer_cap)
    return dataset.shuffle(seed=seed, buffer_size=effective_buffer)


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


def quality_predictor(model_dir: Path):
    """Load once for every source, using fastText's NumPy-2-compatible list API."""
    try:
        from language_quality import prepare_lid
    except ModuleNotFoundError:
        from src.dataset.language_quality import prepare_lid
    import fasttext
    lock = prepare_lid(model_dir)
    model = fasttext.load_model(lock['path'])

    def predict(text):
        labels, scores = model.predict([text], k=3)
        return [(label.removeprefix('__label__'), float(score))
                for label, score in zip(labels[0], scores[0])]
    return predict, lock


def clean_source_document(source, row, predict, threshold=.95):
    """Shared policy for web, wiki and PDF; no source bypasses local checks."""
    try:
        from language_quality import label_document
        from prose_quality import measure_prose
    except ModuleNotFoundError:
        from src.dataset.language_quality import label_document
        from src.dataset.prose_quality import measure_prose
    counters = Counter()
    chunks = list(accepted_chunks(source, row, counters))
    reasons = [key for key in counters if key.startswith('rejected_')]
    if not chunks:
        return [], dict(reasons=reasons or ['no_usable_chunks'], baseline=dict(counters))
    text = row['text']
    language = label_document(text, predict, threshold=threshold)
    prose = measure_prose(text)
    if not language['would_pass']:
        reasons.append('insufficient_italian_coverage')
    reasons.extend(prose['reasons'])
    signals = dict(reasons=reasons, baseline=dict(counters),
                   language={k:v for k,v in language.items() if k!='blocks'}, prose=prose)
    return ([] if reasons else chunks), signals


_WORKER_PREDICTOR = None


def initialize_quality_worker(model_dir):
    global _WORKER_PREDICTOR
    _WORKER_PREDICTOR, _ = quality_predictor(Path(model_dir))


def quality_worker(task):
    key, row, threshold = task
    chunks, signals = clean_source_document(SOURCES[key], row, _WORKER_PREDICTOR, threshold)
    return row, chunks, signals


def cleaned_rows(rows, source, predictor, threshold, executor=None):
    if executor is None:
        for row in rows:
            chunks, signals = clean_source_document(source,row,predictor,threshold)
            yield row, chunks, signals
        return
    from collections import deque
    pending = deque()
    # Keep memory bounded and consume in input order so dedup winners are stable.
    for row in rows:
        pending.append(executor.submit(quality_worker,(source.key,row,threshold)))
        if len(pending) >= 2 * executor._max_workers:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


class GlobalDeduplicator:
    """Disk-backed global exact matching before any split assignment.

    A duplicate retained chunk rejects the entire later document, keeping its
    other chunks out of different splits. First accepted source/document wins.
    This does not claim near-duplicate or arbitrary substring detection.
    """
    def __init__(self, path, *, resume=False):
        import sqlite3
        if resume and not Path(path).is_file():
            raise FileNotFoundError(path)
        self.db = sqlite3.connect(path)
        if resume:
            for table, columns in [('documents', ['hash', 'source', 'id', 'chunks']),
                                   ('chunks', ['hash', 'document_hash'])]:
                if [r[1] for r in self.db.execute(f'PRAGMA table_info({table})')] != columns:
                    self.db.close()
                    raise ValueError(f'Invalid deduplication index: {table}')
            return
        self.db.execute('CREATE TABLE documents (hash TEXT PRIMARY KEY, source TEXT, id TEXT, chunks TEXT, UNIQUE(source,id))')
        self.db.execute('CREATE TABLE chunks (hash TEXT PRIMARY KEY, document_hash TEXT)')

    def add(self, source, document_id, text, chunks, *, commit=True):
        from contextlib import nullcontext
        document_hash = normalized_text_hash(text)
        if self.db.execute('SELECT 1 FROM documents WHERE hash=?', (document_hash,)).fetchone():
            return False, 'duplicate_document', document_hash
        if self.db.execute('SELECT 1 FROM documents WHERE source=? AND id=?', (source,document_id)).fetchone():
            return False, 'duplicate_document_id', document_hash
        unique = {normalized_text_hash(chunk): chunk for chunk in chunks}
        for chunk_hash in unique:
            if self.db.execute('SELECT 1 FROM chunks WHERE hash=?', (chunk_hash,)).fetchone():
                return False, 'shared_exact_chunk', document_hash
        with self.db if commit else nullcontext():
            self.db.execute('INSERT INTO documents VALUES (?,?,?,?)',
                            (document_hash, source, document_id, json.dumps(list(unique.values()))))
            self.db.executemany('INSERT INTO chunks VALUES (?,?)', [(h,document_hash) for h in unique])
        return True, None, document_hash

    def close(self):
        self.db.close()


def assign_partition(document_hash, seed, holdout_permille):
    if not 1 <= holdout_permille < 500:
        raise ValueError('holdout_permille must be between 1 and 499')
    bucket = stable_bucket(f'{seed}:{document_hash}')
    return 'test' if bucket < holdout_permille else 'validation' if bucket < 2*holdout_permille else 'train'


def verified_raw_rows(directory, key):
    """Read local raw input, checking the optional sample manifest when present."""
    path = directory / key / 'sample.parquet'
    companion = path.with_name('sample.json')
    if companion.exists():
        metadata = json.loads(companion.read_text())
        expected = metadata.get('hashes', {}).get('sample.parquet')
        if expected and sha256_file(path) != expected:
            raise ValueError(f'Raw sample checksum mismatch: {path}')
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
        for row in batch.to_pylist():
            if 'metadata_json' in row:
                data = json.loads(row['metadata_json'])
                data['text'] = row['text']
                if not data.get('id') and row.get('document_id'):
                    data['id'] = row['document_id']
                yield data
            else:
                yield row


def materialize_documents(
    *, output_dir, byte_targets, shard_bytes, holdout_permille, seed,
    buffer_size, overwrite, available_mixes, raw_documents_dir=None,
    max_source_documents=100_000, split_seed=42, allow_underfilled=False,
    model_dir=None, language_threshold=.95, predictor=None, workers=1, source_document_limits=None,
):
    """Freeze -> clean all sources -> global dedup -> split -> materialize.

    Candidate acquisition is finite and explicitly budgeted per source. A
    smaller budget can create underfilled splits; production refuses them.
    """
    from dataclasses import replace
    import itertools
    try:
        from language_quality import LANGUAGE_POLICY
        from prose_quality import POLICY
    except ModuleNotFoundError:
        from src.dataset.language_quality import LANGUAGE_POLICY
        from src.dataset.prose_quality import POLICY
    output_dir = Path(output_dir).resolve()
    if output_dir in {Path('/'), PROJECT_ROOT.resolve(), output_dir.parent}:
        raise ValueError(f'Unsafe output directory: {output_dir}')
    if workers < 1:
        raise ValueError('workers must be positive')
    limits = dict.fromkeys(SOURCES,max_source_documents)
    limits.update(source_document_limits or {})
    if set(limits) != set(SOURCES) or any(type(v) is not int or v <= 0 for v in limits.values()):
        raise ValueError('Source document limits must be positive integers for known sources')
    if max_source_documents <= 0 or not 0 <= language_threshold <= 1:
        raise ValueError('Invalid candidate budget or language threshold')
    if raw_documents_dir is not None:
        raw_documents_dir = Path(raw_documents_dir).resolve()
        if raw_documents_dir == output_dir or output_dir in raw_documents_dir.parents:
            raise ValueError('Output cannot replace its raw input')
    if output_dir.exists() and not overwrite:
        raise FileExistsError(output_dir)
    temporary = output_dir.with_name(f'.{output_dir.name}.tmp')
    if temporary.exists():
        raise FileExistsError(f'Incomplete build at {temporary}; preserve or move it before retrying')
    model_dir = model_dir or PROJECT_ROOT / 'artifacts/models/glotlid'
    if predictor is None:
        if workers == 1:
            predictor, model_lock = quality_predictor(model_dir)
        else:
            try:
                from language_quality import prepare_lid
            except ModuleNotFoundError:
                from src.dataset.language_quality import prepare_lid
            model_lock = prepare_lid(model_dir)
    else:
        model_lock = {'injected_predictor': True}
    temporary.mkdir(parents=True)
    raw_schema = pa.schema([('document_id',pa.string()),('text',pa.string()),('metadata_json',pa.string())])
    source_info, counts = {}, {}
    # Complete acquisition before local policy evaluation. Raw files can be reused.
    for key, source in SOURCES.items():
        if raw_documents_dir is None:
            from huggingface_hub import HfApi
            source = replace(source, revision=HfApi().dataset_info(source.dataset, revision=source.revision).sha)
            stream = source_rows(source, seed + list(SOURCES).index(key)*1009, buffer_size)
            provenance = dict(dataset=source.dataset,config=source.config,split=source.split,revision=source.revision)
        else:
            stream = verified_raw_rows(raw_documents_dir,key)
            companion = raw_documents_dir/key/'sample.json'
            provenance = dict(input=str(raw_documents_dir/key/'sample.parquet'),
                              input_sha256=sha256_file(raw_documents_dir/key/'sample.parquet'),
                              original_manifest=json.loads(companion.read_text()) if companion.exists() else None)
        directory=temporary/'raw'/key
        directory.mkdir(parents=True)
        count=0
        iterator=iter(stream)
        try:
            with pq.ParquetWriter(directory/'sample.parquet',raw_schema,compression='zstd') as writer:
                pending=[]
                for row in itertools.islice(iterator,limits[key]):
                    text=row.get('text') or ''
                    pending.append(dict(document_id=stable_document_id(source,row),
                        text=text,metadata_json=json.dumps({k:v for k,v in row.items() if k!='text'},default=str)))
                    count+=1
                    if len(pending)==64:
                        writer.write_table(pa.Table.from_pylist(pending,schema=raw_schema))
                        pending.clear()
                    if count % 1000 == 0: print(f'Freezing {key}: {count} candidates',flush=True)
                if pending: writer.write_table(pa.Table.from_pylist(pending,schema=raw_schema))
        finally:
            close=getattr(iterator,'close',None)
            if close: close()
        source_info[key]=dict(**provenance,rows=count,hashes={'sample.parquet':sha256_file(directory/'sample.parquet')})
        (directory/'sample.json').write_text(json.dumps(source_info[key],indent=2)+'\n')
        print(f'Frozen {count} raw {key} documents',flush=True)
    dedup=GlobalDeduplicator(temporary/'dedup.sqlite')
    executor = None
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        executor = ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                                       initializer=initialize_quality_worker,initargs=(str(model_dir),))
    try:
        with (temporary/'decisions.jsonl').open('w') as decisions:
            for key,source in SOURCES.items():
                counters=Counter()
                for row,chunks,signals in cleaned_rows(verified_raw_rows(temporary/'raw',key),source,predictor,language_threshold,executor):
                    document_id=stable_document_id(source,row)
                    counters['seen']+=1
                    if counters['seen'] % 1000 == 0:
                        print(f'Checking {key}: {counters["seen"]} candidates',flush=True)
                    kept=False
                    doc_hash=normalized_text_hash(row['text'])
                    if chunks:
                        kept, reason, doc_hash=dedup.add(key,document_id,row['text'],chunks)
                        if reason: signals['reasons'].append(reason)
                    counters['accepted' if kept else 'rejected']+=1
                    counters.update(signals['reasons'])
                    decisions.write(json.dumps(dict(source=key,document_id=document_id,
                        text_sha256=hashlib.sha256(row['text'].encode()).hexdigest(),
                        normalized_document_hash=doc_hash,accepted=kept,**signals))+'\n')
                counts[key]=dict(counters)
                print(f'Cleaned {key}: {counters["accepted"]}/{counters["seen"]} globally unique documents',flush=True)
        writers={(part,key):MaterializedParquetWriter(temporary/part/key,shard_bytes)
                 for part in byte_targets for key in SOURCES}
        try:
            for doc_hash,key,document_id,chunks_json in dedup.db.execute('SELECT * FROM documents ORDER BY hash'):
                part=assign_partition(doc_hash,split_seed,holdout_permille)
                writer=writers[part,key]
                if writer.total_text_bytes >= byte_targets[part][key]:
                    continue
                for chunk_index,text in enumerate(json.loads(chunks_json)):
                    writer.write(dict(document_id=document_id,chunk_index=chunk_index,source=key,
                                      text_sha256=normalized_text_hash(text),text=text),len(text.encode()))
        finally:
            for writer in writers.values(): writer.close()
    finally:
        if executor is not None: executor.shutdown(wait=True,cancel_futures=True)
        dedup.close()
    partitions={part:{} for part in byte_targets}
    short=[]
    for (part,key),writer in writers.items():
        target=byte_targets[part][key]
        partitions[part][key]=dict(target_text_bytes=target,text_bytes=writer.total_text_bytes,
                                  chunks=writer.total_chunks,files=writer.files)
        if writer.total_text_bytes < target: short.append(f'{part}/{key}')
    manifest=dict(format_version=2,description='Globally exact-deduplicated, quality-filtered document corpus',
        sources=source_info,mixtures=available_mixes,evaluation_mixture=TEST_MIX,
        partitioning=dict(method='sha256(split_seed:normalized_document_hash)',seed=split_seed,
                          validation_buckets_per_1000=holdout_permille,test_buckets_per_1000=holdout_permille),
        cleaning=dict(language=LANGUAGE_POLICY,language_threshold=language_threshold,
                      language_min_chars=80,language_max_chars=1000,language_confidence=.8,prose=POLICY,
                      min_characters=MIN_CHARACTERS,min_alphabetic_ratio=MIN_ALPHABETIC_RATIO,
                      max_repeated_line_ratio=MAX_REPEATED_LINE_RATIO,max_chunk_characters=MAX_CHUNK_CHARACTERS,
                      max_chunks_per_document=MAX_CHUNKS_PER_DOCUMENT,web_metadata_confidence=.98,pdf_metadata_confidence=.90,
                      model=model_lock,all_sources=True),
        deduplication=dict(method='global normalized exact documents and retained chunks; reject later overlapping document',
                           source_priority=list(SOURCES),near_duplicates=False),
        acquisition=dict(seed=seed,max_source_documents=max_source_documents,source_document_limits=limits,shuffle_buffer=buffer_size),
        quality_workers=workers,
        filter_counters=counts,partitions=partitions,underfilled=short,
        decisions_sha256=sha256_file(temporary/'decisions.jsonl'),
        code_sha256={p:sha256_file(Path(__file__).with_name(p)) for p in ['clean_dataset.py','language_quality.py','prose_quality.py']})
    (temporary/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if short and not allow_underfilled:
        raise RuntimeError(f'Candidate budget underfilled {short}; raw data and diagnostics saved in {temporary}. Increase budget or explicitly allow underfilled study pools.')
    safe_replace_directory(output_dir,temporary,overwrite)
    print(f'Prepared documents: {output_dir}',flush=True)


def supplement_documents(documents_dir, raw_documents_dir, output_dir, source_key,
                         workers=1, model_dir=None, allow_underfilled=False,
                         retain_all_candidates=False, skip_raw_documents=0):
    """Append new passing documents to unfilled budgets, preserving existing shards.

    Original files are hard-linked as immutable evidence; the deduplication index
    is copied. An interrupted attempt never mutates the parent corpus. Supplements
    have separate raw inputs, decisions and provenance in the output manifest.
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    try:
        from language_quality import LANGUAGE_POLICY, prepare_lid
        from prose_quality import POLICY
    except ModuleNotFoundError:
        from src.dataset.language_quality import LANGUAGE_POLICY, prepare_lid
        from src.dataset.prose_quality import POLICY

    parent, raw, output = map(lambda p: Path(p).resolve(),
                              (documents_dir, raw_documents_dir, output_dir))
    if workers < 1 or source_key not in SOURCES:
        raise ValueError('Invalid worker count or source')
    if output == parent or parent in output.parents or output in parent.parents:
        raise ValueError('Supplement output must be separate from its parent')
    temporary = output.with_name(f'.{output.name}.supplement.tmp')
    if output.exists() or temporary.exists():
        raise FileExistsError(f'Output or incomplete attempt exists: {output}')
    manifest = json.loads((parent/'manifest.json').read_text())
    cleaning_code_sha256 = sha256_file(Path(__file__))
    cleaning = manifest['cleaning']
    expected = dict(language=LANGUAGE_POLICY, prose=POLICY, min_characters=MIN_CHARACTERS,
                    min_alphabetic_ratio=MIN_ALPHABETIC_RATIO,
                    max_repeated_line_ratio=MAX_REPEATED_LINE_RATIO,
                    max_chunk_characters=MAX_CHUNK_CHARACTERS,
                    max_chunks_per_document=MAX_CHUNKS_PER_DOCUMENT,
                    language_min_chars=80, language_max_chars=1000, language_confidence=.8,
                    web_metadata_confidence=.98, pdf_metadata_confidence=.90, all_sources=True)
    if manifest.get('format_version') != 2 or any(cleaning.get(k) != v for k,v in expected.items()):
        raise ValueError('Parent cleaning policy differs from current implementation')
    for name in ('language_quality.py', 'prose_quality.py'):
        if manifest['code_sha256'][name] != sha256_file(Path(__file__).with_name(name)):
            raise ValueError(f'Parent quality implementation differs: {name}')
    if any((parent/f'dedup.sqlite{suffix}').exists() for suffix in ('-wal', '-journal')):
        raise ValueError('Parent index must be closed before supplementation')
    provenance = json.loads((raw/source_key/'sample.json').read_text())
    if not 0 <= skip_raw_documents <= provenance.get('rows', 0):
        raise ValueError('Invalid raw document offset')
    raw_hash = sha256_file(raw/source_key/'sample.parquet')
    if provenance.get('hashes', {}).get('sample.parquet') != raw_hash:
        raise ValueError('Supplement requires a verified raw sample manifest')
    if not retain_all_candidates and not any(v[source_key]['text_bytes'] < v[source_key]['target_text_bytes']
               for v in manifest['partitions'].values()):
        raise ValueError('Source budgets are already filled')
    model_dir = Path(model_dir or PROJECT_ROOT/'artifacts/models/glotlid')
    model_lock = prepare_lid(model_dir)
    if model_lock['sha256'] != cleaning['model']['sha256']:
        raise ValueError('Language model differs from parent')
    # Check inherited shards before linking them into another corpus.
    for part, sources in manifest['partitions'].items():
        for key, record in sources.items():
            for file in record['files']:
                if sha256_file(parent/part/key/file['file']) != file['sha256']:
                    raise ValueError('Parent shard checksum mismatch')
    if sha256_file(parent/'decisions.jsonl') != manifest['decisions_sha256']:
        raise ValueError('Parent decisions checksum mismatch')
    def clone_file(src, dst):
        if Path(src).name in ('dedup.sqlite', 'manifest.json'):
            return shutil.copy2(src, dst)
        os.link(src, dst)
        return dst
    shutil.copytree(parent, temporary, copy_function=clone_file)
    supplements = manifest.setdefault('supplements', [])
    relative = Path('supplements')/f'{len(supplements):03d}'
    evidence = temporary/relative
    (evidence/source_key).mkdir(parents=True)
    for name in ('sample.parquet', 'sample.json'):
        shutil.copy2(raw/source_key/name, evidence/source_key/name)
    dedup = GlobalDeduplicator(temporary/'dedup.sqlite', resume=True)
    writers = {part: MaterializedParquetWriter(evidence/'new_shards'/part,
                                               DEFAULT_DOCUMENT_SHARD_BYTES)
               for part in manifest['partitions']}
    counters = Counter()
    executor = None
    predictor = None
    try:
        if workers > 1:
            executor = ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize_quality_worker, initargs=(str(model_dir),))
        else:
            predictor, _ = quality_predictor(model_dir)
        def rows():
            import itertools
            for row in itertools.islice(verified_raw_rows(evidence, source_key), skip_raw_documents, None):
                if provenance.get('dataset') == 'HuggingFaceFW/finewiki':
                    if source_key != 'wiki' or row.get('page_id') is None:
                        raise ValueError('FineWiki requires canonical Wikipedia page IDs')
                    row['id'] = str(row['page_id'])
                yield row
        partitioning = manifest['partitioning']
        if partitioning['validation_buckets_per_1000'] != partitioning['test_buckets_per_1000']:
            raise ValueError('Unsupported asymmetric parent partitioning')
        with (evidence/'decisions.jsonl').open('w') as decisions:
            for row, chunks, signals in cleaned_rows(rows(), SOURCES[source_key], predictor,
                                                    cleaning['language_threshold'], executor):
                doc_id = stable_document_id(SOURCES[source_key], row)
                doc_hash = normalized_text_hash(row['text'])
                kept = False
                if chunks:
                    kept, reason, doc_hash = dedup.add(source_key, doc_id, row['text'], chunks)
                    if reason: signals['reasons'].append(reason)
                counters['seen'] += 1
                counters['accepted' if kept else 'rejected'] += 1
                counters.update(signals['reasons'])
                decisions.write(json.dumps(dict(source=source_key, document_id=doc_id,
                    text_sha256=hashlib.sha256(row['text'].encode()).hexdigest(),
                    normalized_document_hash=doc_hash, accepted=kept, **signals))+'\n')
                if kept:
                    part = assign_partition(doc_hash, partitioning['seed'],
                                            partitioning['test_buckets_per_1000'])
                    record = manifest['partitions'][part][source_key]
                    writer = writers[part]
                    if record['text_bytes'] + writer.total_text_bytes < record['target_text_bytes']:
                        # Write exactly the unique chunks recorded in the global index.
                        unique_chunks = json.loads(dedup.db.execute(
                            'SELECT chunks FROM documents WHERE hash=?', (doc_hash,)).fetchone()[0])
                        for index, text in enumerate(unique_chunks):
                            writer.write(dict(document_id=doc_id, chunk_index=index, source=source_key,
                                              text_sha256=normalized_text_hash(text), text=text), len(text.encode()))
                if counters['seen'] % 1000 == 0:
                    print(f"Supplement {source_key}: {counters['seen']} checked, "
                          f"{counters['accepted']} unique, {writers['train'].total_text_bytes:,} train bytes added", flush=True)
                if not retain_all_candidates and all(manifest['partitions'][part][source_key]['text_bytes'] + writer.total_text_bytes
                       >= manifest['partitions'][part][source_key]['target_text_bytes']
                       for part, writer in writers.items()):
                    break
    finally:
        if executor is not None: executor.shutdown(wait=True, cancel_futures=True)
        for writer in writers.values(): writer.close()
        dedup.close()
    for part, writer in writers.items():
        record = manifest['partitions'][part][source_key]
        next_index = max((int(Path(f['file']).stem.split('-')[-1]) for f in record['files']), default=-1)+1
        for index, file in enumerate(writer.files, next_index):
            name = f'part-{index:05d}.parquet'
            destination = temporary/part/source_key/name
            if destination.exists(): raise FileExistsError(destination)
            (writer.directory/file['file']).rename(destination)
            record['files'].append(dict(file, file=name))
        record['text_bytes'] += writer.total_text_bytes
        record['chunks'] += writer.total_chunks
    shutil.rmtree(evidence/'new_shards')
    supplements.append(dict(directory=str(relative), source=source_key, provenance=provenance,
        raw_start=skip_raw_documents, retain_all_candidates=retain_all_candidates,
        counters=dict(counters), decisions_sha256=sha256_file(evidence/'decisions.jsonl'),
        parent_manifest_sha256=sha256_file(parent/'manifest.json'),
        cleaning_code_sha256=cleaning_code_sha256,
        ordering='Existing accepted documents first, then supplement input order; existing shards unchanged'))
    manifest['underfilled'] = [f'{part}/{key}' for part, sources in manifest['partitions'].items()
        for key, record in sources.items() if record['text_bytes'] < record['target_text_bytes']]
    (temporary/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if manifest['underfilled'] and not allow_underfilled:
        raise RuntimeError(f"Still underfilled {manifest['underfilled']}; saved at {temporary}")
    temporary.rename(output)
    print(f'Supplemented documents: {output}; underfilled={manifest["underfilled"]}', flush=True)


def create_temporary_splits(documents_dir, output_dir, seed=42, max_documents=100,
                            holdout_permille=100, shard_bytes=DEFAULT_DOCUMENT_SHARD_BYTES):
    """Repartition a bounded subset of production TRAIN only, without mutation."""
    documents_dir,output_dir=Path(documents_dir).resolve(),Path(output_dir).resolve()
    if output_dir == documents_dir or output_dir in documents_dir.parents or documents_dir in output_dir.parents:
        raise ValueError('Temporary splits must be outside their parent corpus')
    if output_dir.exists(): raise FileExistsError(output_dir)
    if max_documents < 1: raise ValueError('max_documents must be positive')
    assign_partition('check',seed,holdout_permille)
    parent=json.loads((documents_dir/'manifest.json').read_text())
    if parent.get('format_version') != 2 or not parent.get('deduplication'):
        raise ValueError('Temporary splits require the globally deduplicated document format')
    import sqlite3
    import tempfile
    # Disk-backed grouping keeps complete documents together without loading the corpus.
    with tempfile.TemporaryDirectory() as temp:
        db=sqlite3.connect(str(Path(temp)/'selection.sqlite'))
        try:
            db.execute('CREATE TABLE rows (id TEXT, source TEXT, rank TEXT, value TEXT)')
            for key in SOURCES:
                for file in parent['partitions']['train'][key]['files']:
                    path=documents_dir/'train'/key/file['file']
                    if sha256_file(path)!=file['sha256']: raise ValueError(f'Shard checksum mismatch: {path}')
                    for batch in pq.ParquetFile(path).iter_batches():
                        for row in batch.to_pylist():
                            rank=hashlib.sha256(f"{seed}:{row['document_id']}".encode()).hexdigest()
                            db.execute('INSERT INTO rows VALUES (?,?,?,?)',(row['document_id'],key,rank,json.dumps(row)))
            db.execute('CREATE INDEX row_id ON rows(id,source)')
            selected=list(db.execute('SELECT id,source,rank FROM rows GROUP BY id,source ORDER BY rank LIMIT ?', (max_documents,)))
            if not selected:
                raise ValueError('Parent training partition contains no documents')
            output_dir.mkdir(parents=True)
            writers={(p,k):MaterializedParquetWriter(output_dir/p/k,shard_bytes)
                     for p in ['train','validation','test'] for k in SOURCES}
            try:
                for doc_id,key,rank in selected:
                    part=assign_partition(rank,seed,holdout_permille)
                    for value, in db.execute('SELECT value FROM rows WHERE id=? AND source=?',(doc_id,key)):
                        row=json.loads(value)
                        writers[part,key].write(row,len(row['text'].encode()))
            finally:
                for writer in writers.values(): writer.close()
            manifest=dict(parent)
            manifest.update(temporary=True,parent_manifest_sha256=sha256_file(documents_dir/'manifest.json'),
                parent_corpus=str(documents_dir),selection=dict(parent_partition='train',seed=seed,max_documents=max_documents,selected_documents=len(selected)),
                partitioning=dict(method='seeded document grouping of parent train only',seed=seed,
                    validation_buckets_per_1000=holdout_permille,test_buckets_per_1000=holdout_permille),
                partitions={p:{k:dict(chunks=writers[p,k].total_chunks,text_bytes=writers[p,k].total_text_bytes,files=writers[p,k].files)
                               for k in SOURCES} for p in ['train','validation','test']})
            manifest.pop('underfilled', None)
            manifest['empty_source_partitions'] = [f'{p}/{k}' for (p,k),w in writers.items() if not w.total_chunks]
            (output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        finally: db.close()
    return manifest


def iter_materialized_chunks(
    documents_dir: Path,
    partition: str,
    source_key: str,
    counters: Counter[str],
) -> Iterator[tuple[str, str]]:
    source_dir = documents_dir / partition / source_key
    manifest = json.loads((documents_dir / 'manifest.json').read_text())
    records = manifest['partitions'][partition][source_key]['files']
    files = [source_dir / record['file'] for record in records]
    if not files:
        raise FileNotFoundError(f"no materialized Parquet shards in {source_dir}")
    for path, record in zip(files, records, strict=True):
        if sha256_file(path) != record['sha256']:
            raise ValueError(f'Materialized shard checksum mismatch: {path}')
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
    if documents_dir is None:
        raise ValueError('Prepare globally deduplicated documents before tokenization')
    document_manifest_path = documents_dir / 'manifest.json'
    document_manifest = json.loads(document_manifest_path.read_text())
    if document_manifest.get('format_version') != 2:
        raise ValueError('Rebuild legacy documents before tokenization')
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
        for key in weights:
            iterator = iter_materialized_chunks(documents_dir, partition, key, counters[key])
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
                "sources": {key: document_manifest['sources'][key] for key in weights},
            },
            "tokenizer": tokenizer_metadata(tokenizer, tokenizer_path, tokenizer_digest),
            "document_encoding": ["BOS", "content", "EOS"],
            "filtering": {
                **document_manifest['cleaning'],
                "deduplication": document_manifest['deduplication'],
                "token_stage_exact_overlap_check": True,
            },
            "partitioning": {
                **document_manifest['partitioning'],
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
            "document_acquisition": document_manifest.get('acquisition'),
            "artifact_source_order": list(weights),
            "training_loader_shuffles_token_blocks": partition == "train",
            "materialized_documents": (
                str(documents_dir.resolve()) if documents_dir is not None else None
            ),
            "document_manifest_sha256": sha256_file(document_manifest_path),
            "document_sources": document_manifest['sources'],
            "document_cleaning": document_manifest['cleaning'],
            "document_deduplication": document_manifest['deduplication'],
            "temporary_splits": document_manifest.get('temporary', False),
        }
    )
    print(f"Built {builder.final_dir}: {builder.writer.total_tokens:,} tokens")
    return exact_hashes if partition != "train" else test_hashes


def validate_args(args: argparse.Namespace) -> None:
    if args.max_source_documents <= 0 or not 0 <= args.language_threshold <= 1:
        raise ValueError('Invalid candidate budget or language threshold')
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
        data = json.loads(manifest.read_text())
        if data.get('format_version') != 2 or not data.get('deduplication'):
            raise ValueError('Rebuild legacy document pools with global deduplication before tokenization')


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == 'extra':
        try:
            from extra_dataset import main as extra_main
        except ModuleNotFoundError:
            from src.dataset.extra_dataset import main as extra_main
        extra_main(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] in {'pool-export', 'pool-select'}:
        try:
            from document_pool import export_document_pool, select_document_pool
        except ModuleNotFoundError:
            from src.dataset.document_pool import export_document_pool, select_document_pool
        parser = argparse.ArgumentParser(description='Export all accepted training documents or select a mixture')
        parser.add_argument('--documents-dir', type=Path, required=True)
        parser.add_argument('--output-dir', type=Path, required=True)
        if sys.argv[1] == 'pool-select':
            parser.add_argument('--train-tokens', type=int, required=True)
            parser.add_argument('--weights', type=json.loads, default=MIXES['balanced'])
            parser.add_argument('--bytes-per-token', type=float, default=5.0)
        args = parser.parse_args(sys.argv[2:])
        if sys.argv[1] == 'pool-export':
            export_document_pool(args.documents_dir, args.output_dir)
        else:
            select_document_pool(args.documents_dir, args.output_dir, args.train_tokens,
                                 args.weights, args.bytes_per_token)
        return
    if len(sys.argv) > 1 and sys.argv[1] == 'supplement-documents':
        parser = argparse.ArgumentParser(description='Fill document budgets while preserving existing shards')
        parser.add_argument('--documents-dir', type=Path, required=True)
        parser.add_argument('--raw-documents-dir', type=Path, required=True)
        parser.add_argument('--output-dir', type=Path, required=True)
        parser.add_argument('--source', choices=list(SOURCES), required=True)
        parser.add_argument('--workers', type=int, default=1)
        parser.add_argument('--model-dir', type=Path)
        parser.add_argument('--allow-underfilled-documents', action='store_true')
        parser.add_argument('--retain-all-candidates', action='store_true',
                            help='Keep checking and indexing candidates after selection quotas are filled')
        parser.add_argument('--skip-raw-documents', type=int, default=0,
                            help='Recorded input offset when continuing the same frozen sample')
        args = parser.parse_args(sys.argv[2:])
        supplement_documents(args.documents_dir, args.raw_documents_dir, args.output_dir,
                             args.source, args.workers, args.model_dir, args.allow_underfilled_documents,
                             args.retain_all_candidates, args.skip_raw_documents)
        return
    if len(sys.argv) > 1 and sys.argv[1] == 'split-experiment':
        parser = argparse.ArgumentParser(description='Create isolated temporary splits from parent TRAIN only')
        parser.add_argument('--documents-dir', type=Path, required=True)
        parser.add_argument('--output-dir', type=Path, required=True)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--max-documents', type=int, default=100)
        parser.add_argument('--holdout-permille', type=int, default=100)
        args = parser.parse_args(sys.argv[2:])
        create_temporary_splits(args.documents_dir,args.output_dir,args.seed,args.max_documents,args.holdout_permille)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "audit-prose":
        try:
            from prose_quality import main as prose_main
        except ModuleNotFoundError:
            from src.dataset.prose_quality import main as prose_main
        prose_main(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] in {"audit-pages", "audit-language", "sample-pdfs", "prepare-lid"}:
        try:
            from language_quality import audit_main
        except ModuleNotFoundError:
            from src.dataset.language_quality import audit_main
        audit_main(sys.argv[1:])
        return
    args = parse_args()
    validate_args(args)
    if args.mix == 'selected':
        if args.documents_dir is None or args.documents_only:
            raise ValueError('--mix selected requires an existing pool-select output')
        manifest = json.loads((args.documents_dir/'manifest.json').read_text())
        if not manifest.get('selection') or 'selected' not in manifest.get('mixtures', {}):
            raise ValueError('The input is not a pool-select output')
        selected = {'selected': manifest['mixtures']['selected']}
    else:
        selected = MIXES if args.mix == "all" else {args.mix: MIXES[args.mix]}
    if args.documents_only or args.documents_dir is None:
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
            raw_documents_dir=args.raw_documents_dir,
            workers=args.workers,
            source_document_limits=args.source_document_limits,
            max_source_documents=args.max_source_documents,
            split_seed=args.split_seed,
            allow_underfilled=args.allow_underfilled_documents,
            model_dir=args.model_dir,
            language_threshold=args.language_threshold,
        )
        if args.documents_only:
            return
        args.documents_dir = args.documents_output_dir.resolve()

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
