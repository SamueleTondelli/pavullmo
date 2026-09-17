"""Train SentencePiece from either legacy streaming data or frozen documents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import os
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / "tmp" / "cache" / "huggingface"))

from datasets import load_dataset
import pyarrow.parquet as pq
import sentencepiece as spm
from tqdm import tqdm


DATASET = "gsarti/clean_mc4_it"
VARIANT = "tiny"
N_THREADS = 8
DEFAULT_MIXTURE = "balanced"
DEFAULT_MAX_SENTENCE_LENGTH = 65_536


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument(
        "--byte-budget",
        type=int,
        required=True,
        help="raw UTF-8 text bytes used to train the tokenizer",
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
        required=True,
        help="directory where tokenizer.model and tokenizer.vocab are saved",
    )
    parser.add_argument(
        "--documents-dir",
        type=Path,
        help="tokenizer-independent corpus produced by build_dataset.py --documents-only",
    )
    parser.add_argument(
        "--mixture",
        default=DEFAULT_MIXTURE,
        help=f"source weights from the corpus manifest (default: {DEFAULT_MIXTURE})",
    )
    parser.add_argument(
        "--max-sentence-length",
        type=int,
        default=DEFAULT_MAX_SENTENCE_LENGTH,
        help=(
            "maximum UTF-8 sentence length accepted by SentencePiece; the larger "
            "default accommodates the pipeline's document chunks"
        ),
    )
    return parser.parse_args()


def byte_allocations(total: int, weights: dict[str, float]) -> dict[str, int]:
    keys = list(weights)
    result = {key: int(total * weights[key]) for key in keys}
    result[keys[-1]] += total - sum(result.values())
    return result


def legacy_text_iterator(byte_budget: int, stats: dict[str, int]) -> Iterator[str]:
    dataset = load_dataset(DATASET, VARIANT, split="train", streaming=True)
    for document in dataset:
        if stats["bytes"] >= byte_budget:
            break
        text = document["text"].strip()
        if not text:
            continue
        text_bytes = len(text.encode("utf-8"))
        stats["bytes"] += text_bytes
        stats["documents"] += 1
        yield text


def materialized_source_texts(source_dir: Path) -> Iterator[str]:
    files = sorted(source_dir.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet document shards in {source_dir}")
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["text"]):
            yield from batch.column("text").to_pylist()


def materialized_text_iterator(
    documents_dir: Path,
    byte_budget: int,
    mixture: str,
    stats: dict[str, object],
) -> Iterator[str]:
    manifest_path = documents_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"materialized corpus manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mixtures = manifest.get("mixtures", {})
    if mixture not in mixtures:
        raise ValueError(
            f"mixture {mixture!r} not found in {manifest_path}; "
            f"available: {sorted(mixtures)}"
        )
    weights = mixtures[mixture]
    targets = byte_allocations(byte_budget, weights)
    source_stats: dict[str, dict[str, int]] = {}
    stats["mixture"] = mixture
    stats["weights"] = weights
    stats["sources"] = source_stats

    for source, target in targets.items():
        used = 0
        documents = 0
        for text in materialized_source_texts(documents_dir / "train" / source):
            if used >= target:
                break
            text_bytes = len(text.encode("utf-8"))
            used += text_bytes
            documents += 1
            yield text
        if used < target:
            raise RuntimeError(
                f"materialized source {source!r} supplied {used:,} bytes; "
                f"tokenizer mixture requires at least {target:,}"
            )
        source_stats[source] = {
            "target_bytes": target,
            "actual_bytes": used,
            "chunks": documents,
        }


def progress_iterator(
    texts: Iterator[str], byte_budget: int, stats: dict[str, object]
) -> Iterator[str]:
    with tqdm(
        total=byte_budget,
        desc="Reading tokenizer corpus",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress:
        previous = 0
        for text in texts:
            if "sources" in stats:
                # Per-source statistics are finalized at source boundaries, so
                # count this row directly while the current source is active.
                progress.update(len(text.encode("utf-8")))
            else:
                current = int(stats["bytes"])
                progress.update(current - previous)
                previous = current
            yield text


def main() -> None:
    args = parse_args()
    if (
        args.vocab_size <= 0
        or args.byte_budget <= 0
        or args.max_sentence_length <= 0
    ):
        raise ValueError(
            "--vocab-size, --byte-budget, and --max-sentence-length must be positive"
        )

    output_dir = Path(args.model_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_prefix = output_dir / "tokenizer"
    stats: dict[str, object]
    if args.documents_dir is None:
        stats = {"bytes": 0, "documents": 0}
        texts = legacy_text_iterator(args.byte_budget, stats)  # type: ignore[arg-type]
        source_description = f"{DATASET}/{VARIANT} streaming train"
    else:
        documents_dir = args.documents_dir.resolve()
        stats = {}
        texts = materialized_text_iterator(
            documents_dir, args.byte_budget, args.mixture, stats
        )
        source_description = str(documents_dir)

    print(
        f"Training vocab size {args.vocab_size:,} on {args.byte_budget:,} bytes "
        f"from {source_description}; saving to {tokenizer_prefix}.*"
    )
    spm.SentencePieceTrainer.train(
        sentence_iterator=progress_iterator(texts, args.byte_budget, stats),
        model_prefix=str(tokenizer_prefix),
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        byte_fallback=True,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        user_defined_symbols=["<system>", "<user>", "<assistant>"],
        num_threads=N_THREADS,
        max_sentence_length=args.max_sentence_length,
    )
    metadata = {
        "vocab_size": args.vocab_size,
        "requested_byte_budget": args.byte_budget,
        "max_sentence_length": args.max_sentence_length,
        "source": source_description,
        "statistics": stats,
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
