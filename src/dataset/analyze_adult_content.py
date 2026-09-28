"""Temporary, read-only audit of adult-content signals in frozen document pools.

This is an investigation aid, not a production content filter.  It deliberately
keeps vulgar/pornographic markers separate from clinical or educational sexual
language and writes masked snippets for manual review.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
from typing import Iterable

import pyarrow.parquet as pq
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCUMENTS_DIR = PROJECT_ROOT / "artifacts/documents/balanced_1b_10m_10m"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/analysis/adult_content_balanced_1b_10m_10m"

# Keep the groups interpretable.  Terms here are signals for review, not a
# claim that the containing document is harmful or should be removed.
TERM_GROUPS: dict[str, tuple[str, ...]] = {
    "pornographic": (
        r"porn(?:o|ografia|ografico|ografica|ografici|ografiche)?",
        r"xxx",
        r"gangbang",
        r"hentai",
        r"camgirl",
        r"webcam\s+erotic[ao]",
        r"video\s+hard",
    ),
    "vulgar_explicit": (
        r"cazz(?:o|i|one|oni)",
        r"fig(?:a|he)",
        r"minchi(?:a|e|one|oni)",
        r"pompini?",
        r"sborr(?:a|are|ato|ata|i|o)",
        r"scop(?:are|ata|ato|ando|ami|ami|ano)",
        r"incul(?:are|ata|ato|ando)",
    ),
    "sexual_act": (
        r"masturb(?:azione|arsi|are|ato|ata)",
        r"orgasm(?:o|i|ico|ica)",
        r"fellatio",
        r"cunnilingus",
        r"coito",
        r"rapport(?:o|i)\s+sessual(?:e|i)",
        r"penetrazion(?:e|i)",
        r"sesso\s+orale",
    ),
    "sexual_general": (
        r"sess(?:o|uale|uali|ualità)",
        r"erotic(?:o|a|i|he)",
        r"nudi(?:tà|smo)?",
        r"prostitut(?:a|e|o|i|zione)",
        r"escort",
    ),
    "anatomy": (
        r"pen(?:e|i)",
        r"vagin(?:a|ale|ali|e)",
        r"clitoride",
        r"genital(?:e|i)",
    ),
}

GROUP_WEIGHTS = {
    "pornographic": 5,
    "vulgar_explicit": 4,
    "sexual_act": 2,
    "sexual_general": 1,
    "anatomy": 1,
}
STRONG_MARKER_GROUPS = {"pornographic", "vulgar_explicit"}


def _compile_terms() -> dict[str, re.Pattern[str]]:
    return {
        group: re.compile(
            rf"(?<!\w)(?:{'|'.join(terms)})(?!\w)", re.IGNORECASE
        )
        for group, terms in TERM_GROUPS.items()
    }


COMPILED_TERMS = _compile_terms()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a format-2 document pool for Italian adult-content signals."
    )
    parser.add_argument("--documents-dir", type=Path, default=DEFAULT_DOCUMENTS_DIR)
    parser.add_argument("--partition", choices=("train", "validation", "test", "all"), default="train")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--max-chunks",
        type=int,
        help="prefix limit for smoke tests only; not a source-balanced sample",
    )
    parser.add_argument("--review-samples", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parquet_paths(documents_dir: Path, partition: str) -> list[Path]:
    partitions = ("train", "validation", "test") if partition == "all" else (partition,)
    paths = [
        path
        for name in partitions
        for path in sorted((documents_dir / name).glob("*/*.parquet"))
    ]
    if not paths:
        raise FileNotFoundError(f"no Parquet shards found for {partition!r} under {documents_dir}")
    return paths


def analyze_text(text: str) -> tuple[dict[str, int], Counter[str], int]:
    groups: dict[str, int] = {}
    terms: Counter[str] = Counter()
    score = 0
    for group, pattern in COMPILED_TERMS.items():
        matches = pattern.findall(text)
        count = len(matches)
        for match in matches:
            terms[f"{group}:{match.casefold()}"] += 1
        if count:
            groups[group] = count
            score += GROUP_WEIGHTS[group] * count
    return groups, terms, score


def masked_excerpt(text: str, limit: int = 360) -> str:
    matches = [
        (match.start(), match.end(), group)
        for group, pattern in COMPILED_TERMS.items()
        for match in pattern.finditer(text)
    ]
    if not matches:
        return ""
    start = max(0, min(item[0] for item in matches) - limit // 2)
    excerpt = " ".join(text[start : start + limit].split())
    for group, pattern in COMPILED_TERMS.items():
        excerpt = pattern.sub(f"[{group.upper()}]", excerpt)
    return excerpt


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit(args: argparse.Namespace) -> dict[str, object]:
    if args.batch_size <= 0 or args.review_samples < 0:
        raise ValueError("--batch-size must be positive and --review-samples cannot be negative")
    if args.max_chunks is not None and args.max_chunks <= 0:
        raise ValueError("--max-chunks must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite to replace reports")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    totals = Counter()
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    term_counts: Counter[str] = Counter()
    flagged_documents: dict[str, dict[str, object]] = {}
    stop = False

    paths = parquet_paths(args.documents_dir, args.partition)
    progress = tqdm(desc="Auditing chunks", unit="chunks", total=args.max_chunks)
    for path in paths:
        partition = path.relative_to(args.documents_dir).parts[0]
        parquet = pq.ParquetFile(path)
        required = {"document_id", "chunk_index", "source", "text"}
        missing = required - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for batch in parquet.iter_batches(batch_size=args.batch_size, columns=sorted(required)):
            for row in batch.to_pylist():
                if args.max_chunks is not None and totals["chunks"] >= args.max_chunks:
                    stop = True
                    break
                text = row["text"] or ""
                source = row["source"] or path.parent.name
                groups, terms, score = analyze_text(text)
                strong_marker_hits = sum(groups.get(group, 0) for group in STRONG_MARKER_GROUPS)
                strong_marker = strong_marker_hits > 0
                dense_strong_markers = strong_marker_hits >= 3
                text_bytes = len(text.encode("utf-8"))
                totals.update(chunks=1, text_bytes=text_bytes)
                by_source[source].update(chunks=1, text_bytes=text_bytes)
                progress.update()
                if not groups:
                    continue
                totals.update(flagged_chunks=1, flagged_text_bytes=text_bytes)
                by_source[source].update(flagged_chunks=1, flagged_text_bytes=text_bytes)
                if strong_marker:
                    totals.update(strong_marker_chunks=1, strong_marker_text_bytes=text_bytes)
                    by_source[source].update(strong_marker_chunks=1, strong_marker_text_bytes=text_bytes)
                if dense_strong_markers:
                    totals.update(dense_strong_marker_chunks=1)
                    by_source[source].update(dense_strong_marker_chunks=1)
                term_counts.update(terms)
                key = f"{partition}:{source}:{row['document_id']}"
                document = flagged_documents.setdefault(
                    key,
                    {
                        "partition": partition,
                        "source": source,
                        "document_id": row["document_id"],
                        "matched_chunks": 0,
                        "score": 0,
                        "strong_marker": False,
                        "strong_marker_hits": 0,
                        "groups": Counter(),
                        "terms": Counter(),
                        "excerpt": "",
                    },
                )
                document["matched_chunks"] += 1
                document["score"] += score
                document["strong_marker"] = document["strong_marker"] or strong_marker
                document["strong_marker_hits"] += strong_marker_hits
                document["groups"].update(groups)
                document["terms"].update(terms)
                if not document["excerpt"] or score > document.get("excerpt_score", -1):
                    document["excerpt"] = masked_excerpt(text)
                    document["excerpt_score"] = score
            if stop:
                break
        if stop:
            break
    progress.close()

    docs = sorted(
        flagged_documents.values(),
        key=lambda item: (-int(item["score"]), str(item["document_id"])),
    )
    strong_marker_docs = sum(bool(item["strong_marker"]) for item in docs)
    dense_strong_marker_docs = sum(int(item["strong_marker_hits"]) >= 3 for item in docs)
    summary: dict[str, object] = {
        "documents_dir": str(args.documents_dir.resolve()),
        "partition": args.partition,
        "sample_limited": args.max_chunks is not None,
        "chunks_scanned": totals["chunks"],
        "text_bytes_scanned": totals["text_bytes"],
        "flagged_chunks": totals["flagged_chunks"],
        "flagged_chunk_rate": totals["flagged_chunks"] / totals["chunks"] if totals["chunks"] else 0.0,
        "strong_marker_chunks": totals["strong_marker_chunks"],
        "strong_marker_chunk_rate": totals["strong_marker_chunks"] / totals["chunks"] if totals["chunks"] else 0.0,
        "dense_strong_marker_chunks": totals["dense_strong_marker_chunks"],
        "dense_strong_marker_chunk_rate": totals["dense_strong_marker_chunks"] / totals["chunks"] if totals["chunks"] else 0.0,
        "flagged_documents": len(docs),
        "strong_marker_documents": strong_marker_docs,
        "dense_strong_marker_documents": dense_strong_marker_docs,
        "source_breakdown": {},
        "limitations": [
            "Lexical signals require manual review and are not removal labels.",
            "A strong marker is explicit vocabulary, not proof of pornographic context.",
            "Counts miss euphemisms and context that does not use listed Italian terms.",
            "Document rates use only flagged-document storage; the denominator reported here is chunks.",
        ],
    }
    for source, counts in sorted(by_source.items()):
        summary["source_breakdown"][source] = {
            "chunks_scanned": counts["chunks"],
            "flagged_chunks": counts["flagged_chunks"],
            "flagged_chunk_rate": counts["flagged_chunks"] / counts["chunks"] if counts["chunks"] else 0.0,
            "strong_marker_chunks": counts["strong_marker_chunks"],
            "strong_marker_chunk_rate": counts["strong_marker_chunks"] / counts["chunks"] if counts["chunks"] else 0.0,
            "dense_strong_marker_chunks": counts["dense_strong_marker_chunks"],
            "dense_strong_marker_chunk_rate": counts["dense_strong_marker_chunks"] / counts["chunks"] if counts["chunks"] else 0.0,
        }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(
        args.output_dir / "term_counts.csv",
        ["group", "matched_form", "occurrences"],
        (
            {"group": key.split(":", 1)[0], "matched_form": key.split(":", 1)[1], "occurrences": count}
            for key, count in term_counts.most_common()
        ),
    )
    review_rows = []
    for item in docs[: args.review_samples]:
        review_rows.append(
            {
                "partition": item["partition"],
                "source": item["source"],
                "document_id": item["document_id"],
                "matched_chunks": item["matched_chunks"],
                "score": item["score"],
                "strong_marker": item["strong_marker"],
                "strong_marker_hits": item["strong_marker_hits"],
                "groups": json.dumps(item["groups"], ensure_ascii=False, sort_keys=True),
                "terms": json.dumps(item["terms"], ensure_ascii=False, sort_keys=True),
                "masked_excerpt": item["excerpt"],
            }
        )
    write_csv(
        args.output_dir / "review_samples.csv",
        ["partition", "source", "document_id", "matched_chunks", "score", "strong_marker", "strong_marker_hits", "groups", "terms", "masked_excerpt"],
        review_rows,
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = audit(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"\nReports written to {args.output_dir}")


if __name__ == "__main__":
    main()
