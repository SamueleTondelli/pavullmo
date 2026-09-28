"""Temporary domain-level audit for selected and isolated-extra web corpora.

The tool is read-only.  It restores URL domains from raw provenance, measures
adult-content lexical signals using ``analyze_adult_content``, and emits domain
tables suitable for choosing a diverse replacement set.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow.parquet as pq
from tqdm import tqdm

from analyze_adult_content import STRONG_MARKER_GROUPS, analyze_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCUMENTS = PROJECT_ROOT / "artifacts/documents/balanced_1b_10m_10m"
DEFAULT_EXTRA = PROJECT_ROOT / "artifacts/corpora/extra"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/analysis/domain_replacement"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents-dir", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--extra-dir", type=Path, default=DEFAULT_EXTRA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def domain_of(url: object) -> str:
    if not isinstance(url, str):
        return "(missing)"
    try:
        domain = (urlsplit(url).hostname or "").casefold().rstrip(".")
    except ValueError:
        return "(invalid)"
    if domain.startswith("www."):
        domain = domain[4:]
    return domain or "(missing)"


def selected_web_paths(root: Path) -> list[Path]:
    return [
        path
        for partition in ("train", "validation", "test")
        for path in sorted((root / partition / "web").glob("*.parquet"))
    ]


def extra_web_paths(root: Path) -> list[Path]:
    return sorted((root / "clean" / "web").glob("*/*.parquet"))


def document_ids(paths: list[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["document_id"], batch_size=16_384):
            result.update(value for value in batch.column(0).to_pylist() if value)
    return result


def selected_domains(root: Path, wanted: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    raw = root / "raw" / "web" / "sample.parquet"
    for batch in tqdm(
        pq.ParquetFile(raw).iter_batches(
            columns=["document_id", "metadata_json"], batch_size=16_384
        ),
        desc="Restoring selected domains",
    ):
        for row in batch.to_pylist():
            doc_id = row["document_id"]
            if doc_id in wanted:
                metadata = json.loads(row["metadata_json"])
                result[doc_id] = domain_of(metadata.get("url"))
    return result


def extra_domains(root: Path, wanted: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    raw_paths = sorted((root / "raw").glob("*.parquet"))
    for raw in raw_paths:
        parquet = pq.ParquetFile(raw)
        if not {"id", "url"}.issubset(parquet.schema_arrow.names):
            continue
        for batch in tqdm(
            parquet.iter_batches(columns=["id", "url"], batch_size=16_384),
            desc=f"Restoring extra domains ({raw.name[:8]})",
        ):
            for row in batch.to_pylist():
                doc_id = f"web:id:{row['id']}"
                if doc_id in wanted:
                    result[doc_id] = domain_of(row["url"])
    return result


def audit(paths: list[Path], domains: dict[str, str]) -> dict[str, Counter[str]]:
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    last_document: str | None = None
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in tqdm(
            parquet.iter_batches(columns=["document_id", "text"], batch_size=2048),
            desc=f"Auditing {path.parent.name}/{path.name}",
            leave=False,
        ):
            for row in batch.to_pylist():
                doc_id = row["document_id"]
                domain = domains.get(doc_id, "(unresolved)")
                text = row["text"] or ""
                groups, _, _ = analyze_text(text)
                strong_hits = sum(groups.get(name, 0) for name in STRONG_MARKER_GROUPS)
                counter = stats[domain]
                if doc_id != last_document:
                    counter["documents"] += 1
                    last_document = doc_id
                counter["chunks"] += 1
                counter["utf8_bytes"] += len(text.encode("utf-8"))
                counter["signal_chunks"] += bool(groups)
                counter["strong_marker_chunks"] += strong_hits > 0
                counter["dense_strong_marker_chunks"] += strong_hits >= 3
                counter["strong_marker_hits"] += strong_hits
    return stats


def rows(stats: dict[str, Counter[str]]) -> list[dict[str, object]]:
    result = []
    for domain, count in stats.items():
        chunks = count["chunks"]
        result.append(
            {
                "domain": domain,
                "documents": count["documents"],
                "chunks": chunks,
                "utf8_bytes": count["utf8_bytes"],
                "signal_chunks": count["signal_chunks"],
                "signal_chunk_rate": count["signal_chunks"] / chunks,
                "strong_marker_chunks": count["strong_marker_chunks"],
                "strong_marker_chunk_rate": count["strong_marker_chunks"] / chunks,
                "dense_strong_marker_chunks": count["dense_strong_marker_chunks"],
                "dense_strong_marker_chunk_rate": count["dense_strong_marker_chunks"] / chunks,
                "strong_marker_hits": count["strong_marker_hits"],
            }
        )
    return result


FIELDS = [
    "domain", "documents", "chunks", "utf8_bytes", "signal_chunks",
    "signal_chunk_rate", "strong_marker_chunks", "strong_marker_chunk_rate",
    "dense_strong_marker_chunks", "dense_strong_marker_chunk_rate",
    "strong_marker_hits",
]


def write_table(path: Path, values: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(values)


def main() -> None:
    args = arguments()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_paths = selected_web_paths(args.documents_dir)
    extra_paths = extra_web_paths(args.extra_dir)
    if not selected_paths or not extra_paths:
        raise FileNotFoundError("selected or extra clean web Parquet files are missing")
    selected_ids = document_ids(selected_paths)
    extra_ids = document_ids(extra_paths)
    selected_domain_map = selected_domains(args.documents_dir, selected_ids)
    extra_domain_map = extra_domains(args.extra_dir, extra_ids)
    selected = rows(audit(selected_paths, selected_domain_map))
    extra = rows(audit(extra_paths, extra_domain_map))

    selected.sort(key=lambda item: (-int(item["dense_strong_marker_chunks"]), -int(item["utf8_bytes"])))
    extra.sort(key=lambda item: (-int(item["utf8_bytes"]), str(item["domain"])))
    clean_candidates = [
        item for item in extra
        if item["domain"] not in {"(missing)", "(invalid)", "(unresolved)"}
        and int(item["documents"]) >= 5
        and float(item["strong_marker_chunk_rate"]) <= 0.002
        and float(item["dense_strong_marker_chunk_rate"]) <= 0.0005
    ]
    write_table(args.output_dir / "selected_domains_suspicious.csv", selected)
    write_table(args.output_dir / "extra_domains_by_capacity.csv", extra)
    write_table(args.output_dir / "extra_domains_clean_candidates.csv", clean_candidates)
    summary = {
        "selected": {"documents": len(selected_ids), "resolved_documents": len(selected_domain_map), "domains": len(selected)},
        "extra": {"documents": len(extra_ids), "resolved_documents": len(extra_domain_map), "domains": len(extra)},
        "clean_candidate_domains": len(clean_candidates),
        "candidate_thresholds": {"minimum_documents": 5, "maximum_strong_marker_chunk_rate": 0.002, "maximum_dense_strong_marker_chunk_rate": 0.0005},
    }
    (args.output_dir / "domain_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
