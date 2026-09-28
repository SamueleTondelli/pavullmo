"""Export every URL hostname represented in the local accepted document corpora.

This is an inventory, not a blocklist. Hostnames retain subdomains except for
the common ``www.`` prefix. Counts are documents, not token estimates.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "artifacts/corpora/italian_strict_v2"
ORIGINAL = ROOT / "artifacts/documents/balanced_1b_10m_10m"
SANITIZED = ROOT / "artifacts/documents/balanced_1b_10m_10m_sanitized_v1"
EXTRA = ROOT / "artifacts/corpora/extra"
OUTPUT = ROOT / "artifacts/analysis/website_inventory"

BITS = {
    "pool_train": 1 << 0,
    "pool_validation": 1 << 1,
    "pool_test": 1 << 2,
    "original_train": 1 << 3,
    "original_validation": 1 << 4,
    "original_test": 1 << 5,
    "sanitized_train": 1 << 6,
    "sanitized_validation": 1 << 7,
    "sanitized_test": 1 << 8,
    "extra": 1 << 9,
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=POOL)
    parser.add_argument("--original", type=Path, default=ORIGINAL)
    parser.add_argument("--sanitized", type=Path, default=SANITIZED)
    parser.add_argument("--extra", type=Path, default=EXTRA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def hostname(url: object) -> str:
    try:
        host = (urlsplit(str(url or "")).hostname or "").casefold().rstrip(".")
    except ValueError:
        return "(invalid URL)"
    if host.startswith("www."):
        host = host[4:]
    return host or "(missing URL)"


def add_materialized(membership: dict[str, int], root: Path, prefix: str) -> Counter[str]:
    manifest = json.loads((root / "manifest.json").read_text())
    totals: Counter[str] = Counter()
    for partition, sources in manifest["partitions"].items():
        bit = BITS[f"{prefix}_{partition}"]
        for source, record in sources.items():
            last_id = None
            for file in record["files"]:
                path = root / partition / source / file["file"]
                for batch in pq.ParquetFile(path).iter_batches(columns=["document_id"], batch_size=32768):
                    for doc_id in batch.column(0).to_pylist():
                        if doc_id == last_id:
                            continue
                        last_id = doc_id
                        if membership.get(doc_id, 0) & bit:
                            continue
                        membership[doc_id] = membership.get(doc_id, 0) | bit
                        totals[f"{prefix}_{partition}_{source}"] += 1
            expected = record.get("documents")
            if expected is not None and totals[f"{prefix}_{partition}_{source}"] != expected:
                raise ValueError(f"document count mismatch: {root}/{partition}/{source}")
    return totals


def add_extra(membership: dict[str, int], root: Path) -> Counter[str]:
    manifest = json.loads((root / "manifest.json").read_text())
    totals: Counter[str] = Counter()
    for batch_info in manifest["batches"]:
        source = batch_info["source"]
        last_id = None
        for file in batch_info["files"]:
            path = root / "clean" / source / Path(batch_info["raw_file"]).stem / file["file"]
            for batch in pq.ParquetFile(path).iter_batches(columns=["document_id"], batch_size=32768):
                for doc_id in batch.column(0).to_pylist():
                    if doc_id == last_id:
                        continue
                    last_id = doc_id
                    if membership.get(doc_id, 0) & BITS["extra"]:
                        continue
                    membership[doc_id] = membership.get(doc_id, 0) | BITS["extra"]
                    totals[f"extra_{source}"] += 1
        if totals[f"extra_{source}"] != batch_info["accepted_documents"]:
            raise ValueError(f"extra count mismatch: {source}")
    return totals


def raw_rows(path: Path, source: str, extra: bool):
    parquet = pq.ParquetFile(path)
    if extra:
        if not {"id", "url"}.issubset(parquet.schema_arrow.names):
            return
        for batch in parquet.iter_batches(columns=["id", "url"], batch_size=32768):
            for row in batch.to_pylist():
                yield f"{source}:id:{row['id']}", row["url"]
    else:
        for batch in parquet.iter_batches(columns=["document_id", "metadata_json"], batch_size=32768):
            for row in batch.to_pylist():
                doc_id = row["document_id"]
                yield doc_id, row["metadata_json"]


def main() -> None:
    args = arguments()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"inventory output exists: {output}")
    membership: dict[str, int] = {}
    totals = Counter()
    for prefix, root in (("pool", args.pool), ("original", args.original), ("sanitized", args.sanitized)):
        totals.update(add_materialized(membership, root, prefix))
    totals.update(add_extra(membership, args.extra))
    print(f"Accepted document memberships loaded: {len(membership):,}", flush=True)

    websites: dict[tuple[str, str], dict[str, object]] = {}
    missing_by_source: Counter[str] = Counter()
    for source in ("web", "wiki", "edu_pdf"):
        paths = [args.pool / "raw" / source / "sample.parquet"]
        paths.extend(sorted((args.pool / "supplements").glob(f"*/{source}/sample.parquet")))
        for path in paths:
            for doc_id, raw_meta in raw_rows(path, source, extra=False):
                mask = membership.pop(doc_id, 0)
                if not mask:
                    continue
                url = json.loads(raw_meta or "{}").get("url")
                host = hostname(url)
                if host.startswith("("):
                    missing_by_source[f"pool_{source}"] += 1
                key = (source, host)
                record = websites.setdefault(key, {"source": source, "hostname": host, "counts": Counter(), "examples": []})
                for name, bit in BITS.items():
                    if mask & bit:
                        record["counts"][name] += 1
                if url and len(record["examples"]) < 3:
                    record["examples"].append((url, doc_id))
        print(f"Resolved parent raw URLs: {source}", flush=True)

    for batch_info in json.loads((args.extra / "manifest.json").read_text())["batches"]:
        if not batch_info["accepted_documents"]:
            continue
        source = batch_info["source"]
        path = args.extra / batch_info["raw_file"]
        for doc_id, url in raw_rows(path, source, extra=True):
            mask = membership.pop(doc_id, 0)
            if not mask:
                continue
            host = hostname(url)
            if host.startswith("("):
                missing_by_source[f"extra_{source}"] += 1
            key = (source, host)
            record = websites.setdefault(key, {"source": source, "hostname": host, "counts": Counter(), "examples": []})
            for name, bit in BITS.items():
                if mask & bit:
                    record["counts"][name] += 1
            if url and len(record["examples"]) < 3:
                record["examples"].append((url, doc_id))
        print(f"Resolved extra raw URLs: {source}", flush=True)

    if membership:
        by_prefix = Counter(doc_id.split(":", 1)[0] for doc_id in membership)
        raise ValueError(f"accepted document IDs missing raw URL provenance: {dict(by_prefix)}")

    output.mkdir(parents=True)
    fields = ["source", "hostname", *BITS, "original_total", "sanitized_total", "pool_total", "extra_total",
              "sample_url_1", "sample_document_id_1", "sample_url_2", "sample_document_id_2",
              "sample_url_3", "sample_document_id_3", "review_decision", "review_notes"]
    csv_path = output / "websites.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for _, record in sorted(websites.items(), key=lambda pair: pair[0]):
            counts = record["counts"]
            row = {"source": record["source"], "hostname": record["hostname"]}
            row.update({name: counts[name] for name in BITS})
            for prefix in ("original", "sanitized", "pool"):
                row[f"{prefix}_total"] = sum(counts[f"{prefix}_{part}"] for part in ("train", "validation", "test"))
            row["extra_total"] = counts["extra"]
            for index, (url, doc_id) in enumerate(record["examples"], start=1):
                row[f"sample_url_{index}"] = url
                row[f"sample_document_id_{index}"] = doc_id
            writer.writerow(row)
    summary = {
        "scope": "Accepted documents only; hostnames keep subdomains except www; source distinguishes web, wiki, edu_pdf",
        "website_source_rows": len(websites),
        "unique_hostnames": len({host for _, host in websites}),
        "document_counts": dict(totals),
        "unparsed_url_documents": dict(missing_by_source),
        "input_manifests": {name: str((path / "manifest.json").resolve()) for name, path in
                            (("pool", args.pool), ("original", args.original), ("sanitized", args.sanitized), ("extra", args.extra))},
        "output": str(csv_path),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
