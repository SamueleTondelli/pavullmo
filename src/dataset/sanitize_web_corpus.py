"""Build a versioned web-sanitized corpus from a frozen format-2 corpus.

The parent is never modified.  Web documents from quarantined domains are
removed and replaced with globally deduplicated documents from the isolated
extra pool.  Replacement documents keep the parent's deterministic split and
are selected across topical domain groups with per-group byte targets.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

try:
    import clean_dataset as c
except ModuleNotFoundError:
    from src.dataset import clean_dataset as c


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARENT = PROJECT_ROOT / "artifacts/documents/balanced_1b_10m_10m"
DEFAULT_EXTRA = PROJECT_ROOT / "artifacts/corpora/extra"
DEFAULT_ANALYSIS = PROJECT_ROOT / "artifacts/analysis/domain_replacement"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/documents/balanced_1b_10m_10m_sanitized_v1"
GROUP_WEIGHTS = {
    "national_international": 0.17,
    "regional_local_news": 0.14,
    "technology_science": 0.18,
    "environment_health_social": 0.11,
    "law_economics_institutions": 0.14,
    "education_culture_travel": 0.05,
    "sports_motoring": 0.09,
    "optional_reserve": 0.12,
}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--extra-dir", type=Path, default=DEFAULT_EXTRA)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--policy-version", default="pessimistic-domain-v1")
    parser.add_argument("--selection-seed", default="sanitized-v1")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--repair-index", action="store_true", help="restore canonical document hashes in an existing output")
    return parser.parse_args()


def domain_of(url: object) -> str:
    try:
        domain = (urlsplit(str(url or "")).hostname or "").casefold().rstrip(".")
    except ValueError:
        return ""
    return domain[4:] if domain.startswith("www.") else domain


def load_shortlist(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["domain"]: row["category"] for row in rows}
    if not result or set(result.values()) - set(GROUP_WEIGHTS):
        raise ValueError("replacement shortlist is empty or has unknown categories")
    return result


def restore_parent_domains(parent: Path, selected_ids: set[str]) -> dict[str, str]:
    result = {}
    raw = parent / "raw/web/sample.parquet"
    for batch in pq.ParquetFile(raw).iter_batches(
        columns=["document_id", "metadata_json"], batch_size=16_384
    ):
        for row in batch.to_pylist():
            if row["document_id"] in selected_ids:
                result[row["document_id"]] = domain_of(
                    json.loads(row["metadata_json"]).get("url")
                )
    if len(result) != len(selected_ids):
        raise ValueError(f"unresolved parent domains: {len(selected_ids) - len(result)}")
    return result


def restore_extra_domains(extra: Path, shortlist: dict[str, str]) -> dict[str, tuple[str, str]]:
    result = {}
    for raw in sorted((extra / "raw").glob("*.parquet")):
        parquet = pq.ParquetFile(raw)
        if not {"id", "url"}.issubset(parquet.schema_arrow.names):
            continue
        for batch in parquet.iter_batches(columns=["id", "url"], batch_size=16_384):
            for row in batch.to_pylist():
                domain = domain_of(row["url"])
                if domain in shortlist:
                    result[f"web:id:{row['id']}"] = (domain, shortlist[domain])
    return result


def parent_rows(parent: Path, partition: str, source: str):
    record = json.loads((parent / "manifest.json").read_text())["partitions"][partition][source]
    for file in record["files"]:
        path = parent / partition / source / file["file"]
        if c.sha256_file(path) != file["sha256"]:
            raise ValueError(f"parent shard checksum mismatch: {path}")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            yield from batch.to_pylist()


def grouped(rows):
    current = None
    buffer = []
    for row in rows:
        if current is not None and row["document_id"] != current:
            yield current, buffer
            buffer = []
        current = row["document_id"]
        buffer.append(row)
    if current is not None:
        yield current, buffer


def clone_nonweb(parent: Path, temporary: Path, manifest: dict) -> None:
    for partition, sources in manifest["partitions"].items():
        for source in sources:
            if source == "web":
                continue
            destination = temporary / partition / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(parent / partition / source, destination, copy_function=os.link)


def extra_candidates(extra: Path, domain_map: dict[str, tuple[str, str]], split: dict, selection_seed: str):
    db = sqlite3.connect(f"file:{extra / 'dedup.sqlite'}?mode=ro", uri=True)
    result = defaultdict(list)
    try:
        for doc_hash, source, doc_id, chunks_json in db.execute(
            "SELECT hash,source,id,chunks FROM documents WHERE source='web'"
        ):
            provenance = domain_map.get(doc_id)
            if provenance is None:
                continue
            domain, category = provenance
            partition = c.assign_partition(
                doc_hash, split["seed"], split["test_buckets_per_1000"]
            )
            chunks = json.loads(chunks_json)
            byte_count = sum(len(text.encode("utf-8")) for text in chunks)
            rank = hashlib.sha256(f"{selection_seed}:{doc_hash}".encode()).hexdigest()
            result[partition, category].append(
                (rank, doc_hash, doc_id, domain, chunks, byte_count)
            )
    finally:
        db.close()
    for values in result.values():
        values.sort()
    return result


def choose_replacements(candidates, partition: str, deficit: int):
    targets = {name: deficit * weight for name, weight in GROUP_WEIGHTS.items()}
    positions = defaultdict(int)
    filled = defaultdict(int)
    chosen = []
    while sum(item[5] for item in chosen) < deficit:
        available = [
            name for name in GROUP_WEIGHTS
            if positions[name] < len(candidates[partition, name])
        ]
        if not available:
            raise RuntimeError(f"replacement pool underfilled for {partition}")
        category = min(
            available,
            key=lambda name: (filled[name] / targets[name], name),
        )
        item = candidates[partition, category][positions[category]]
        positions[category] += 1
        filled[category] += item[5]
        chosen.append((*item, category))
    return chosen, dict(filled)


def restore_canonical_hashes(index_path: Path, parent: Path, extra: Path) -> dict[str, int]:
    """Restore raw-document hashes in the rebuilt index after chunk uniqueness checks."""
    db = sqlite3.connect(index_path)
    try:
        db.execute("ATTACH DATABASE ? AS parent", (str(parent / "dedup.sqlite"),))
        db.execute("ATTACH DATABASE ? AS extra", (str(extra / "dedup.sqlite"),))
        missing = db.execute(
            """SELECT COUNT(*) FROM documents AS s
               WHERE NOT EXISTS (SELECT 1 FROM parent.documents AS p
                                 WHERE p.source=s.source AND p.id=s.id)
                 AND NOT EXISTS (SELECT 1 FROM extra.documents AS e
                                 WHERE e.source=s.source AND e.id=s.id)"""
        ).fetchone()[0]
        if missing:
            raise RuntimeError(f"{missing} sanitized documents have no canonical source index")
        changed = db.execute(
            """SELECT COUNT(*) FROM documents AS s
               WHERE s.hash != COALESCE(
                 (SELECT p.hash FROM parent.documents AS p WHERE p.source=s.source AND p.id=s.id),
                 (SELECT e.hash FROM extra.documents AS e WHERE e.source=s.source AND e.id=s.id))"""
        ).fetchone()[0]
        with db:
            db.execute("CREATE TEMP TABLE canonical_map (old_hash TEXT PRIMARY KEY, new_hash TEXT NOT NULL)")
            db.execute(
                """INSERT INTO canonical_map
                   SELECT s.hash, COALESCE(
                     (SELECT p.hash FROM parent.documents AS p WHERE p.source=s.source AND p.id=s.id),
                     (SELECT e.hash FROM extra.documents AS e WHERE e.source=s.source AND e.id=s.id))
                   FROM documents AS s"""
            )
            db.execute(
                """UPDATE chunks SET document_hash=(
                     SELECT m.new_hash FROM canonical_map AS m WHERE m.old_hash=chunks.document_hash)
                   WHERE document_hash IN (SELECT old_hash FROM canonical_map)"""
            )
            db.execute(
                """UPDATE documents SET hash=(
                     SELECT m.new_hash FROM canonical_map AS m WHERE m.old_hash=documents.hash)"""
            )
            db.execute("DROP TABLE canonical_map")
        return {"documents_without_source": missing, "canonical_hashes_restored": changed}
    finally:
        db.close()


def main() -> None:
    options = args()
    parent, extra, analysis, output = map(
        Path.resolve, (options.parent, options.extra_dir, options.analysis_dir, options.output_dir)
    )
    temporary = output.with_name(f".{output.name}.tmp")
    if options.repair_index:
        if not output.is_dir():
            raise FileNotFoundError(output)
        print(json.dumps(restore_canonical_hashes(output / "dedup.sqlite", parent, extra), indent=2))
        return
    if output.exists() or temporary.exists():
        if not options.overwrite:
            raise FileExistsError(f"output exists: {output} or {temporary}")
        if output.exists():
            shutil.rmtree(output)
        if temporary.exists():
            shutil.rmtree(temporary)
    manifest = json.loads((parent / "manifest.json").read_text())
    quarantine = set((analysis / "quarantine_domains.txt").read_text().splitlines())
    shortlist = load_shortlist(analysis / "replacement_domain_shortlist.csv")
    selected_ids = {
        row["document_id"]
        for partition in manifest["partitions"]
        for _, rows in grouped(parent_rows(parent, partition, "web"))
        for row in rows[:1]
    }
    parent_domain_map = restore_parent_domains(parent, selected_ids)
    extra_domain_map = restore_extra_domains(extra, shortlist)
    candidates = extra_candidates(extra, extra_domain_map, manifest["partitioning"], options.selection_seed)

    temporary.mkdir(parents=True)
    clone_nonweb(parent, temporary, manifest)
    provenance = []
    replacement_log = []
    new_web_records = {}
    for partition in manifest["partitions"]:
        writer = c.MaterializedParquetWriter(
            temporary / partition / "web", c.DEFAULT_DOCUMENT_SHARD_BYTES
        )
        retained_documents = 0
        removed_documents = 0
        removed_bytes = 0
        for doc_id, rows in grouped(parent_rows(parent, partition, "web")):
            domain = parent_domain_map[doc_id]
            if domain in quarantine:
                removed_documents += 1
                removed_bytes += sum(len(row["text"].encode("utf-8")) for row in rows)
                continue
            retained_documents += 1
            provenance.append((doc_id, domain, "parent", "retained", partition))
            for row in rows:
                writer.write(row, len(row["text"].encode("utf-8")))
        target = manifest["partitions"][partition]["web"]["target_text_bytes"]
        deficit = max(0, target - writer.total_text_bytes)
        chosen, category_bytes = choose_replacements(candidates, partition, deficit)
        for _, doc_hash, doc_id, domain, chunks, byte_count, category in chosen:
            provenance.append((doc_id, domain, "extra", category, partition))
            replacement_log.append(
                dict(partition=partition, document_id=doc_id, domain=domain,
                     category=category, normalized_document_hash=doc_hash,
                     chunks=len(chunks), utf8_bytes=byte_count)
            )
            for index, text in enumerate(chunks):
                writer.write(
                    dict(document_id=doc_id, chunk_index=index, source="web",
                         text_sha256=c.normalized_text_hash(text), text=text),
                    len(text.encode("utf-8")),
                )
        writer.close()
        new_web_records[partition] = dict(
            target_text_bytes=target,
            text_bytes=writer.total_text_bytes,
            chunks=writer.total_chunks,
            documents=retained_documents + len(chosen),
            files=writer.files,
            retained_documents=retained_documents,
            removed_documents=removed_documents,
            removed_text_bytes=removed_bytes,
            replacement_documents=len(chosen),
            replacement_text_bytes=sum(item[5] for item in chosen),
            replacement_category_bytes=category_bytes,
        )

    for partition, record in new_web_records.items():
        manifest["partitions"][partition]["web"] = record
    manifest["underfilled"] = [
        f"{partition}/{source}"
        for partition, sources in manifest["partitions"].items()
        for source, record in sources.items()
        if record["text_bytes"] < record["target_text_bytes"]
    ]
    if manifest["underfilled"]:
        raise RuntimeError(f"sanitized corpus underfilled: {manifest['underfilled']}")
    manifest["sanitization"] = dict(
        version=options.policy_version,
        parent=str(parent),
        parent_manifest_sha256=c.sha256_file(parent / "manifest.json"),
        extra_pool=str(extra),
        extra_manifest_sha256=c.sha256_file(extra / "manifest.json"),
        quarantine_domains=len(quarantine),
        quarantine_sha256=c.sha256_file(analysis / "quarantine_domains.txt"),
        replacement_shortlist_sha256=c.sha256_file(analysis / "replacement_domain_shortlist.csv"),
        group_weights=GROUP_WEIGHTS,
        selection_order=f"sha256({options.selection_seed}:normalized_document_hash), balanced by category bytes",
        code_sha256=c.sha256_file(Path(__file__)),
    )
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (temporary / "replacements.jsonl").open("w", encoding="utf-8") as handle:
        for row in replacement_log:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    provenance_schema = pa.schema([
        ("document_id", pa.string()), ("domain", pa.string()),
        ("origin", pa.string()), ("category", pa.string()), ("partition", pa.string()),
    ])
    pq.write_table(
        pa.Table.from_pylist([
            dict(zip(("document_id", "domain", "origin", "category", "partition"), row, strict=True))
            for row in provenance
        ], schema=provenance_schema),
        temporary / "web_provenance.parquet", compression="zstd",
    )

    dedup = c.GlobalDeduplicator(temporary / "dedup.sqlite", resume=False)
    try:
        with dedup.db:
            for partition, sources in manifest["partitions"].items():
                for source in sources:
                    for doc_id, rows in grouped(parent_rows(temporary, partition, source)):
                        chunks = [row["text"] for row in rows]
                        full_text = "\n\n".join(chunks)
                        kept, reason, _ = dedup.add(
                            source, doc_id, full_text, chunks, commit=False
                        )
                        if not kept:
                            raise RuntimeError(f"duplicate in sanitized corpus: {doc_id}: {reason}")
    finally:
        dedup.close()
    restore_canonical_hashes(temporary / "dedup.sqlite", parent, extra)
    temporary.rename(output)
    print(json.dumps({
        "output": str(output), "quarantine_domains": len(quarantine),
        "web": new_web_records, "replacement_documents": len(replacement_log),
    }, indent=2))


if __name__ == "__main__":
    main()
