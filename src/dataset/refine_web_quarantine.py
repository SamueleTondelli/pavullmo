"""Freeze a second, evidence-backed site quarantine for the selected web corpus.

The rule is intentionally conservative about keeping questionable material:
obvious adult hostnames and sites with repeated dense explicit markers are
excluded. It does not inspect or modify the document corpus itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARENT = ROOT / "artifacts/analysis/domain_replacement"
DEFAULT_OUTPUT = ROOT / "artifacts/analysis/domain_replacement_v2"

# These match hostname text, not document content. The negative lookbehind
# avoids ordinary Italian words such as "possessori".
ADULT_HOSTNAME = re.compile(
    r"porn|erotic|escort|hentai|xxx|(?<!pos)sesso|sexy|(?:^|[.-])sex|(?:^|[.-])nude|(?:^|[.-])nudist",
    re.IGNORECASE,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-analysis", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = arguments()
    parent = args.parent_analysis.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    original = set((parent / "quarantine_domains.txt").read_text().splitlines())
    with (parent / "selected_domains_suspicious.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    additions = []
    for row in rows:
        domain = row["domain"]
        if domain in original or domain.startswith("("):
            continue
        reasons = []
        if ADULT_HOSTNAME.search(domain):
            reasons.append("explicit_hostname")
        if (
            int(row["dense_strong_marker_chunks"]) >= 3
            and float(row["dense_strong_marker_chunk_rate"]) >= 0.05
            and float(row["strong_marker_chunk_rate"]) >= 0.20
        ):
            reasons.append("repeated_dense_explicit_markers")
        if reasons:
            additions.append({**row, "reason": "+".join(reasons)})
    additions.sort(key=lambda row: (-int(row["utf8_bytes"]), row["domain"]))
    quarantine = original | {row["domain"] for row in additions}
    shortlist = parent / "replacement_domain_shortlist.csv"
    with shortlist.open(newline="", encoding="utf-8") as handle:
        replacement_domains = {row["domain"] for row in csv.DictReader(handle)}
    overlap = quarantine & replacement_domains
    if overlap:
        raise ValueError(f"replacement shortlist contains quarantined domains: {sorted(overlap)}")

    output.mkdir(parents=True)
    (output / "quarantine_domains.txt").write_text("\n".join(sorted(quarantine)) + "\n")
    shutil.copy2(shortlist, output / shortlist.name)
    with (output / "additional_domain_evidence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["reason", *rows[0].keys()])
        writer.writeheader()
        writer.writerows(additions)
    summary = {
        "policy": "conservative-site-v2",
        "parent_quarantine_domains": len(original),
        "additional_domains": len(additions),
        "total_quarantine_domains": len(quarantine),
        "additional_selected_documents": sum(int(row["documents"]) for row in additions),
        "additional_selected_utf8_bytes": sum(int(row["utf8_bytes"]) for row in additions),
        "additional_dense_marker_chunks": sum(int(row["dense_strong_marker_chunks"]) for row in additions),
        "explicit_hostname_pattern": ADULT_HOSTNAME.pattern,
        "dense_site_rule": "at least 3 dense chunks, at least 5% dense chunks, at least 20% strong-marker chunks",
        "limitations": [
            "Hostname matching can reject non-adult sites; evidence CSV supports review.",
            "Lexical markers can miss euphemisms and can flag legitimate medical or literary material.",
            "This is site filtering, not document-level content filtering or a final safety certificate.",
        ],
        "inputs_sha256": {
            "quarantine_domains.txt": sha256(parent / "quarantine_domains.txt"),
            "selected_domains_suspicious.csv": sha256(parent / "selected_domains_suspicious.csv"),
            "replacement_domain_shortlist.csv": sha256(shortlist),
        },
    }
    (output / "policy.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
