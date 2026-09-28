"""Create a seeded, randomized HTML document-review bundle.

The explorer is read-only. Review labels stay in browser storage until exported
as JSON; they never modify the corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import html
import json
from pathlib import Path

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCUMENTS = PROJECT_ROOT / "artifacts/documents/balanced_1b_10m_10m_sanitized_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/analysis/document_explorer.html"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents-dir", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--max-characters", type=int, default=5000)
    parser.add_argument("--partition", choices=("train", "validation", "test", "all"), default="train")
    parser.add_argument("--source", choices=("web", "wiki", "edu_pdf", "all"), default="all")
    parser.add_argument("--sampling", choices=("balanced", "uniform"), default="balanced")
    parser.add_argument("--hostname", action="append", default=[], help="sample only these exact web hostnames; repeat for multiple sites")
    return parser.parse_args()


def rows(documents: Path, partition: str, source: str):
    manifest = json.loads((documents / "manifest.json").read_text())
    partitions = manifest["partitions"] if partition == "all" else {partition: manifest["partitions"][partition]}
    for part, sources in partitions.items():
        chosen = sources if source == "all" else {source: sources[source]}
        for source_name, record in chosen.items():
            for file in record["files"]:
                path = documents / part / source_name / file["file"]
                for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
                    for row in batch.to_pylist():
                        yield part, source_name, row


def grouped(values):
    key = None
    chunks = []
    metadata = None
    for partition, source, row in values:
        current = (partition, source, row["document_id"])
        if key is not None and current != key:
            yield metadata, chunks
            chunks = []
        key = current
        metadata = dict(partition=partition, source=source, document_id=row["document_id"])
        chunks.append(row["text"] or "")
    if key is not None:
        yield metadata, chunks


def provenance(documents: Path) -> dict[str, dict[str, str]]:
    path = documents / "web_provenance.parquet"
    if not path.exists():
        return {}
    return {
        row["document_id"]: row
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16_384)
        for row in batch.to_pylist()
    }


def sample_documents(options: argparse.Namespace) -> list[dict[str, object]]:
    if options.samples <= 0 or options.max_characters <= 0:
        raise ValueError("--samples and --max-characters must be positive")
    if options.hostname and options.source not in ("web", "all"):
        raise ValueError("--hostname can only be combined with --source web or all")
    domain_map = provenance(options.documents_dir)
    if options.hostname and not domain_map:
        raise ValueError("--hostname requires a corpus with web_provenance.parquet")
    selected_hosts = {name.casefold().removeprefix("www.") for name in options.hostname}
    sampling = "uniform" if selected_hosts else options.sampling
    groups = ("retained_web", "replacement_web", "wiki", "edu_pdf")
    if sampling == "uniform":
        quotas = {"all": options.samples}
    else:
        active = groups if options.source == "all" else (
            ("retained_web", "replacement_web") if options.source == "web" else (options.source,)
        )
        weights = {"retained_web": 2, "replacement_web": 2, "wiki": 1, "edu_pdf": 1}
        total_weight = sum(weights[group] for group in active)
        quotas = {group: options.samples * weights[group] // total_weight for group in active}
        for group in active[: options.samples - sum(quotas.values())]:
            quotas[group] += 1
    heaps: dict[str, list[tuple[int, dict[str, object]]]] = {group: [] for group in quotas}
    source = "web" if selected_hosts else options.source
    for metadata, chunks in grouped(rows(options.documents_dir, options.partition, source)):
        extra = domain_map.get(metadata["document_id"], {})
        if selected_hosts and extra.get("domain") not in selected_hosts:
            continue
        score = int.from_bytes(
            hashlib.sha256(f"{options.seed}:{metadata['partition']}:{metadata['source']}:{metadata['document_id']}".encode()).digest()[:8],
            "big",
        )
        text = "\n\n".join(chunks)
        item = {
            **metadata,
            "domain": extra.get("domain", ""),
            "origin": extra.get("origin", ""),
            "category": extra.get("category", ""),
            "characters": len(text),
            "chunks": len(chunks),
            "text": text[: options.max_characters],
            "truncated": len(text) > options.max_characters,
        }
        group = ("replacement_web" if extra.get("origin") == "extra" else "retained_web") if metadata["source"] == "web" else metadata["source"]
        if sampling == "uniform":
            group = "all"
        if group not in heaps:
            continue
        heap = heaps[group]
        if len(heap) < quotas[group]:
            heapq.heappush(heap, (-score, item))
        elif score < -heap[0][0]:
            heapq.heapreplace(heap, (-score, item))
    return [item for group in quotas for _, item in sorted(heaps[group], key=lambda pair: -pair[0])]


def render(options: argparse.Namespace, documents: list[dict[str, object]]) -> str:
    payload = json.dumps(documents, ensure_ascii=False).replace("<", "\\u003c")
    title = html.escape(f"Pavullmo document review · seed {options.seed}")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
:root{{color-scheme:light dark;font-family:system-ui,sans-serif}}body{{max-width:980px;margin:auto;padding:20px}}
.bar{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}}button,select,input,textarea{{font:inherit;padding:8px}}
button[aria-pressed=true]{{outline:3px solid Highlight}}.meta{{color:GrayText}}pre{{white-space:pre-wrap;font:16px/1.55 ui-serif,serif}}
.review{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}textarea{{width:100%;box-sizing:border-box}}
@media(max-width:600px){{.review{{grid-template-columns:1fr 1fr}}}}
</style></head><body>
<h1>{title}</h1><div class="bar"><button id="prev">Previous</button><button id="next">Next</button>
<label>Jump <input id="jump" type="number" min="1" max="{len(documents)}" value="1"></label>
<label>Show <select id="filter"><option value="all">All</option><option value="unreviewed">Unreviewed</option><option value="keep">Keep</option><option value="reject">Reject</option><option value="unsure">Unsure</option></select></label>
<button id="export">Export reviews</button></div><p id="status" aria-live="polite"></p><p id="meta" class="meta"></p>
<div class="review"><button data-label="keep">Keep</button><button data-label="reject">Reject</button><button data-label="unsure">Unsure</button><button data-label="clear">Clear</button></div>
<label>Notes<textarea id="notes" rows="3"></textarea></label><pre id="text"></pre>
<script>const docs={payload}; const key='pavullmo-review-{options.seed}'; let reviews=JSON.parse(localStorage.getItem(key)||'{{}}');let index=0;
const $=id=>document.getElementById(id);function eligible(i){{const f=$('filter').value,r=reviews[docs[i].document_id];return f==='all'||(f==='unreviewed'&&!r?.label)||r?.label===f}}
function move(step){{for(let n=0;n<docs.length;n++){{index=(index+step+docs.length)%docs.length;if(eligible(index))break}}render()}}
function render(){{const d=docs[index],r=reviews[d.document_id]||{{}};$('jump').value=index+1;$('status').textContent=`${{index+1}} / ${{docs.length}} · reviewed ${{Object.values(reviews).filter(x=>x.label).length}}`;
$('meta').textContent=[d.partition,d.source,d.domain,d.origin,d.category,`${{d.characters}} chars`,`${{d.chunks}} chunks`,d.truncated?'preview truncated':''].filter(Boolean).join(' · ');$('text').textContent=d.text;$('notes').value=r.notes||'';
document.querySelectorAll('[data-label]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.label===r.label)))}}
function save(label){{const d=docs[index];if(label==='clear')delete reviews[d.document_id];else reviews[d.document_id]={{label,notes:$('notes').value,document_id:d.document_id,partition:d.partition,source:d.source,domain:d.domain}};localStorage.setItem(key,JSON.stringify(reviews));move(1)}}
$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('jump').onchange=e=>{{index=Math.max(0,Math.min(docs.length-1,+e.target.value-1));render()}};$('filter').onchange=()=>move(1);
document.querySelectorAll('[data-label]').forEach(b=>b.onclick=()=>save(b.dataset.label));$('notes').onchange=()=>{{const d=docs[index],r=reviews[d.document_id]||{{}};reviews[d.document_id]={{...r,notes:$('notes').value}};localStorage.setItem(key,JSON.stringify(reviews))}};
$('export').onclick=()=>{{const blob=new Blob([JSON.stringify({{seed:{options.seed},reviews}},null,2)],{{type:'application/json'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='pavullmo-review-{options.seed}.json';a.click();URL.revokeObjectURL(a.href)}};render();</script>
</body></html>"""


def main() -> None:
    options = arguments()
    documents = sample_documents(options)
    if not documents:
        raise RuntimeError("no documents matched the requested partition/source")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(render(options, documents), encoding="utf-8")
    print(json.dumps(dict(output=str(options.output), samples=len(documents), seed=options.seed), indent=2))


if __name__ == "__main__":
    main()
