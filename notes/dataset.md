# Dataset guide

## Objective

Prepare reproducible Italian text for a small model learning grammar, syntax and
factual/conceptual content. Use FineWeb2 (`ita_Latn`), Wikipedia (`20231101.it`) and
FinePDFs-Edu (`ita_Latn`); Italian-PD is excluded. Freeze cleaned text independently
of the tokenizer so it can be reused.

## Processing steps

- **Freeze raw candidates:** pin source revisions, shuffle with a recorded seed,
  and save original text, metadata and checksums for reproducible reuse.
- **Check basic quality:** enforce source-language metadata, minimum length,
  alphabetic content and repetition limits to reject obvious extraction noise.
- **Measure Italian coverage:** apply GlotLID passage checks to every source;
  require 95% Italian non-whitespace characters. Short passages need stronger
  confidence, and unknown text counts against coverage.
- **Check prose:** require 1,000 visible characters and 60% complete-looking prose;
  reject excessive fragmented boundaries, tables and corrupted characters.
- **Normalize and chunk:** normalize whitespace, retain chunks of 200–16,000
  characters and at most eight evenly spaced chunks per document to limit size.
- **Deduplicate globally:** compare normalized documents and retained chunks across
  sources before splitting. Reject later duplicate IDs/documents or documents
  sharing an exact chunk. First passing occurrence wins: web, wiki, then PDFs.
- **Assign permanent splits:** hash document content with a split seed for roughly
  96% train / 2% validation / 2% test; keep each document's chunks together.
- **Write and verify:** record policy, decisions and checksums; verify text shards
  before tokenizer training or tokenization.
- **Optional temporary splits:** select whole documents from parent **training
  data only**, then repartition with another seed and output directory, preserving
  permanent evaluation holdouts.

These are conservative heuristics, not factual verification or guaranteed reading
order. Deduplication is exact, not fuzzy. Source-specific calibration remains
necessary, especially for Wikipedia. IlPost and Fanpage are possible later sources.

## Final output

Under `artifacts/documents/`:

- `raw/{source}/sample.parquet`: reusable original inputs and sampling manifests.
- `{train,validation,test}/{source}/part-*.parquet`: text, document ID, local chunk
  index, source and normalized text hash—the final **pre-tokenization** output.
- `manifest.json`, `decisions.jsonl`, `dedup.sqlite`: provenance, policies, seeds,
  counts, checksums, rejection evidence and the global candidate index.

Acquisition defaults to 100,000 candidates per source; use
`--source-document-limits` for per-source JSON overrides and `--workers` for
bounded parallel cleaning (about 1.6 GB model memory per worker). Unfilled source/split byte
budgets fail and preserve diagnostics; `--allow-underfilled-documents` permits
study pools. Tokenization still enforces token quotas and writes the existing
uint16 shards plus `metadata.json` under `artifacts/datasets/`. Legacy format-1
text pools must be rebuilt; existing token artifacts are unchanged.

## Main interfaces

In `src/dataset/clean_dataset.py`:

| Interface | Responsibility |
|---|---|
| `materialize_documents()` / `--documents-only` | Acquire, clean, deduplicate, split and save; optionally reuse `--raw-documents-dir`. |
| `clean_source_document()` | Shared quality checks and chunk selection, with rejection reasons. |
| `GlobalDeduplicator.add()` | Global exact document/chunk uniqueness. |
| `assign_partition()` | Deterministic content-based split assignment. |
| `create_temporary_splits()` / `split-experiment` | Bounded, seeded experiments using parent training data only. |
| `iter_materialized_chunks()` | Read manifest-listed shards and verify checksums. |

`build_dataset.py --documents-dir …` encodes frozen text;
`tokenizer/train_tokenizer.py --documents-dir …` trains on its training partition.
See [README](../README.md#dataset-preparation) for setup and commands.
