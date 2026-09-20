# Dataset guide

## Objective

Prepare reproducible Italian text for a small model learning grammar, syntax and
factual/conceptual content. Use FineWeb2 (`ita_Latn`), Wikipedia (`20231101.it`) and
FinePDFs-Edu (`ita_Latn`); Italian-PD is excluded. FineWiki (`it`, August 2025)
can supplement Wikipedia when its cleaned quota is underfilled. Freeze cleaned text independently
of the tokenizer so it can be reused. Keep a growing source-tagged training pool;
choose dataset size and source weights afterwards, without recleaning it.

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
- **Retain the pool:** keep all accepted candidates in the global index, including
  those beyond a particular mixture's quotas. Export every training-assigned
  document, keeping evaluation shards fixed and unused holdout candidates reserved.
- **Select a version:** choose training size and source weights from that pool.
  Selection retains whole documents and source tags; it fails if a source lacks
  sufficient text. Different training mixtures share the same evaluation sets.
- **Write and verify:** record policy, decisions and checksums; verify text shards
  before tokenizer training or tokenization.
- **Recover shortfalls:** add frozen candidates with the same quality policy and
  global duplicate index. Preserve existing shards and filled holdouts; canonical
  Wikipedia page IDs prevent newer revisions of held-out articles entering train.
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
- `supplements/` (when used): additional raw inputs and separate decision logs;
  manifest entries record provenance, processed counts and checksums. Existing
  accepted documents take priority over supplements. Cleaning stops once the
  supplemented source's budgets are filled; remaining raw candidates are unused.
  Use `--retain-all-candidates` to continue cleaning and indexing the entire frozen
  input, then export the enlarged pool. `--skip-raw-documents` records a continuation
  offset into the same input.

Reusable pool versions live under `artifacts/corpora/`; selected datasets remain
under `artifacts/documents/`. Every text row includes `source`, `document_id`,
`chunk_index` and `text_sha256`. Pool manifests record source revisions and cleaning
policy. Existing strict documents stay unchanged; any future relaxed policy needs
separate calibration and explicit provenance before promotion.

For training byte capacities `B_source` and weights `w_source`, the largest mixture
is approximately `min(B_source / w_source)` bytes over positive weights. Divide by
the estimated bytes/token for an approximate token budget; exact quotas are only
known after tokenization. Downloaded candidate counts are not usable capacity.

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
| `supplement_documents()` / `supplement-documents` | Fill missing budgets from verified raw candidates into a separate output; retain existing shards and holdouts. |
| `document_pool.export_document_pool()` / `pool-export` | Export all accepted training documents from the closed global index; preserve evaluation shards. |
| `document_pool.select_document_pool()` / `pool-select` | Choose `--train-tokens`, `--weights` and `--bytes-per-token` without recleaning; preserve evaluation shards. |
| `clean_source_document()` | Shared quality checks and chunk selection, with rejection reasons. |
| `GlobalDeduplicator.add()` | Global exact document/chunk uniqueness. |
| `assign_partition()` | Deterministic content-based split assignment. |
| `create_temporary_splits()` / `split-experiment` | Bounded, seeded experiments using parent training data only. |
| `iter_materialized_chunks()` | Read manifest-listed shards and verify checksums. |

`build_dataset.py --documents-dir …` encodes frozen text;
`tokenizer/train_tokenizer.py --documents-dir …` trains on its training partition.
For a `pool-select` output, use `clean_dataset.py --documents-dir … --mix selected`
for tokenization and tokenizer training's `--mixture selected` to read its chosen
weights from the manifest. Exact token quotas may require more text than the byte estimate.
See [README](../README.md#dataset-preparation) for setup and commands.

## Optional separate extra pool

Run manually; repeat the same command to resume:

```bash
.venv/bin/python src/dataset/clean_dataset.py extra --max-shards 3
.venv/bin/python src/dataset/clean_dataset.py extra --status
```

`extra_dataset.py` collects FineWeb2, Wikipedia and FinePDFs-Edu training shards
in round-robin order into **`artifacts/corpora/extra/`**. It pins source revisions,
applies the current strict cleaning policy, and rejects duplicates both within
the extra pool and against `italian_strict_v2` (opened read-only). It never merges
with production data, creates production splits, or tokenizes text.

- **Limit:** 20 decimal GB of cumulative raw payload downloads across invocations,
  including retry reservations; interrupted reads can conservatively consume
  unused allowance. There is no additional shared download cache. Cleaned output,
  the index and small HTTP metadata are outside this limit. `--max-download-gb`
  can set a smaller initial limit; resume with the same value.
- **Progress:** `state.json` records pinned shard order, download allowance and
  completed batches; `dedup.sqlite` commits each document's decision and cursor
  together. Interrupted downloads resume with HTTP byte ranges.
- **Output:** `raw/` retains input Parquet files; `clean/{source}/{shard}/` contains
  source-tagged cleaned chunks and checksums. The separate `manifest.json` has
  kind `isolated_extra_pool`, not the production split format. Quality decisions
  are retained in the index's `decisions` table.
- **Controls:** `--max-shards` bounds each invocation (default 3); `--min-free-gb`
  reserves disk space (default 20); `--exclude-pool` chooses a frozen reference.
  A lock prevents concurrent collectors. Existing artifacts are never overwritten;
  policy or reference changes require a new extra-pool directory.

The shared GlotLID model must already be prepared. Collection is not scheduled
and starts only when this command is run. `extra_dataset.run()` is the main
interface; `--status` performs no collection.
