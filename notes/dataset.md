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

## Web-quality audit and next filters

The `balanced_1b_10m_10m` audit found a distributed explicit SEO-spam network,
not one dominant adult domain. The conservative successor policy quarantines a
domain when it contributes at least two densely explicit chunks and at least 50%
of its selected chunks are densely explicit; all `bakeca.it` subdomains are also
excluded. Keep the parent immutable and record domain provenance for retained and
replacement documents.

The first sanitized successor is
`artifacts/documents/balanced_1b_10m_10m_sanitized_v1/`. It is a document pool,
not yet a retokenized 1B/10M/10M training artifact. Its web provenance is in
`web_provenance.parquet`, and the exact selected replacements are in
`replacements.jsonl`. The 300-document review page at
`artifacts/analysis/document_explorer.html` samples 100 retained web, 100
replacement web, 50 Wikipedia, and 50 educational-PDF documents; labels and
notes remain in the browser until exported. The original web still has
117,538 distinct retained domains after quarantine, including large shares
from `fanpage.it`, `it.topwar.ru`, and `ilgiornale.it`. Dense explicit-marker
chunks remain, especially in `sbandieratoridovara.it`; domain filtering is
therefore a first pass, not a final safety certificate.
The train-web lexical audit fell from 18,336/602,045 dense explicit-marker
chunks (3.05%) in the parent to 1,260/620,781 (0.20%) in this successor.
These are heuristic signals, not human-labeled adult-content rates.

The second sanitized successor is
`artifacts/documents/balanced_1b_10m_10m_sanitized_v2/`, built from the
original frozen selection with the additional conservative site rule in
`src/dataset/refine_web_quarantine.py`. Its versioned evidence and exact
1,252-domain quarantine are under `artifacts/analysis/domain_replacement_v2/`.
The rule adds 187 hostnames based on explicit hostname patterns or repeated
dense explicit markers (at least 3 dense chunks, at least 5% of a site's
chunks dense, and at least 20% containing a strong marker). It removes 18,337
original train-web documents and selects 40,425 replacement documents from
the deduplicated extra pool across multiple site categories. The replacement
selection and provenance are recorded in `replacements.jsonl` and
`web_provenance.parquet`; no quarantined hostname remains. All 26 document
shard checksums, split assignments, and deduplication index references were
verified. The train-web lexical audit is in
`artifacts/analysis/adult_content_sanitized_v2/`: 1,002/623,869 (0.161%)
chunks still meet its dense explicit-marker heuristic. This rule is not a
document-level adult-content filter or a safety guarantee.

The corresponding 16K-tokenizer balanced token artifact is built with:

```sh
.venv/bin/python src/dataset/build_dataset.py \
  --documents-dir artifacts/documents/balanced_1b_10m_10m_sanitized_v2 \
  --tokenizer artifacts/tokenizers/production_16k/tokenizer.model \
  --output-dir artifacts/datasets/balanced_sanitized_v2_16k \
  --mix balanced --train-tokens 1000000000 \
  --validation-tokens 10000000 --test-tokens 10000000
```

This leaves all prior document pools and token artifacts untouched. Inspect
the three `metadata.json` files for final token counts and shard hashes before
training; the path is a distinct candidate and is not the default dataset path.

For a complete source-site inventory, run
`.venv/bin/python src/dataset/inventory_websites.py --output-dir artifacts/analysis/website_inventory_next`
with a new output directory.
The resulting `websites.csv` has one row per source type and URL hostname for
all accepted documents in the reusable pool, original frozen selection,
sanitized successor, and isolated extra pool. Counts are distinct documents;
subdomains remain separate except for `www.`, and the columns preserve the
three train/validation/test partitions. Three example URLs and document IDs
per row support manual review. Blank decision/notes columns are for reviewers;
editing the inventory does not filter any corpus.
To inspect one hostname in the sanitized corpus, run the document explorer with
`--hostname site.example --samples 100 --output artifacts/analysis/site-example-review.html`;
repeat `--hostname` to combine sites. This is a read-only seeded sample, not a
filtering decision.

The next filtering iteration should be staged and ablated rather than folded into
one opaque score:

1. URL/domain deny rules for confirmed spam networks and adult paths, versioned
   with evidence and counts.
2. Existing language, prose, exact-document and exact-chunk checks.
3. Template/repetition checks at line, paragraph and word n-gram level; specifically
   measure dominant repeated spans across unrelated domains.
4. Near-duplicate MinHash over language-tokenized word 5-grams, evaluated both
   within acquisition shards and globally. Do not assume the most aggressive scope
   is best.
5. A lightweight Italian quality classifier trained from manually reviewed
   positive/negative documents, with held-out labels and small-model pretraining
   ablations before promotion.
6. Seeded human review of random accepted and rejected documents, stratified by
   source/domain and including boundary-score examples.

This ordering follows the reproducible lessons from FineWeb/FineWeb2, RefinedWeb,
Dolma, DataComp-LM and MADLAD-400: combine provenance, simple interpretable rules,
repetition handling, deduplication and empirical model-based filtering; retain
manual audits; and validate thresholds per language. In particular, FineWeb
reported that more cross-dump or line-level deduplication was not automatically
better, while DataComp-LM found that a small learned quality filter was a strong
baseline. Relevant primary references:

- [FineWeb technical report](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)
- [FineWeb2](https://arxiv.org/abs/2506.20920)
- [RefinedWeb](https://arxiv.org/abs/2306.01116)
- [Dolma](https://arxiv.org/abs/2402.00159)
- [DataComp-LM](https://arxiv.org/abs/2406.11794)
- [MADLAD-400](https://arxiv.org/abs/2309.04662)
