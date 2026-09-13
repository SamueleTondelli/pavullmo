# Dataset pipeline

The production dataset pipeline builds three comparable Italian-only
pretraining mixtures. Synthetic data is deliberately excluded for now so this
experiment isolates natural-data source composition and conservative cleaning.

The main entry point is:

```bash
uv run python dataset/build_dataset.py --overwrite
```

By default it creates three 1B-token training artifacts, one shared 10M-token
validation artifact, and one untouched 10M-token test artifact under
`dataset/ds/`.

## Sources and mixtures

Four source families are streamed from Hugging Face:

| Key | Dataset | Intended content | Main limitation |
|---|---|---|---|
| `web` | `HuggingFaceFW/fineweb-2`, `ita_Latn` | Broad contemporary Italian web | Variable genre and quality |
| `wiki` | `wikimedia/wikipedia`, `20231101.it` | Encyclopedic/reference prose | Narrow editorial style |
| `books` | `PleIAs/Italian-PD` | Public-domain long-form Italian | Historical language and OCR errors |
| `edu_pdf` | `HuggingFaceFW/finepdfs-edu`, `ita_Latn` | Educational documents extracted from PDFs | Educational does not imply academic |

The source revisions and dataset pages are recorded in every artifact's
`metadata.json`. Review upstream terms before redistributing derived data.

Percentages below are token percentages:

| Artifact | Web | Wiki | Books | Educational PDFs | Purpose |
|---|---:|---:|---:|---:|---|
| `train_web` | 70% | 10% | 10% | 10% | Broad-web baseline |
| `train_balanced` | 40% | 20% | 20% | 20% | Balanced source comparison |
| `train_knowledge` | 20% | 25% | 25% | 30% | Knowledge/long-form emphasis |
| `validation` | 40% | 20% | 20% | 20% | Shared model selection |
| `test` | 40% | 20% | 20% | 20% | Shared final evaluation |

Only the training weights differ. Architecture, optimizer, token budget,
tokenizer, validation distribution, test distribution, and training seed
should remain fixed when comparing the mixtures.

## How the data is cleaned

The production baseline uses a small set of transparent, source-aware rules:

1. **Language checks.** FineWeb2 must have the Italian label and a GlotLID
   score of at least `0.98`. FinePDFs-Edu must have the full-document Italian
   label and score at least `0.90`. Wikipedia and Italian-PD are already
   language-specific collections.
2. **Empty and tiny extraction removal.** A document needs at least 200
   characters after surrounding whitespace is removed.
3. **Text-content check.** At least 55% of visible characters must be
   alphabetic. This rejects many tables, symbol dumps, broken extractions, and
   heavily numeric pages without requiring a learned quality model.
4. **Repetition check.** No more than 30% of nonempty normalized lines may be
   repeated. This targets menus, duplicated headers, OCR loops, and templates.
5. **Long-document handling.** Accepted documents are split near paragraph
   boundaries into chunks of at most 16,000 characters. At most eight evenly
   distributed chunks are retained from one source document so one enormous
   book or PDF cannot dominate a source allocation.
6. **Exact deduplication.** Text is case-folded and whitespace-normalized before
   hashing. Exact duplicate chunks are retained once per artifact.
7. **Evaluation isolation.** A stable SHA-256 hash of the source and upstream
   document ID assigns the whole document to train, validation, or test before
   chunking. Chunks from one book or article therefore cannot cross splits.
   Exact normalized evaluation text is also excluded from training.

These rules remove obvious technical failures; they do not claim to identify
all low-quality writing. The current pipeline does not yet provide
cross-source near-deduplication, a dedicated OCR-quality model, robust PII
removal, toxicity filtering, or a verified academic-paper classifier. The
observed rejection counts and every threshold are stored in `metadata.json`.

Streams are processed one source at a time to keep memory bounded. Source
shards and small row buffers are shuffled deterministically. Although source
regions are written sequentially, the training loader shuffles token blocks
across the completed artifact.

## Tokenizer-independent document layer

The pipeline can stop after downloading, cleaning, splitting, and chunking the
documents. This freezes the corpus before committing to a tokenizer:

```bash
uv run python dataset/build_dataset.py \
  --documents-only \
  --documents-output-dir dataset/documents \
  --overwrite
```

The result is partitioned Parquet data under
`dataset/documents/{train,validation,test}/{source}/`, plus a `manifest.json`
containing source revisions, filters, split logic, mixture definitions, byte
counts, rejection counts, file hashes, and provenance.

Because no tokenizer exists at this stage, corpus sizing is necessarily based
on raw UTF-8 bytes. The default allowance is five document bytes for each
eventual requested token, separately for every source's largest required
quota. It includes all three mixtures without storing three redundant copies
of shared documents. This is a capacity estimate, not an eventual token count;
increase it with `--document-bytes-per-token` if a future tokenizer exhausts a
source pool. Parquet shard size is configurable with
`--document-shard-bytes`.

Train a new tokenizer from a deliberate mixture of the frozen training
documents:

```bash
uv run python src/tokenizer/train_tokenizer.py \
  --vocab-size 16000 \
  --byte-budget 1073741824 \
  --model-prefix src/tokenizer/next \
  --documents-dir dataset/documents \
  --mixture balanced
```

Tokenizer training accepts document chunks up to 65,536 UTF-8 bytes by default,
which is safely above the cleaning pipeline's 16,000-character chunk limit. Use
`--max-sentence-length` only if that chunking policy changes.

Finally, encode the exact same frozen corpus using that tokenizer:

```bash
uv run python dataset/build_dataset.py \
  --documents-dir dataset/documents \
  --tokenizer src/tokenizer/next/tokenizer.model \
  --overwrite
```

At this final stage, training, validation, and test token quotas are exact.
Changing the tokenizer therefore requires rebuilding the binary token
artifacts, but no redownload, recleaning, or resplitting of source documents.
The old direct stream-and-tokenize command remains useful when corpus reuse is
not needed.

## Production and local builds

Production defaults:

```bash
uv run python dataset/build_dataset.py --overwrite
```

The validation and test sizes are parameters rather than fixed assumptions.
For example, a 1B-token build with 25M-token evaluation sets is:

```bash
uv run python dataset/build_dataset.py \
  --validation-tokens 25000000 \
  --test-tokens 25000000 \
  --overwrite
```

Before the production run, rehearse the complete pipeline locally with 10M
tokens per training mixture and 1M-token evaluation sets:

```bash
uv run python dataset/build_dataset.py \
  --train-tokens 10000000 \
  --validation-tokens 1000000 \
  --test-tokens 1000000 \
  --output-dir dataset/ds_local_10m \
  --overwrite
```

Build only one mixture during debugging with `--mix web`, `--mix balanced`, or
`--mix knowledge`. The shared validation and test artifacts are still built.

## Inspecting FineWeb2 locally

Create a deterministic 10,000-document FineWeb2 sample without running the
production build:

```bash
uv run python dataset/build_dataset.py --sample-only --overwrite
uv run python dataset/explore_dataset.py --overwrite
```

This creates `dataset/fineweb2_sample.parquet` and a profiled copy under
`dataset/exploration/fineweb2/`. The source Parquet file is not modified. The
profile adds document length, URL domain, digit/symbol/uppercase ratios, and a
repeated-line ratio, then writes summary tables and plots.

To inspect records interactively:

```bash
uv run python
```

```python
import duckdb

db = duckdb.connect()
db.from_parquet(
    "dataset/exploration/fineweb2/profiled_documents.parquet"
).create_view("documents")

db.sql("""
    SELECT id, url, word_count, language_score, left(text, 300) AS preview
    FROM documents
    WHERE url_domain LIKE '%comune.vicenza.it'
    ORDER BY word_count DESC
""").show()
```

Print a complete selected record by its `id`:

```python
document_id = "paste-id-here"
row = db.execute(
    "SELECT * FROM documents WHERE id = ?", [document_id]
).fetchdf().iloc[0]
print(row["url"])
print(row["text"])
```

Useful review groups include documents below 100 words, the longest 1%, lower
language-confidence records, high repeated-line ratios, and large upstream
MinHash clusters. These measurements are inspection signals, not automatic
rejection rules.

## Training and evaluation

The existing training interface resolves all three mixtures against the same
`validation` artifact:

```bash
DATASET_VARIANT=web uv run python src/pavullmo/pretrain_base.py
DATASET_VARIANT=balanced uv run python src/pavullmo/pretrain_base.py
DATASET_VARIANT=knowledge uv run python src/pavullmo/pretrain_base.py
```

The evaluation artifacts are written in source order. For final comparisons,
evaluate the complete validation artifact; a short prefix-only validation run
would disproportionately measure its web region. `test` must not be used
for hyperparameter selection. The training script does not consume it; a
separate final-test command remains a subsequent pipeline task.

## Background

- [FineWeb2 technical report](https://arxiv.org/abs/2506.20920)
- [FineWeb dataset curation and ablations](https://arxiv.org/abs/2406.17557)
- [Italian Public Domain Books dataset](https://huggingface.co/datasets/PleIAs/Italian-PD)
