`src/` contains model, training, and execution code, `src/dataset/` builds datasets, `artifacts/` stores local artifacts, and `notes/` holds project documentation.

# PavuLLMo

## Dataset preparation

Prepare text before training a tokenizer or encoding training data:

```bash
uv sync --extra cloud --extra curation
uv run --extra curation python src/dataset/clean_dataset.py --documents-only
uv run python src/dataset/tokenizer/train_tokenizer.py \
  --documents-dir artifacts/documents --mixture balanced \
  --vocab-size 16000 --byte-budget 1073741824 --model-prefix artifacts/tokenizers
uv run python src/dataset/build_dataset.py --documents-dir artifacts/documents
```

The preparation command freezes, filters and globally exact-deduplicates all
sources before splitting. Its default 100,000-candidate limit per source may
need increasing with `--max-source-documents` to fill the requested budgets.
Use `--raw-documents-dir PATH` to reuse frozen raw inputs.

For a small experiment using only the prepared training partition:

```bash
uv run python src/dataset/clean_dataset.py split-experiment \
  --documents-dir artifacts/documents \
  --output-dir artifacts/experiments/small_run \
  --seed 731 --max-documents 1000 --holdout-permille 100
```

Pass that output as `--documents-dir` to tokenizer/token builders and choose
smaller byte/token budgets. Tiny splits may not fill every source quota.
The default production token outputs are three 1B-token mixtures and separate
10M-token validation and test artifacts. See the concise
[dataset guide](notes/dataset.md) for the objective, processing steps, outputs
and interfaces. Explicit `--sample-only` and `--source` modes remain legacy
inspection paths; they do not run the integrated cleaning pipeline.

Then create the modal volume with

```bash
  python src/dataset/create_modal_volume.py
```

To start a pre-train run on modal

```bash
  EXPERIMENT_NAME=<experiment_name> DATASET_VARIANT=<dataset_size> modal run src/pavullmo/modal_pretrain_base.py
```

All hyperparameters can be configured as environment variables. BATCH_SIZE=32 seems to be the maximum batch size before the memory explodes.

## Artifact interface

Dataset builders write token shards and `metadata.json` to `artifacts/datasets/`;
training reads those files through `DATASET_DIR`, without importing builders.
Keep tokenizer models and their metadata together in `artifacts/tokenizers/` and pass
`--tokenizer` explicitly when selecting a different tokenizer. Tokenizer-building
scripts live in `src/dataset/tokenizer/`.

Local defaults are `artifacts/documents/` for cleaned documents (where supported),
`artifacts/models/` for checkpoints, `artifacts/runs/` for TensorBoard, `artifacts/results/` for
run registries, and `artifacts/cache/huggingface/` for downloads. Existing command-line
and environment overrides remain supported. Modal continues to use `/datasets`
and `/outputs`; the local directory layout does not change remote volumes.

`artifacts/` is ignored by Git. Preserve it when changing branches: it holds local
artifacts, not disposable copies. Token shard and checkpoint formats are unchanged. `notes/` may retain
historical paths and published experiment results.
