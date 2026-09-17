`dataset/` builds datasets, `src/` contains model, training, and execution code, `tmp/` stores local artifacts, and `notes/` holds project documentation.

# PavuLLMo

## Setup
The production dataset command builds three controlled 1B-token Italian source
mixtures, a shared parameterized 10M-token validation artifact, and a separate
10M-token final test artifact:

```bash
uv run python dataset/build_dataset.py --overwrite
```

Use smaller counts for the local 10M-token rehearsal:

```bash
uv run python dataset/build_dataset.py \
  --train-tokens 10000000 \
  --validation-tokens 1000000 \
  --test-tokens 1000000 \
  --output-dir dataset/ds_local_10m \
  --overwrite
```

See [`notes/dataset.md`](notes/dataset.md) for the source and mixture rationale,
cleaning rules, local exploration tutorial, configurable validation/test sizes,
the tokenizer-independent document download, limitations, and training
commands. The earlier single-source and Parquet
sampling modes remain available through their explicit `--source` and
`--sample-only` flags.

Then create the modal volume with

```bash
  python dataset/create_modal_volume.py
```

To start a pre-train run on modal

```bash
  EXPERIMENT_NAME=<experiment_name> DATASET_VARIANT=<dataset_size> modal run src/pavullmo/modal_pretrain_base.py
```

All hyperparameters can be configured as enviroment variables. BATCH_SIZE=32 seems to be the maximum batch size before the memory explodes.

## Artifact interface

Dataset builders write token shards and `metadata.json` to `tmp/datasets/`;
training reads those files through `DATASET_DIR`, without importing builders.
Keep tokenizer models and their metadata together in `tmp/tokenizers/` and pass
`--tokenizer` explicitly when selecting a different tokenizer. Tokenizer-building
scripts live in `dataset/tokenizer/`.

Local defaults are `tmp/documents/` for cleaned documents (where supported),
`tmp/models/` for checkpoints, `tmp/runs/` for TensorBoard, `tmp/results/` for
run registries, and `tmp/cache/huggingface/` for downloads. Existing command-line
and environment overrides remain supported. Modal continues to use `/datasets`
and `/outputs`; the local directory layout does not change remote volumes.

`tmp/` is ignored by Git. Preserve it when changing branches: it holds local
artifacts, not disposable copies. Existing generated files can be moved from
`dataset/ds/`, `dataset/documents/`, `models/`, and `runs/` to the corresponding
paths above. Dataset and checkpoint formats are unchanged. `notes/` may retain
historical paths and published experiment results.
