`dataset/` builds datasets, `src/` contains model, training, and execution code, `tmp/` stores local artifacts, and `notes/` holds project documentation.

# PavuLLMo

## Setup
After setting up the uv project/venv and the modal cli, create the local tokenized dataset with

```bash
  python dataset/build_dataset.py
```

3 diffferent dataset sizes will be created: 10M, 100M and 1B tokens.

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
