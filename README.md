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
