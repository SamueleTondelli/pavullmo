`src/` contains model, training, and execution code, `src/dataset/` builds datasets, `artifacts/` stores local artifacts, and `notes/` holds project documentation.

# PavuLLMo

## Setup
After setting up the uv project/venv and the modal cli, create the local tokenized dataset with

```bash
  python src/dataset/build_dataset.py
```

3 diffferent dataset sizes will be created: 10M, 100M and 1B tokens.

Then create the modal volume with

```bash
  python src/dataset/create_modal_volume.py
```

To start a pre-train run on modal

```bash
  EXPERIMENT_NAME=<experiment_name> DATASET_VARIANT=<dataset_size> modal run src/pavullmo/modal_pretrain_base.py
```

All hyperparameters can be configured as enviroment variables. BATCH_SIZE=32 seems to be the maximum batch size before the memory explodes.

## Checkpointing and resuming

Both `pretrain_base.py` and `pretrain_base_muon.py` save their final checkpoint
under `MODEL_OUTPUT_DIR/<EXPERIMENT_NAME>.pt`. Set `SAVE_TRAINING_STATE=true`
to include the last applied gradients, all optimizer and scheduler states,
Python/PyTorch/CUDA random-number states, shuffled data position, token count,
and clipping count. The default `false` keeps the smaller model-only checkpoint;
existing model-only checkpoints remain usable for evaluation but cannot resume.
Checkpoints are written atomically after successful training, at a completed
optimizer step, with no partially accumulated update.

Set `RESUME_CHECKPOINT` to a full checkpoint to start a new training phase from
its weights, gradients, and optimizer history. The new phase uses the current
environment settings, so LR, warmup, schedule/decay, weight decay, Adam betas,
Muon settings, batch size, gradient accumulation, dropout, context length,
seed, and dataset may change. For example, with the original architecture settings:

```bash
SAVE_TRAINING_STATE=true EPOCHS=1 LR=0.0003 WARMUP_STEPS=100 \
  WEIGHT_DECAY=0.01 DATASET_VARIANT=100m EXPERIMENT_NAME=continued_run \
  RESUME_CHECKPOINT=artifacts/models/original_run.pt \
  uv run python src/pavullmo/pretrain_base.py
```

`EPOCHS` and `MAX_STEPS` describe only the new phase, with run counters starting
at zero. Training starts at the beginning of the selected dataset, shuffled
using the current `SEED`. A fresh cosine/WSD schedule uses the new duration and
warmup settings. Saved scheduler counters, dataset positions, and RNG states
are retained in the checkpoint but are not restored for the new phase.

Adam moment estimates and optimizer step counters, and Muon momentum buffers,
are preserved. Current optimizer group options replace the saved options,
including LR, norm/Muon LR multipliers, weight decay, betas, and epsilon.
Last-step gradients are restored, then cleared before the first new accumulation
window because their update was already applied in the source run.

The model's parameter structure and optimizer types/parameter assignments must
match the checkpoint: vocabulary size, block count, embedding/FFN dimensions,
attention heads, QK normalization, QKV layout, and Canon layers cannot change.
The new dataset must use a compatible tokenizer and the same vocabulary size.

Both Modal wrappers forward these variables. On Modal, use a checkpoint path
in the mounted output Volume, such as `/outputs/models/original_run.pt`, and
give the continuation a unique `EXPERIMENT_NAME`. Both variables are recorded
in the checkpoint and run CSV configuration.

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
artifacts, not disposable copies. The dataset format is unchanged, and full
checkpoints retain the model and configuration fields used by evaluation. `notes/` may retain
historical paths and published experiment results.
