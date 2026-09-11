# PavuLLMo supervised fine-tuning prototype

This directory contains a minimal end-to-end SFT pipeline. The included ten
examples only verify the mechanics; they are far too few to produce a useful
assistant or meaningful validation/test estimates.

## Build the dataset

From the repository root:

```bash
uv run python src/post-training/build_post_dataset.py
```

This validates `data/sft_examples.jsonl`, deterministically assigns 8 examples
to training, 1 to validation, and 1 to test, formats role-tagged conversations,
and writes `train.pt`, `validation.pt`, `test.pt`, and `metadata.json` under the
ignored `src/post-training/dataset/` directory. Each artifact stores token IDs
and a boolean loss mask. System/user tokens provide context, while the
`<assistant>` marker, assistant answer, and EOS token are training targets.

## Validate the supplied checkpoint and data without training

The `.pt.zip` filename is already a valid PyTorch checkpoint; do not unzip it.

```bash
BASE_CHECKPOINT=/home/leonardo/Desktop/35m_tok16k_tuned_4b.pt.zip \
EVAL_ONLY=true \
uv run python src/post-training/post_train_sft.py
```

Add `EVALUATE_TEST=true` only for a final held-out evaluation. Test evaluation
is deliberately disabled during ordinary development runs.

## Run SFT

```bash
BASE_CHECKPOINT=/home/leonardo/Desktop/35m_tok16k_tuned_4b.pt.zip \
EXPERIMENT_NAME=35m_tok16k_sft_smoke \
LR=1e-5 \
BATCH_SIZE=2 \
EPOCHS=3 \
uv run python src/post-training/post_train_sft.py
```

`DEVICE=auto` uses CUDA when available and otherwise falls back to CPU. CUDA
uses BF16 autocast; model parameters and AdamW state remain FP32. The script
performs full-parameter fine-tuning, computes assistant-only cross-entropy,
reports validation loss during training, optionally evaluates test loss,
writes TensorBoard events under `post-training-runs/`, saves the checkpoint
under `post-training-models/`, and appends a row to
`src/post-training/post_training_runs.csv`.

Set `EVALUATE_TEST=true` for the final test evaluation. With a real dataset, do
not repeatedly inspect or tune against the test split.

## Upload SFT inputs to Modal

The uploader validates all artifacts and their compatibility with the base
checkpoint before writing them to `pavullmo-post-training-inputs`:

```bash
uv run --extra cloud python src/post-training/create_modal_volume.py
```

The Volume layout is:

```text
/checkpoints/35m_tok16k_tuned_4b.pt.zip
/dataset/metadata.json
/dataset/train.pt
/dataset/validation.pt
/dataset/test.pt
```

After rebuilding a real dataset, replace only the remote dataset directory:

```bash
uv run --extra cloud python src/post-training/create_modal_volume.py \
  --component dataset --force
```

The uploader still reads the local checkpoint metadata in dataset-only mode so
it can reject an incompatible tokenizer vocabulary or context length before
uploading.

## Run SFT on Modal

The wrapper mounts the input Volume read-only and stores checkpoints, events,
and `post_training_runs.csv` in `pavullmo-post-training-outputs`:

```bash
EXPERIMENT_NAME=35m_tok16k_sft \
uv run --extra cloud modal run src/post-training/modal_post_train_sft.py
```

For a one-step pipeline smoke test:

```bash
EXPERIMENT_NAME=35m_tok16k_sft_smoke \
EPOCHS=1 MAX_STEPS=1 WARMUP_STEPS=0 BATCH_SIZE=2 \
VALIDATION_INTERVAL=1 COMPILE_MODEL=false EVALUATE_TEST=true \
uv run --extra cloud modal run src/post-training/modal_post_train_sft.py
```

Infrastructure can be changed with `MODAL_POST_INPUT_VOLUME_NAME`,
`MODAL_POST_OUTPUT_VOLUME_NAME`, `BASE_CHECKPOINT_NAME`, `MODAL_GPU`,
`MODAL_CPU`, `MODAL_MEMORY_MB`, and `MODAL_TIMEOUT_SECONDS`.
