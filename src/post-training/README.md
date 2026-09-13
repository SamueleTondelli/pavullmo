# PavuLLMo supervised fine-tuning prototype

This directory contains a minimal end-to-end SFT pipeline. The included ten
examples only verify the mechanics; they are far too few to produce a useful
assistant or meaningful validation/test estimates.

## Prepare the Italian Open SFT Chat dataset

The source-specific adapter under
`dataset_sources/italian_open_sft_chat/` downloads the Hugging Face dataset and
defaults to its `short` single-turn examples from the `magpie` and `meta`
sources. It preserves the publisher's train/validation/test assignments and
uses PavuLLMo's tokenizer to reject any fully formatted record over 1,024
tokens.

```bash
uv run python src/post-training/dataset_sources/italian_open_sft_chat/prepare.py

uv run python src/post-training/build_post_dataset.py \
  --split-source-dir src/post-training/data/italian_open_sft_chat
```

The downloaded normalized JSONL and generated PyTorch artifacts are ignored by
Git. The adapter code, filtering policy, and documentation are tracked.

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
For a source that already supplies splits, pass `--split-source-dir`; the
builder validates disjoint IDs and preserves those splits instead of reshuffling
them.

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

Each run also writes `metrics.csv` and `loss_curves.png` beside its TensorBoard
events. The horizontal axis is optimizer step. An epoch means one pass through
all training examples; with gradient accumulation, several batches contribute
to one optimizer step. Fractional epoch progress is logged to TensorBoard as
`progress/epoch`.

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

For a run that should survive closing the terminal, put Modal's detach option
before the script path:

```bash
EXPERIMENT_NAME=italian_open_sft_short_v1 \
EPOCHS=1 BATCH_SIZE=16 GRAD_ACCUM_STEPS=2 LR=1e-5 \
WARMUP_STEPS=20 VALIDATION_INTERVAL=100 COMPILE_MODEL=false \
uv run --extra cloud modal run --detach src/post-training/modal_post_train_sft.py
```

The launch output prints the Modal App page and a temporary live TensorBoard
URL. The App page shows status, logs, GPU/CPU/RAM metrics, and function-call
history. TensorBoard shows `train/loss`, `validation/loss`, learning rate,
gradient norm, and epoch progress while the container is alive. After the run,
the output Volume retains the checkpoint, TensorBoard events, metrics CSV, and
static loss plot even though the temporary TensorBoard URL has expired.

## Try the post-trained model

The generic autoregressive loop and checkpoint loader remain in
`src/pavullmo/generate.py`. The post-training `chat.py` entry point reuses them
but formats each request exactly like the SFT data: BOS, optional `<system>`,
`<user>`, then `<assistant>`. This distinction matters; sending an unformatted
plain prompt does not activate the chat behavior learned during SFT.

Test the checkpoint directly from the Modal output Volume:

```bash
uv run --extra cloud modal run src/post-training/modal_chat.py \
  --prompt "Spiegami in tre frasi perché il cielo appare blu."
```

Choose another stored model with `SFT_CHECKPOINT_NAME=filename.pt`. Sampling
controls are exposed as `--max-new-tokens`, `--temperature`, `--top-k`, and
`--seed`; `--temperature 0` performs deterministic greedy decoding.

For local interactive use, first download the checkpoint:

```bash
uv run --extra cloud modal volume get pavullmo-post-training-outputs \
  models/italian_open_sft_short_v1_20260913.pt \
  post-training-models/italian_open_sft_short_v1_20260913.pt

uv run python src/post-training/chat.py \
  post-training-models/italian_open_sft_short_v1_20260913.pt
```
