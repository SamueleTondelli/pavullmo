# pavullmo

This project aims to pretrain a small (~50M parameter) decoder-only language
model for Italian. It is an early-stage `uv` project: the tokenizer, model,
pre-tokenized datasets, and a local single-GPU pretraining loop are implemented.

## Data and tokenizer

- The corpus is `gsarti/clean_mc4_it`, loaded from Hugging Face Datasets in
  streaming mode. It provides `tiny`, `small`, `medium`, `large`, and `full`
  variants; current tokenizer work uses `tiny`.
- `src/tokenizer/train_tokenizer.py` trains a 16,000-token SentencePiece BPE on
  up to 1,000,000 documents, with full character coverage and byte fallback.
- Token IDs are fixed as UNK=0, BOS=1, EOS=2, and PAD=3. The vocabulary also
  includes `<system>`, `<user>`, and `<assistant>`.
- `src/tokenizer/count_tokens.py` samples the streamed corpus and extrapolates
  token counts for each dataset variant from the dataset card's approximate
  word counts.
- `dataset/build_dataset.py` streams the `tiny` configuration once and creates
  deterministic 10M-, 100M-, and 1B-token training prefixes plus the complete
  validation split. Each nonempty document is encoded as
  `[BOS, *content_tokens, EOS]`.
- Generated artifacts live under `dataset/ds/` and are intentionally ignored by
  Git. Tokens are flat, little-endian `uint16` streams in shards of at most 50M
  tokens, with counts, hashes, tokenizer information, and source details in each
  artifact's `metadata.json`.

## Model

- `src/model/model.py` implements a pre-normalized decoder-only Transformer
  with RMSNorm, SwiGLU feed-forward layers, and causal scaled-dot-product
  attention.
- Rotary positional embeddings (RoPE) are applied to queries and keys. The
  configured context length is a hard limit, and each attention head must have
  an even dimension.
- Model inputs are integer token IDs with shape `[batch, sequence]`; outputs are
  vocabulary logits with shape `[batch, sequence, vocab_size]`.

## Pretraining

- `src/pavullmo/pretrain_base.py` is the local single-GPU next-token pretraining
  entry point. It reads `dataset/ds/train_<variant>` and
  `dataset/ds/validation` directly; it does not tokenize text during training.
- The memory-mapped dataset loader constructs `sequence_length + 1` token spans
  and returns shifted input/target tensors of shape `[sequence_length]`. Blocks
  use a stride of `sequence_length`, so adjacent blocks share their boundary
  token and no next-token transition is lost. Reads across shard boundaries are
  supported.
- Training uses fused AdamW, BF16 autocast, optional gradient accumulation,
  gradient clipping, linear learning-rate warmup, and cosine decay to `MIN_LR`.
  Model compilation is enabled by default with static shapes.
- Train loss, pre-clipping gradient norm, and learning rate are logged after
  every optimizer step. Validation loss is computed periodically over a fixed
  number of validation batches. All metrics and the run configuration are
  written to TensorBoard under `runs/<experiment_name>` by default.
- After successful training, the final model checkpoint is written under
  `MODEL_OUTPUT_DIR`. It contains the model state, final losses, global step,
  dataset variant, hyperparameter mapping, and exact hyperparameter string.
- The final train/validation losses, script name, dataset variant, experiment
  name, model path, and complete hyperparameter string are appended to
  `src/pavullmo/pretrain_runs.csv`. Override the registry path with `RUNS_CSV`.
- Every hyperparameter must be read from an environment variable; do not add a
  hard-coded model, optimizer, data, schedule, evaluation, logging, or
  compilation hyperparameter. Whenever a hyperparameter is added or changed,
  also include it in the script's `hyperparameters` mapping so its exact
  `NAME=value` setting is recorded in the CSV `config` column for every
  successful run.
- Important runtime controls include `DATASET_VARIANT`, `BATCH_SIZE`,
  `GRAD_ACCUM_STEPS`, `EPOCHS`, `MAX_STEPS`, `VALIDATION_INTERVAL`,
  `VALIDATION_STEPS`, `NUM_WORKERS`, `LOG_DIR`, `COMPILE_MODEL`, and
  `COMPILE_MODE`.
- Start the dashboard with
  `.venv/bin/tensorboard --logdir runs --port 6006`. TensorBoard events flush
  every five seconds by default; configure this with
  `TENSORBOARD_FLUSH_SECS`.

## Modal data and pretraining

- `dataset/create_modal_volume.py` validates the four generated dataset
  artifacts and uploads them to the root of the `pavullmo-datasets` Modal
  Volume. Existing files are not overwritten unless `--force` is passed.
- Run the uploader locally with
  `uv run --extra cloud python dataset/create_modal_volume.py`. The training
  Volume root contains `train_10m`, `train_100m`, `train_1b`, and `validation`
  directly, matching the layout expected by `DATASET_DIR`.
- `src/pavullmo/modal_pretrain_base.py` wraps `pretrain_base.py` in a Modal App.
  Its image is built from the frozen `uv.lock`, defaults to one L4, mounts the
  dataset Volume read-only at `/datasets`, and creates or reuses the
  `pavullmo-training` output Volume at `/outputs`.
- Launch a run from the project root with
  `uv run --extra cloud modal run src/pavullmo/modal_pretrain_base.py`. Use the
  same hyperparameter environment variables as local pretraining and use
  Modal's `--detach` option when the job must outlive the local terminal.
- The wrapper forces `DATASET_DIR=/datasets`, `LOG_DIR=/outputs/runs`,
  `MODEL_OUTPUT_DIR=/outputs/models`, and
  `RUNS_CSV=/outputs/pretrain_runs.csv`. It forwards all other supported
  pretraining environment variables before importing `pretrain_base.py`.
- Each remote run starts TensorBoard in the training container and prints a
  live Modal tunnel URL. The URL is public, remains active only for that run,
  and sees event updates immediately. The output Volume is explicitly
  committed when the function exits, including after a failed run so partial
  diagnostic events are retained.
- Modal infrastructure overrides are `MODAL_DATASET_VOLUME_NAME`,
  `MODAL_OUTPUT_VOLUME_NAME`, `MODAL_GPU`, `MODAL_CPU`, `MODAL_MEMORY_MB`, and
  `MODAL_TIMEOUT_SECONDS`. These configure infrastructure and are distinct from
  recorded training hyperparameters.
- Give every run a unique, filename-safe `EXPERIMENT_NAME`. Reusing a name
  merges TensorBoard events and replaces `<experiment_name>.pt`, making older
  CSV rows point at the replacement checkpoint.
- Repeated sequential wrapper launches are suitable for a small sweep. Do not
  launch runs concurrently against the same output Volume: each run currently
  appends the same CSV and commits the same Volume, and concurrent writers can
  lose updates. A parallel sweep must write one result file and checkpoint
  directory per run, then have one coordinator merge the result files into the
  CSV after all workers finish.
- Keep local-only image construction inside `if modal.is_local()` because Modal
  imports the wrapper again as `/root/modal_pretrain_base.py`. Apply
  `.add_local_dir(...)` last in the image chain unless using `copy=True`.

## Environment

- Use Python 3.12 and manage dependencies and the lockfile with `uv`.
- Training dependencies are PyTorch 2.9.1 with its CUDA 12.8 wheels,
  `datasets<4`, SentencePiece, tqdm, and TensorBoard 2.21.
- The optional `cloud` extra provides the Modal client. Full training is
  expected to run on a Modal L4 (24 GB); local smoke tests target an RTX 3070
  (8 GB).
- The pretraining script requires CUDA and a GPU for which
  `torch.cuda.is_bf16_supported()` is true. Keep parameters and optimizer state
  in FP32 and use BF16 through autocast rather than converting the model itself.
- Use PyTorch's native scaled-dot-product attention. Its FlashAttention-2
  backend works on both target GPUs; do not add the third-party `flash-attn`
  package unless a concrete unsupported API is required.

Keep `pyproject.toml` and `uv.lock` synchronized, preserve the pinned GPU stack,
and run `pavullmo-check` after changing PyTorch, CUDA-related dependencies, or
GPU environment checks.
