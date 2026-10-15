# pavullmo

This project pretrains small decoder-only language models for Italian. It is a
Python 3.12 `uv` project with tokenizer/dataset builders, a configurable model,
and one shared single-GPU PretrainingTrainer used by local and Modal launchers.

## Data and tokenizer

- `src/dataset/` owns the existing dataset and tokenizer builders on main.
  `build_dataset.py` streams `gsarti/clean_mc4_it`, using its `tiny` split;
  the tokenizer scripts live under `src/dataset/tokenizer/` and retain their
  vocabulary-size, byte-budget and model-prefix CLI inputs.
- Tokenizer IDs are UNK=0, BOS=1, EOS=2 and PAD=3, with `<system>`, `<user>`
  and `<assistant>` symbols. Preserve the existing tokenizer implementation.
- The dataset builder writes deterministic 10M-, 100M- and 1B-token training
  prefixes and complete validation, encoding documents as `[BOS, *content, EOS]`.
- Generated artifacts stay in `artifacts/datasets/`, with tokenizers under
  `artifacts/tokenizers/`. Tokens are sharded little-endian uint16 streams plus
  `metadata.json` with counts, hashes, tokenizer identity and source details.
- This integration does not import the scaling branch's FineWeb/production
  pipelines or its dataset/tokenizer changes. The shared reader supports fixed
  evaluation offsets when metadata supplies them; main's builder is unchanged.

## Model and shared components

- `src/model/model.py` implements a pre-normalized decoder-only Transformer
  with RMSNorm, SwiGLU, RoPE and causal native scaled-dot-product attention.
  The model and architectural layers take `ModelConfig`. Context length is a
  hard limit; embedding dimension must divide evenly into attention heads and
  each head dimension must be even.
- `model.split_qkv_projections` selects separate query/key/value parameters
  while retaining a concatenated attention projection in the forward pass.
  Its default is false. Inputs are integer IDs shaped `[batch, sequence]` and
  outputs are logits shaped `[batch, sequence, vocab_size]`.
- All tuned model presets explicitly enable `model.split_qkv_projections`.
- `src/components/` owns `GenericTrainer`, `PretrainingTrainer`,
  `CheckpointManager`, the optimizer wrapper,
  token dataset/loaders and shared diagnostic grouping. Launchers do not
  duplicate training algorithms or depend on each other's implementation.
- `GenericTrainer` in `generic_trainer.py` initializes shared model/optimizer,
  compilation, logging/writer and checkpoint-manager components. Its constructor
  takes a stage's `TrainerConfig` subclass plus an explicit seed and computed
  `total_steps`; it preserves the concrete config type. It owns no training
  loop, datasets, accumulation, stepping, evaluation or checkpoint state.
- `GenericTrainer.forward_backward(microbatch, loss_fn, loss_kwargs=...)` performs
  one forward and backward and returns a detached scalar loss. A tensor is
  passed positionally; a mapping is expanded as model kwargs. The caller moves
  inputs/targets to the device. The specialized loss supplies all normalization
  and accumulation scaling. The helper marks compiled forward steps but never
  clears/clips gradients or steps optimizers/schedules.
- `PretrainingTrainer` takes only `PretrainingConfig`. It constructs components, validates matching
  train/validation tokenizer hashes and vocabulary size, seeds Python/PyTorch/
  CUDA, and computes the optimizer-step horizon from data length and run duration.
- Startup logs dump each top-level config section as single-line TOML:
  `GenericTrainer` owns the fields defined by `TrainerConfig`, while
  `PretrainingTrainer` logs only its additional sections and retained extras.
  Derive these groups from the config schema rather than forwarding lists.
- Config fields are user-selected settings. Do not add computed state such as
  scheduler total steps to config classes, or rewrite config values during setup.
  Pass computed values directly to component constructors, for example
  `Optimizer(config.optimizer, model, total_steps=total_steps)`.
- The memory-mapped reader returns shifted input/target spans of context length.
  Blocks share their boundary token, support cross-shard reads, and shuffle with
  a deterministic epoch sampler. Prefetch does not advance the consumed cursor.
  Training drops incomplete batches/accumulation groups. Validation is ordered,
  retains partial final batches and checks complete fixed-sample coverage.
- `optimizer.kind` selects AdamW or Muon for hidden matrices plus AdamW for
  other parameters. Norms, biases and tied embeddings are excluded from decay.
  Schedule settings are nested under `optimizer.schedule`, with warmup and
  cosine or WSD decay. The wrapper exposes separate optimizer/scheduler steps.

## Configuration and local training

- `src/config/` defines typed Pydantic config classes and generic TOML/dictionary
  serialization. Add hyperparameters and defaults to `schema.py`; do not maintain
  extra serialization or provider forwarding lists. All training settings come
  from `PretrainingConfig`. Config classes do not inspect argument parsers.
- `TrainerConfig` contains common name/model/optimizer/runtime/logging/checkpoint
  sections; `PretrainingConfig` adds train_dataset/validation_dataset/training/
  evaluation. Both dataset sections are independent `TokenDatasetConfig`
  instances with an explicit artifact `path`. New training stages
  define their own typed root subclasses. Existing TOML config/checkpoint
  contents remain compatible with the Python class rename.
- `PretrainingConfig.name` is mandatory, has no default and must be filename-safe. Use a
  unique name for each experiment to avoid replacing checkpoints/mixing events.
- Unknown config fields/sections are retained and produce Loguru warnings;
  missing optional settings receive defaults. Explicit override paths reject
  typos. Preserve this behavior so configs remain shareable across branches.
- Presets under `src/pavullmo/configs/` are partial TOML files. Authored `extends`
  lists compose recursively, relative to the declaring file. CLI `--extends`
  appends semicolon-separated presets after existing includes. The main file's
  own values win over includes; typed CLI `--override` assignments win last.
  Composition happens before defaults/validation. Saved configs are standalone
  TOML without include directives. Source files are never rewritten.
- Both launchers accept file paths or bundled preset names, optionally omitting
  `.toml`. Extra CLI includes search beside the main file, then the working
  directory, then bundled presets. Local flags can be repeated. Quote whole
  semicolon-separated arguments so the shell passes them intact.
- `src/pavullmo/pretrain.py` is the local entry point. For example, from the root:

  ```bash
  uv run python src/pavullmo/pretrain.py --config 10m_tuned_config.toml \
    --override 'name=trial;train_dataset.path=train_16k_195m;validation_dataset.path=validation_16k;optimizer.kind=muon_adamw'
  ```

- `train_dataset.path` and `validation_dataset.path` are explicit artifact paths, relative to
  `runtime.dataset_dir` or absolute. There is no prefix/variant path inference.
  Legacy `[data] train=.../validation=...` configs migrate with a warning;
  explicit new dataset fields take precedence and unknown legacy settings are
  retained with warnings. Saved configs use the new dataset sections only
  (plus any retained unknown settings).
  Dataset/output/CSV paths come exclusively from the config, resolve relative
  to the process working directory, and are never replaced by launchers.
- Local defaults are `artifacts/datasets`, `artifacts/runs` and
  `artifacts/results/pretrain_runs.csv`. Each run directory contains `config.toml`,
  `tensorboard/` and `checkpoints/step_XXXXXXXX.pt`. Loguru's optional file sink
  follows `logging.file`, independently of the output root.
- Training uses BF16 autocast with FP32 parameters/optimizer state, optional
  accumulation and z-loss, finite-loss/gradient checks and gradient clipping.
  Compilation wraps the training forward model; validation uses the base model
  eagerly and computes token-weighted cross-entropy over the configured batches.
- Completed steps log loss/objective/z-loss, pre-clipping gradient norm, clipping
  frequency, learning rates and throughput to TensorBoard and Loguru. Sparse
  diagnostics include parameter/gradient-group norms. Split QKV retains the
  combined `gradient_groups/attention_qkv` metric and adds query/key/value norms.
- Successful runs append final metrics, checkpoint path and complete single-line TOML config
  to `runtime.runs_csv`. The `dataset_variant` column contains the explicit
  training artifact path. Use one writer per CSV. Start local TensorBoard with
  `.venv/bin/tensorboard --logdir artifacts/runs --port 6006`.
- Every config inherits `to_string()`/`from_string()` for the CSV representation:
  `config = { ... }` uses nested TOML inline tables and escapes string line
  breaks. `from_string()` also accepts older multiline TOML config values.
  Regular config files and checkpoint `config_toml` still use `to_toml()`.
  CSV appends recognize matching older multiline rows without rewriting them.

## Checkpointing and resume

- PretrainingTrainer owns model/optimizer/scheduler state, progress, consumed data cursor,
  losses and Python/PyTorch/CUDA/loader RNG streams. Checkpoints contain plain
  state dictionaries and TOML configuration, not config/model Python instances
  or compiled wrappers. The computed horizon belongs to checkpoint progress.
- Save only after a completed optimizer update AND scheduler step, with no pending
  accumulation; step zero is also valid. PretrainingTrainer enforces this boundary. Writes
  use a same-directory temporary file, fsync and atomic replacement; retention
  prunes only managed files after successful publication.
- `checkpoint.interval_steps` controls periodic saves (zero disables them).
  Completion and graceful stop still save. `checkpoint.keep_last` controls
  retention. `checkpoint.save_training_state=false` exports weights/config only
  and cannot provide a full resume.
- `--resume <checkpoint>` reads the embedded config and restores the trajectory.
  Model/optimizer/schedule/training/evaluation settings, data identities, horizon,
  parameter assignments and device type/PyTorch version must match. Runtime
  paths, logging, retention and the label may change through explicit overrides.
  `--config` and `--resume` are mutually exclusive; `--extends` applies only to
  `--config`. Resume does not require the original preset files.
- `request_stop()`, or SIGINT/SIGTERM in the main thread, finishes the current
  complete update, saves and returns. Failed/incomplete updates cannot be saved
  or continued until a checkpoint is restored. Flush/close writers through the
  PretrainingTrainer context manager. Keep RNG restoration last when changing resume code.

## Modal data and training

- `src/dataset/create_modal_volume.py` uploads validated artifacts to a Modal
  dataset Volume; existing files require `--force` to overwrite. Run it with
  `uv run --extra cloud python src/dataset/create_modal_volume.py`. Ensure the
  selected Volume contains the artifact paths configured for training.
- `src/pavullmo/pretrain_modal.py` uses the same PretrainingTrainer and resolved TOML as local
  training. Its image uses frozen `uv.lock`, defaults to an L4, and mounts the
  dataset Volume read-only at `/datasets` and output Volume at `/outputs`.
- Explicitly include `modal_config.toml` to select `/datasets`, `/outputs/runs`
  and `/outputs/pretrain_runs.csv`. The wrapper does not inject paths. Example:

  ```bash
  uv run --extra cloud modal run src/pavullmo/pretrain_modal.py \
    --config 10m_tuned_config.toml --extends modal_config \
    --override 'name=cloud_trial;train_dataset.path=train_16k_195m;validation_dataset.path=validation_16k;optimizer.kind=muon_adamw'
  ```

- Infrastructure flags are `--gpu`, `--cpu`, `--memory-mb`, `--timeout-seconds`,
  `--dataset-volume`, `--output-volume` and `--no-tensorboard`; they stay outside
  `PretrainingConfig`. Use `modal run --detach ...` for terminal-independent runs.
- Modal reads `--resume` exactly as supplied in the remote container; prefer an
  absolute checkpoint path under `/outputs`. Relative paths use its working
  directory. The saved config must point to accessible remote inputs/outputs.
- TensorBoard watches the configured output root through a public tunnel for the
  run's lifetime. The output Volume is committed after every published checkpoint
  and on function exit, including after failures. Persistence uses the small
  runtime `CheckpointManager.after_save` callback, not provider code in PretrainingTrainer.
- Use unique run names and sequential launches against a shared output Volume.
  Parallel sweeps need separate output Volumes/registries and a coordinator merge
  to avoid losing updates from concurrent CSV writers or Volume commits.
- Keep local image construction inside `if modal.is_local()` because Modal also
  imports the wrapper under `/root/pretrain_modal.py`. Apply `.add_local_dir(...)`
  last in the image chain unless using `copy=True`.

## Environment and verification

- Use Python 3.12 and manage dependencies/lockfile with `uv`. Keep `pyproject.toml`
  and `uv.lock` synchronized and preserve PyTorch 2.9.1 with CUDA 12.8 wheels,
  `datasets<4`, SentencePiece, TensorBoard 2.21 and the pinned causal-conv1d wheel.
  The optional `cloud` extra provides Modal. Config dependencies are Pydantic,
  tomlkit and Loguru.
- PretrainingTrainer requires CUDA with `torch.cuda.is_bf16_supported()`. Full runs target
  a Modal L4 (24 GB); local smoke tests target an RTX 3070 (8 GB). Use BF16
  autocast, not BF16 model parameters. Use native SDPA/FlashAttention-2; do not
  add third-party flash-attn without a concrete unsupported API requirement.
- Run `pavullmo-check` after changes to PyTorch, CUDA dependencies or GPU checks.
  Run relevant tests with `.venv/bin/python -m unittest discover -s tests`.
  CUDA and multiprocessing-loader tests need GPU access and local worker sockets.

## Repository ownership

- `src/dataset/` owns dataset/tokenizer building; `src/config/` owns configuration;
  `src/model/` owns model architecture; `src/components/` owns shared training;
  `src/pavullmo/` owns local/cloud execution and generation entry points.
- `generate.py` takes a saved model directory and a tokenizer path. It loads the
  most recently published managed checkpoint under `checkpoints/`, or a single
  legacy `.pt` beside `config.toml`. `--checkpoint` selects an absolute path or
  a path relative to the supplied directory. Current checkpoints use their
  embedded `[model]` config; legacy checkpoints use the directory's `[model]`.
  Generation constructs only `ModelConfig` and the model, works on CPU/CUDA,
  and does not require a valid full PretrainingConfig or accessible training datasets.
- `migrate_checkpoint_config.py <legacy.pt>` writes a partial `[model]` TOML
  config beside the checkpoint after validating weight keys/shapes. It recovers
  missing layout flags from weights and leaves the `.pt` unchanged. Replacing
  existing config files requires `--force`. Migrated configs support generation
  only; they do not enable PretrainingTrainer resume. Multiple legacy checkpoints in one
  directory require explicit selection and a matching shared model config.
- `artifacts/` owns generated datasets, tokenizers, checkpoints, registries, logs
  and caches and is ignored by Git. `notes/` preserves documentation and historical
  experiment evidence. Preserve artifacts and historical evidence when cleaning up.
- Preserve the token-shard plus metadata dataset/training interface and existing
  dataset-building CLI controls when changing paths.

- This main integration includes only shared config/model/training components,
  their launchers, dependencies, docs and relevant tests. Do not import
  branch-specific evaluation/scaling programs or dataset/tokenizer pipelines.
