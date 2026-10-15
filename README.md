`src/` contains model, training, and execution code, `src/dataset/` builds datasets, `artifacts/` stores local artifacts, and `notes/` holds project documentation.

The new typed TOML configuration layer lives in `src/config/`; see
[configuration usage and compatibility](notes/configuration.md). Shared training components live in `src/components/`; `pretrain.py` and
`pretrain_modal.py` are the local and Modal entry points.

`PretrainingTrainer` inherits common model/optimizer setup and the single-microbatch
`forward_backward()` helper from `GenericTrainer`. It owns the pretraining loss,
accumulation, optimizer/scheduler steps, evaluation and checkpoint state. Its root
config is `PretrainingConfig`, extending the common `TrainerConfig` sections.
Existing TOML presets and checkpoint configurations keep the same fields.

# PavuLLMo

## Setup
Install the locked dependencies with `uv sync` (`uv sync --extra cloud` for Modal).
On Linux x86_64, `causal-conv1d` uses the upstream prebuilt wheel for Python 3.12,
PyTorch 2.9, CUDA 12, and the C++11 ABI. This avoids an isolated source build
selecting a different PyTorch version and requires no local CUDA toolkit or GCC
configuration. If the pinned Python/PyTorch/CUDA stack changes, update the wheel
URL in `pyproject.toml` to match and regenerate `uv.lock`.

`CanonLayer` uses `causal_conv1d_fn` on CUDA for kernel sizes 2–4 and
FP32/FP16/BF16 inputs, with BF16 autocast and FP32 training parameters. Its residual
connection, weight shape, and initialization remain compatible with existing
checkpoints. CPU inputs and unsupported kernel sizes/dtypes use PyTorch Conv1d.
Compiled runs use sequence-contiguous kernel inputs to support PyTorch 2.9's
custom-op tracing requirements.

After setting up the uv project/venv and the modal cli, create the local tokenized dataset with

```bash
  python src/dataset/build_dataset.py
```

3 diffferent dataset sizes will be created: 10M, 100M and 1B tokens.

Then create the modal volume with

```bash
  python src/dataset/create_modal_volume.py
```

## Training

Save a run configuration such as `experiment.toml` in the project root:

```toml
extends = ["src/pavullmo/configs/10m_config.toml"]
name = "my_run"

[train_dataset]
path = "train_10m"

[validation_dataset]
path = "validation"

[optimizer]
kind = "muon_adamw"  # Or "adamw".
```

Choose artifacts with a tokenizer matching the model vocabulary. Presets under
`src/pavullmo/configs/` are partial TOML files; tuned presets extend their model
size preset. They preserve the former shell settings and contain no dataset
selectors or default run name. Optimizer selection defaults to AdamW unless set
in the run configuration or overridden.

Run locally or on Modal with the same configuration and override syntax:

```bash
uv run python src/pavullmo/pretrain.py --config experiment.toml \
  --override 'training.batch_size=8;optimizer.lr=0.002'

uv run --extra cloud modal run src/pavullmo/pretrain_modal.py --config experiment.toml \
  --extends modal_config --override 'training.batch_size=8;optimizer.lr=0.002'
```

Quote the whole `--override` argument to keep semicolons out of shell parsing.
Values use TOML types; simple strings and paths may be bare. Local `--override`
can also be repeated. Config classes receive only the parsed dictionary.

Both launchers accept `--extends 'first;second.toml'`, appending presets after the
config's existing includes. The main config's own settings win over includes;
CLI overrides win last. Paths or bundled preset names work with an optional
`.toml` suffix, so `--config 10m_tuned_config.toml --extends modal_config` selects
the tuned model and standard Modal paths. The run name remains required.

Modal accepts `--gpu`, `--cpu`, `--memory-mb`, `--timeout-seconds`,
`--dataset-volume`, `--output-volume`, and `--no-tensorboard`. Use Modal's
`--detach` before the script path to keep the run alive after the terminal exits.
Modal mounts datasets at `/datasets` and outputs at `/outputs`. Paths come only
from the run config: `--extends modal_config` loads `modal_config.toml`, which
sets the dataset root, output root and CSV for those mounts. Explicit values in
the main config or `--override` still take precedence. Put matching
artifacts on the dataset Volume and choose output/CSV paths under `/outputs`
for persistence. TensorBoard watches the configured output root through a public
tunnel for the duration of the run.

A small local sweep can stay a throwaway shell loop:

```bash
for lr in 0.001 0.002 0.003; do
  uv run python src/pavullmo/pretrain.py --config experiment.toml \
    --override "name=lr_${lr};optimizer.lr=$lr"
done
```

## Checkpointing and resuming

Full checkpoints are saved periodically, at completion, and on graceful stop,
after a completed optimizer and scheduler update. Local outputs live under
`artifacts/runs/<name>/`, including resolved `config.toml`, TensorBoard events and
`checkpoints/step_XXXXXXXX.pt`. Successful runs are registered in
`artifacts/results/pretrain_runs.csv`, including the complete configuration as
single-line TOML (`config = { ... }`). Every config provides `to_string()` and
`from_string()` for this form; `from_string()` also reads older multiline TOML
CSV values. Config files and checkpoint snapshots retain their regular TOML format.

Resume from the embedded configuration without the source preset files:

```bash
uv run python src/pavullmo/pretrain.py \
  --resume artifacts/runs/my_run/checkpoints/step_00001000.pt

uv run --extra cloud modal run src/pavullmo/pretrain_modal.py \
  --resume /outputs/runs/my_run/checkpoints/step_00001000.pt
```

Modal reads the resume path as supplied in the remote container; no mount prefix
is added. Absolute paths are recommended; relative paths use its working directory.
Resume restores optimizer/scheduler state, RNG streams and the consumed data
cursor. Runtime paths, logging, retention and the run name may be overridden;
model/training/optimizer/evaluation settings must match the original trajectory.
`--extends` applies only with `--config`; use `--override` for resume edits.
Ctrl+C/SIGTERM request a complete-update stop and save. Use one writer per CSV
and output Volume; parallel sweeps need separate output locations and a later
coordinator merge. See [configuration details](notes/configuration.md).

## Generation

Pass the saved run directory and its matching SentencePiece tokenizer:

```bash
uv run python src/pavullmo/generate.py artifacts/runs/my_run \
  artifacts/tokenizers/my_tokenizer/tokenizer.model
```

Generation selects the most recently saved managed checkpoint under
`checkpoints/` and loads its embedded `[model]` config, without constructing a
PretrainingTrainer or requiring the training datasets. Full and weights-only checkpoints
both work. Use `--checkpoint checkpoints/step_00001000.pt` to select a specific
save; absolute checkpoint paths also work. Generation supports CUDA and CPU.

For an older checkpoint, first write a model-only `config.toml` beside it:

```bash
uv run python src/pavullmo/migrate_checkpoint_config.py artifacts/models/old_run.pt
uv run python src/pavullmo/generate.py artifacts/models \
  artifacts/tokenizers/my_tokenizer/tokenizer.model --checkpoint old_run.pt
```

The helper converts embedded legacy hyperparameters, recovers missing parameter
layout flags from the weights, and validates model keys/shapes before writing.
It leaves the checkpoint unchanged. Existing configs require `--force` to replace.
These partial configs support generation, not training resume. A directory with
one legacy `.pt` needs no `--checkpoint`; multiple legacy files require selection.
Each selected legacy checkpoint must match the directory's `[model]` config.

## Artifact interface

Dataset builders write token shards and `metadata.json` to `artifacts/datasets/`;
training reads explicitly selected artifact paths under `runtime.dataset_dir`,
without importing builders.
Keep tokenizer models and their metadata together in `artifacts/tokenizers/` and pass
`--tokenizer` explicitly when selecting a different tokenizer. Tokenizer-building
scripts live in `src/dataset/tokenizer/`.

Local defaults include `artifacts/runs/` for training outputs, `artifacts/results/` for run registries,
and `artifacts/cache/huggingface/` for downloads. Dataset-building scripts retain
their existing CLI/environment controls. The new training launchers use TOML
and CLI overrides. Modal uses `/datasets` and `/outputs`.

`artifacts/` is ignored by Git. Preserve it when changing branches: it holds local
artifacts, not disposable copies. The dataset/tokenizer builders and their artifact format are unchanged. `notes/` retains historical
paths and published experiment results.

## Main branch scope

This branch keeps its existing clean_mc4_it dataset/tokenizer pipeline under
`src/dataset/`. The shared config/trainer refactor does not bring over the scaling
branch's production data pipeline, analysis programs or evaluation tools.
The example uses the 16,000-token base preset to match the usual tokenizer;
tuned presets use 16,384 tokens and require matching dataset/tokenizer artifacts.
