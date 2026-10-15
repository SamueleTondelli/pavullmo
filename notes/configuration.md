# Typed TOML configuration

`src/config/` provides the configuration layer for the new PretrainingTrainer.
The new local and Modal entry points are `src/pavullmo/pretrain.py` and
`src/pavullmo/pretrain_modal.py`; shared training components live in `src/components/`.
Evaluation entry points still need adaptation to the refactored
model and checkpoint APIs. No
training or GPU imports are needed to load, edit, or save a configuration.

The installed package is `config` (repository tests can also use `src.config`).
Classes inherit from Pydantic's `BaseModel` through `Config`. Settings have typed
defaults except for the experiment name: `PretrainingConfig.name` has no default and is
required. All classes automatically share dictionary/TOML
conversion, validation, composition, and unknown-setting warnings.

```python
from config import PretrainingConfig, configure_logging

configure_logging()  # Optional: Loguru already has a default stderr sink.
run = PretrainingConfig.load("configs/experiment.toml")
configure_logging(run.logging)  # Explicitly install configured console/file sinks.

# Values are typed Python values, not command-line strings.
trial = run.with_overrides({"name": "lr_trial", "optimizer.lr": 0.001})
trial.save("artifacts/runs/lr_trial/config.toml")

payload = trial.to_dict()
remote_config = PretrainingConfig.from_dict(payload)
assert remote_config.to_dict() == payload

text = trial.to_toml()
reloaded = PretrainingConfig.from_toml(text)
```

## Configuration groups

| Group | Contents |
| --- | --- |
| `model` | Architecture, context length, dropout, initialization and attention options |
| `train_dataset` | `TokenDatasetConfig` for training: explicit artifact `path` |
| `validation_dataset` | Independent `TokenDatasetConfig` for validation |
| `optimizer` | `adamw` or `muon_adamw`, base LR and norm multiplier |
| `optimizer.adamw` | Betas, epsilon, weight decay and fused implementation |
| `optimizer.muon` | LR multiplier, momentum, decay and Newton-Schulz options |
| `optimizer.schedule` | Cosine/WSD, minimum LR, warmup and decay fraction |
| `training` | Batch/accumulation, duration, clipping, z-loss and seed |
| `evaluation` | Validation interval and number of batches |
| `runtime` | Dataset/output roots, CSV, loader and compilation settings |
| `logging` | Loguru settings, TensorBoard flushing and diagnostic interval |
| `checkpoint` | Periodic interval, retention count and full-state saving |

Existing numerical defaults are carried over where applicable. New checkpoint
defaults request full-state saving at completion/graceful stop, keeping two checkpoints.
`training.max_steps` of zero means no cap, and `checkpoint.interval_steps` of zero disables periodic
saves. The PretrainingTrainer validates tokenizer compatibility and available steps, while
the validation loader checks coverage of fixed evaluation samples.
Startup operations such as resume/init-from belong to the launcher and are not
embedded in a reusable run configuration.

The experiment name must be explicitly provided as the top-level `name` in the
composed TOML files, a dictionary, or an already parsed override such as
`{"name": "lr_trial"}`. Partial presets can omit it, but loading a complete
`PretrainingConfig` without it raises a validation error. Empty/invalid names are also
rejected; names must be filename-safe and start with an ASCII letter or digit.

## Model, optimizer and data-loader APIs

```python
from model import DecoderTransformer
from components.optimizer import Optimizer
from components.data import TrainDataLoader, ValidationDataLoader

model = DecoderTransformer(run.model)
train = TrainDataLoader("artifacts/datasets/train_10m", run)
validation = ValidationDataLoader("artifacts/datasets/validation", run)

# Computed execution state is passed separately, never added to the config.
total_steps = run.training.epochs * (len(train) // run.training.grad_accum_steps)
if run.training.max_steps:
    total_steps = min(total_steps, run.training.max_steps)
optimizer = Optimizer(run.optimizer, model, total_steps=total_steps)

# After backward/accumulation/clipping:
optimizer.step_optimizer()
optimizer.step_schedule()
```

The Transformer, its blocks, attention, FFN, RoPE and Canon architectural layers
take only `ModelConfig`. The generic `CausalDepthwiseConv` primitive retains
explicit channel/kernel dimensions; the Canon layers derive those dimensions
from the config. Parameter names and layouts remain compatible with existing
state dictionaries. `model.attn_bias` defaults to false and
`model.canon_kernel_size` defaults to four.

`ScheduleConfig` is nested under `OptimizerConfig`. Previously saved top-level
`schedule` sections migrate with a warning; explicitly nested settings take
precedence. Saves use the new nested layout. `total_steps` is a required keyword
argument to `Optimizer`, held as execution state rather than a config field.
The constructor validates that it is a positive integer larger than warmup.
The optimizer wrapper owns
AdamW, or Muon for hidden matrices plus AdamW for the remaining parameters. It
preserves embedding/norm/bias decay exclusions, norm/Muon LR multipliers and the
existing cosine/WSD formulas. Its named `optimizers` and `schedulers` dictionaries
are available for diagnostics and checkpoint state. Gradient clearing,
accumulation, clipping and device placement remain trainer responsibilities.

The loader wrappers take an explicit artifact path and the run config for
batching, context length and runtime settings; they never infer a dataset path
from prefix/variant or `data` fields. The sharded uint16 reader preserves shifted
targets, shared boundary tokens and fixed validation offsets. Training shuffles
deterministically and drops incomplete batches. Call
`train.set_epoch(epoch, start_batch=...)` to select the epoch/cursor; prefetching
does not advance the recorded cursor. Each loader has its own seeded generator,
so iterator creation leaves the model's dropout RNG alone. Validation is ordered,
retains the final partial batch and requires enough evaluation steps to cover a
fixed validation sample. Worker persistence and prefetching are configurable via
`runtime.persistent_workers` and `runtime.prefetch_factor` and are only enabled
when workers exist.

## Shared trainer components

`TrainerConfig` contains the common name, model, optimizer, runtime, logging and
checkpoint sections. `PretrainingConfig` extends it with independent
`train_dataset` and `validation_dataset` token dataset configs, training and
evaluation settings. Other stages can define their own typed root
subclasses; serialization, composition and overrides work without changes.
The rename from `RunConfig` to `PretrainingConfig` changes Python imports only:
existing resolved TOML, presets and checkpoint configurations remain loadable.

`TokenDatasetConfig` describes one artifact via its required `path` field.
Pretraining supplies defaults of `train_10m` and `validation` for its two sections.
Each path can be absolute or relative to `runtime.dataset_dir`; sequence length
still comes from `model.seq_len`, while batch and worker settings retain their
existing training/runtime sections. For example:

```toml
[train_dataset]
path = "train_16k_195m"

[validation_dataset]
path = "validation_16k"
```

Use `--override 'train_dataset.path=another_artifact'` to change a selection.
Former `[data]` sections with `train`/`validation` string fields migrate with a
warning when loading saved configs/checkpoints. Explicit new fields take
precedence; unknown legacy fields remain preserved and warn normally. Saved
configs use the new sections and retain any unknown settings.

`GenericTrainer[StageConfig]` initializes shared components from that stage's
config plus an explicit `seed` and computed `total_steps`. It creates the model,
optimizer/schedules, optional compiled forward model, Loguru/TensorBoard output
and checkpoint manager. It does not construct datasets or implement a training
loop, evaluation, accumulation, optimizer stepping or checkpoint state.

Startup emits one INFO log per top-level config section in compact TOML.
`GenericTrainer` logs the common fields declared by `TrainerConfig`;
`PretrainingTrainer` logs its additional dataset/training/evaluation sections
and any retained root extras, excluding the shared fields. This partition is
derived from the schema, so new shared fields are picked up automatically.

`forward_backward(microbatch, loss_fn, loss_kwargs=...)` performs one model
forward under BF16 CUDA autocast, calls `loss_fn(outputs, **loss_kwargs)`, runs
backward and returns a detached scalar loss. A tensor is passed positionally;
a mapping is expanded into model keyword arguments. Inputs and loss arguments
must already be on the correct device. The loss callback controls normalization
and accumulation scaling. The helper never clears/clips gradients or steps an
optimizer/schedule. It marks compiled CUDA-graph forward steps; ownership of
accumulation buffers remains with the specialized trainer.

`PretrainingTrainer` in `components/pretraining_trainer.py` inherits that base.
Its `compute_loss()` implements cross-entropy plus optional z-loss and scales
the objective for its accumulation count. It owns the complete pretraining
loop, diagnostics, validation, progress and save/resume state. CheckpointManager
depends on a small structural trainer interface rather than the pretraining
class; each stage supplies its own state dictionaries and save-boundary rules.

## Training and checkpointing

```python
from components.pretraining_trainer import PretrainingTrainer

with PretrainingTrainer(run) as trainer:
    trainer.train()
    path = trainer.checkpoints.latest_checkpoint

# Reconstruct from the embedded TOML config and restore full state.
with PretrainingTrainer.from_checkpoint(path) as trainer:
    trainer.train()  # Continues remaining steps; completed runs do not retrain.

# Paths can be explicitly changed through config overrides when relocating.
with PretrainingTrainer.from_checkpoint(path, overrides={
    "runtime.dataset_dir": "/datasets",
    "runtime.output_dir": "/outputs/runs",
    "runtime.runs_csv": "/outputs/pretrain_runs.csv",
}) as trainer:
    pass
```

`PretrainingTrainer(PretrainingConfig)` requires BF16-capable CUDA, seeds Python/PyTorch/CUDA, resolves
train/validation artifact paths under `runtime.dataset_dir` (absolute paths remain
absolute), constructs loaders and validates tokenizer vocabulary/hash compatibility.
It calculates optimizer steps from full training batches, gradient accumulation,
epochs and the optional step cap, then passes that horizon directly to
`Optimizer(..., total_steps=...)`. Model parameters and optimizer buffers
stay FP32. Compilation wraps the forward model while the base model remains
available for state dictionaries.

The constructor configures Loguru, saves `config.toml` under
`runtime.output_dir/name`, and opens TensorBoard under that directory's
`tensorboard/` subdirectory. Use the context manager or `close()` to close the
writer. Training, evaluation, CSV result registration and automatic checkpoint
timing are owned by `PretrainingTrainer` and shared by both launchers.

`PretrainingTrainer` takes only `PretrainingConfig`. Components retain a validated copy of the
user's settings without adding computed values or replacing fields during
setup/training. Dataset/output roots and the CSV path come exclusively from
`config.runtime`, resolved against the process working directory. Neither the
constructor nor a provider wrapper supplies alternate paths. Saved TOML,
checkpoint configuration and CSV configuration contain the user settings.
Computed horizons/counters belong to component attributes and checkpoint state.

`train()` runs to the resolved optimizer-step horizon. Each update accumulates
the configured number of full microbatches, uses BF16 autocast with FP32 model
parameters and optimizer buffers, adds optional z-loss, checks for finite loss
and gradients, clips gradients, steps all optimizers, then steps all schedules.
Only then are progress and the data cursor committed. An epoch's incomplete
accumulation group is dropped, consistently with the resolved horizon. Compiled
training marks each microbatch as a CUDA-graph step and copies its first gradient
buffers outside graph storage for accumulation; validation uses the base model
eagerly to accommodate its partial final batch.

Cross-entropy, objective/z-loss, pre-clipping gradient norm, clipping frequency,
learning rates used for each update and token count are written to
TensorBoard at every completed step and summarized through Loguru. Parameter
and pre-clipping gradient-group norms are written at the first/final update and
every `logging.diagnostics_interval` updates. Split QKV retains the combined
`gradient_groups/attention_qkv` metric and adds query/key/value metrics.

TensorBoard throughput averages windows of `logging.throughput_interval_steps`
updates (default 20). Windows also finish before validation/checkpointing and
at completion or graceful stop, including partial windows. The wall-clock timer
includes data loading and per-step logging but excludes validation/checkpoint
pauses. CUDA synchronizes at timing boundaries rather than every update.
`runtime/cuda_peak_allocated_bytes` still samples allocator statistics each
update without synchronization; it reports the process lifetime peak.

`validate()` evaluates the first `evaluation.steps` ordered batches, weighting
cross-entropy by target tokens, including the partial last batch. It restores
the previous model mode even on failure. Training calls it every
`evaluation.interval_steps` and at the final step; it records current/best loss
and `last_validation_step`. Validation has its own loader RNG and never consumes
the training dropout RNG.

Periodic saves follow `checkpoint.interval_steps`; zero disables periodic saves.
Final and graceful-stop saves still happen. `request_stop()` asks the loop to
finish its current complete update, save and return. In the main thread, SIGINT
(Ctrl+C) and SIGTERM request the same stop; previous signal handlers are restored
on return. In a worker thread, the wrapper should call `request_stop()` itself.
The same PretrainingTrainer can continue via another `train()` call, or a new PretrainingTrainer can
resume the saved checkpoint. Exceptions propagate and flush metrics; an
incomplete/failed update cannot be saved or continued until a checkpoint is
restored. Abrupt process termination can resume the most recent published save.

On success, the final checkpoint and full resolved TOML configuration are
registered in `runtime.runs_csv`. The historical column names are retained:
`dataset_variant` holds the explicit `train_dataset.path` artifact path, `train_script`
identifies `components.pretraining_trainer.PretrainingTrainer`, and `config` contains
single-line TOML from `config.to_string()`. Repeated
completion calls with the same checkpoint path/config do not duplicate a row.
Nested sections use inline tables in a `config = { ... }` document; embedded
newlines in strings are escaped, so the config never spans physical CSV lines.
All config classes inherit `to_string()` and `from_string()` automatically.
Read a result with `PretrainingConfig.from_string(row["config"])`; this method
also accepts older multiline TOML values. Existing CSVs are not rewritten, and
matching older multiline rows still count for duplicate detection. `to_toml()`,
saved config files and embedded checkpoint configs retain their usual format.
Use a unique run name and one CSV writer at a time; parallel cloud sweeps should
give workers separate registries and merge them in their coordinator.

The PretrainingTrainer itself owns progress: `global_step`, `epoch`, `next_batch` (the next
consumed microbatch position), `tokens_seen`, `clipped_steps`, `train_loss`,
`validation_loss`, `best_validation_loss` and `last_validation_step`.
`state_dict()` captures the base model, named optimizer/scheduler states and parameter assignments, these counters,
training settings, dataset identities, model train/eval mode, and Python/PyTorch/
CUDA/loader RNG streams. `load_state_dict()` restores those states, sets the
sampler cursor and restores RNG last. Applied gradients are cleared on restore.
The current epoch's iterator seed is also saved. Recreating an iterator on
resume restores that seed without consuming the saved loader RNG stream again;
logical loader RNG advancement is consistent with persistent/nonpersistent
workers. Prefetched but unconsumed token blocks are re-read deterministically.

`trainer.checkpoints` is a small `CheckpointManager` for synchronous file I/O:

- `trainer.save_checkpoint()` writes `checkpoints/step_XXXXXXXX.pt` atomically.
- `trainer.load_checkpoint(path)` restores into an existing compatible trainer;
  omitting the path loads `trainer.checkpoints.latest_checkpoint`.
- `PretrainingTrainer.from_checkpoint(path, overrides=...)` constructs a trainer from the
  checkpoint's resolved TOML configuration before restoring its state.
- `CheckpointManager.read(path)` inspects a checkpoint on CPU without constructing
  a trainer. Configuration is embedded as TOML text; no config/model Python
  instances or compiled wrappers are pickled.

Retention uses `checkpoint.keep_last`, runs only after a successful write, and
touches only managed step files. The latest checkpoint is the most recently
saved file, including when restarting from an older step. Setting
`checkpoint.save_training_state=false` saves weights and configuration only;
such exports cannot resume full trainer state.

Full resume validates the original model/optimizer/schedule/training/evaluation
settings and horizon, parameter assignments, dataset identities and device type/
PyTorch version. Runtime paths, loader/compilation controls, logging, retention
and the experiment label may change. Unknown config settings remain preserved,
but are excluded from the supported training-settings signature. Dataset identity
uses token counts, tokenizer/shard hashes declared in metadata and fixed evaluation
offsets, independently of file locations; startup does not rehash the entire corpus.
Older config files containing `optimizer.schedule.total_steps` follow the normal
unknown-field rule: preserved with a warning, never applied. Older refactor
checkpoint signatures ignore that former field; the horizon in checkpoint
progress is still validated against the actual dataset/training duration.

Saving requires a completed optimizer **and** scheduler update, with no pending
accumulation (initial step zero is also valid). PretrainingTrainer enforces this boundary,
and resume validates that the cursor, token count and scheduler step agree. PretrainingTrainer
save/load here implements full resume, not a new training phase or weights-only
initialization. `CheckpointManager.after_save` is an optional runtime callback
invoked after successful publication/retention. The Modal wrapper uses it to
commit its output Volume after every checkpoint, including periodic saves.

Generation uses `generate.py <saved-directory> <tokenizer.model>`. It selects
the most recently saved managed checkpoint under `checkpoints/`, loading only
its embedded `[model]` as `ModelConfig` and its weights. It supports full and
weights-only checkpoints on CPU or CUDA without validating training paths or
constructing a PretrainingTrainer. `--checkpoint` can select an absolute path or a path
relative to the supplied directory.

For legacy checkpoints, run `migrate_checkpoint_config.py <old.pt>` first.
It writes a standalone, partial `config.toml` with a `[model]` section beside
the checkpoint, translating recorded hyperparameters and recovering missing
layout flags from weights. Weight keys/shapes are checked before publication.
The checkpoint is unchanged; `--force` is required to replace an existing
config. Legacy generation reads this sidecar config. One `.pt` is selected
automatically; several require `--checkpoint`, and each must match the shared
model config. Migration does not provide full training resume.

## Presets and composition

Preset files contain partial configuration sections:

```toml
# models/10m.toml
[model]
n_blocks = 8
embed_dim = 256
attn_heads = 4
ffn_dim = 704
```

```toml
# experiment.toml
extends = ["models/10m.toml", "optimizers/muon.toml"]
name = "experiment"

[optimizer]
lr = 0.001
```

`extends` is a reserved application directive, not TOML syntax for includes.
Paths are relative to the declaring file; absolute paths are also accepted.
Nested includes work, while missing files, invalid include lists and cycles fail.

Resolution order is:

1. Each included file, recursively, in listed order, followed by additional
   include paths supplied through `load(..., extends=[...])` or CLI `--extends`.
2. The declaring file's own settings.
3. Subsequent top-level files passed to `PretrainingConfig.load([first, second])`.
4. Explicit dotted overrides passed to `load(..., overrides={...})`.
5. Validation and defaults for settings still missing.

Tables merge recursively. Scalars and arrays replace earlier values; arrays do
not append. Presets are merged as raw dictionaries before Pydantic inserts
defaults. Saved configs contain the full resolved values and retained unknown
settings, without `extends`; source files are not required to reload them.
Snapshots use a same-directory temporary file and atomic replacement.

`PretrainingConfig.load(path, extends=["runtime/cloud.toml"])` appends parsed include
paths to the file's existing `extends` for that load, without modifying the
source. Additional paths resolve relative to that top-level file, just like
authored includes; their own nested includes still resolve relative to them.

## Overrides and command-line integration

The config layer accepts already parsed Python values at dotted field paths.
It does not read environment variables or command-line arguments, and does not
depend on `argparse`.

```python
# The train scripts parse --override arguments into this dictionary.
overrides = {
    "optimizer.lr": 0.001,
    "model.qk_norm": True,
    "optimizer.kind": "muon_adamw",
    "optimizer.muon.ns_coefficients": [3.4445, -4.775, 2.0315],
}
run = PretrainingConfig.load("configs/experiment.toml", overrides=overrides)

# Already loaded configs support the same API through a validated copy.
trial = run.with_overrides({"training.batch_size": 16})
```

Values are not parsed or coerced from command-line strings here: a string
`"16"` for an integer field fails validation. Explicit override paths must be
supported by this branch; typos raise errors. `pavullmo.pretrain.parse_overrides`
converts CLI values into the dictionary passed to `load` or `with_overrides`.
The config classes and shared components do not depend on argument parsers or
read environment variables for training configuration.

```bash
uv run python src/pavullmo/pretrain.py --config src/pavullmo/configs/10m_tuned_config.toml \
  --override 'name=trial;train_dataset.path=train_16k_195m;validation_dataset.path=validation_16k;optimizer.kind=muon_adamw'

uv run --extra cloud modal run src/pavullmo/pretrain_modal.py \
  --config 10m_tuned_config.toml --extends modal_config \
  --override 'name=cloud_trial;train_dataset.path=train_16k_195m;validation_dataset.path=validation_16k;optimizer.kind=muon_adamw'
```

The semicolon-separated list is one shell-quoted argument. Values use TOML syntax
(numbers, lowercase booleans, quoted strings, arrays and inline tables); bare
strings/paths are also accepted. Semicolons inside TOML strings are retained.
Later assignments win; local `--override` may be repeated. Presets retain their
original model/optimizer values, compose with `extends`, and omit dataset
selection and run names. `optimizer.kind` defaults to AdamW, including presets
that also specify Muon settings; choose `muon_adamw` explicitly when desired.

Both launchers accept `--extends 'first;second.toml'`: these files append to the
main config's existing includes, in the given order. Local `--extends` can also
be repeated. The main file's own values still win over all includes, and explicit
`--override` assignments win last. The additional includes are not appended to
every nested file, and source files are never rewritten.

CLI `--config` accepts file paths or bundled preset names, with an optional
`.toml` suffix. Additional CLI include paths are searched beside the main config,
then in the working directory, then under `src/pavullmo/configs/` (or the installed
package's `configs/` directory). Main config paths search the working directory
then bundled presets. Authored TOML includes retain their normal file-relative
resolution. `modal_config.toml` supplies only the three standard Modal runtime
paths: `/datasets`, `/outputs/runs`, and `/outputs/pretrain_runs.csv`.

Local training accepts exactly one of `--config` and `--resume`. Resume reads the
embedded TOML rather than a config file. It accepts overrides for runtime,
logging, retention and labels; trajectory settings must match the checkpoint.
`--extends` requires `--config`; resumed checkpoints already contain a complete
resolved config, so use explicit `--override` assignments for resume edits.
The Modal CLI follows the same rule. Its resume path is read in the remote
container as supplied, with no implicit mount prefix. Use an absolute path such
as `/outputs/runs/trial/checkpoints/step_00001000.pt` for a checkpoint on the
output Volume. Relative resume paths resolve against the remote working directory.

Modal infrastructure flags are `--gpu` (L4), `--cpu` (4), `--memory-mb` (32768),
`--timeout-seconds` (86400), `--dataset-volume` (`pavullmo-datasets`) and
`--output-volume` (`pavullmo-training`). `--no-tensorboard` disables the live
public TensorBoard tunnel. Use `modal run --detach ...` for terminal-independent
runs. Infrastructure flags stay outside `PretrainingConfig`. The image uses the frozen
lockfile and mounts all `src/` modules; local image construction is skipped when
Modal imports the wrapper under `/root`.

Fresh-run files and includes resolve locally, then the complete standalone TOML
is sent to Modal, including unknown settings. Modal mounts the dataset Volume
read-only at `/datasets` and the output Volume at `/outputs`. The user must
configure `runtime.dataset_dir`, `runtime.output_dir` and `runtime.runs_csv` for
these locations, through a TOML config/preset or explicit CLI overrides. The
bundled `--extends modal_config` supplies their standard values when the main
file does not explicitly set them. The
wrapper does not set or inject paths. Relative `train_dataset.path`/`validation_dataset.path`
paths select artifacts within the configured dataset root; absolute paths must
exist in the remote container. TensorBoard watches the configured output root,
including after resume overrides. Outputs intended to persist must be configured
under `/outputs`; the Volume is committed after checkpoints and on function exit,
including after training or dashboard failures. No list of hyperparameters needs
maintenance when fields are added.

## Compatibility and logging

Missing optional fields/sections receive defaults; the experiment name remains
mandatory. Unknown fields and sections are
retained and generate Loguru warnings with full paths, for example
`training.batch_szie` or `profiling`. A completely unknown section produces one
warning, not a warning for every child. Recognized fields always undergo strict
validation; an unsupported optimizer selector or invalid batch size fails.

`load`, `from_toml`, `from_dict`, and `model_validate` log unknown settings.
Direct Pydantic constructors are intended for authored Python configurations;
use these boundary methods when consuming external dictionaries. No config
classes need custom serializers when fields are added or removed. Renames and
semantic changes still require explicit migration policy.

`configure_logging` replaces Loguru sinks only when explicitly called, supports
console and optional rotating file output, and disables exception-local dumps.
Config I/O emits debug composition/override messages, info load/save messages,
warnings for unknown settings, and errors for failed file loading/saving or
validation. Importing the package never changes existing logging sinks.

TOML has no null value. Config fields should use explicit flags or sentinel
values (such as an empty optional log-file string) rather than `None`. Dictionary
extras may retain `None`, but TOML export rejects it with its field path rather
than silently dropping it. TOML-native dates in unknown sections are preserved.
Generated snapshots do not retain source comments/formatting.

Run config tests with `uv run python -m unittest discover -s tests -p test_config.py`.
