# Synthetic text generation on Modal

`src/synth-gen/generate_synthetic_modal.py` runs offline vLLM inference on one
Modal worker and saves decoded text to `pavullmo-training`. It does not import
Pavullmo's tokenizer, load training datasets, or change pretraining dependencies.
The inference image separately installs pinned `vllm==0.32.0` and its own
PyTorch/CUDA stack. It uses `nvidia/cuda:13.0.2-devel-ubuntu24.04`, matching
the CUDA 13.0 vLLM release wheels, with Python 3.12, `nvcc`, CUDA headers, a C++
compiler, and Ninja. The compiler is necessary because FlashInfer can compile
sampling/attention kernels during engine startup even when vLLM is installed
from a prebuilt wheel. Image builds check `nvcc` and `c++` before installing
the inference packages. Install the local launcher with `uv sync --extra cloud`.

## Input and launch

Create a local UTF-8 JSONL file. Each nonblank line is a JSON string or an object
with a nonempty `prompt` string. JSON escaping permits multiline prompts. Object
fields such as `id`, source identifiers, and recipe metadata are preserved:

```jsonl
{"id":"wiki-1","prompt":"Riscrivi questo brano in italiano chiaro, preservando i fatti.\n\nIl testo originale..."}
{"id":"wiki-2","prompt":"Trasforma questo brano in domande e risposte.\n\nIl testo originale..."}
```

A `.txt` file is also accepted, with one prompt per nonblank physical line.
Use JSONL for multiline prompts. The whole input is validated locally before
cloud resources are created, then uploaded through the output Volume rather
than embedded in a large remote function call.

Create a run TOML file, for example `src/synth-gen/configs/generation.toml`:

```toml
extends = ["synthetic_qwen35_4b.toml"]
name = "qwen35-4b-toml-01"
prompts = "artifacts/synthetic/prompts_100x100.jsonl"
batch_size = 4096
warmup_prompts = 8

[engine]
max_model_len = 8192
max_num_seqs = 256
max_num_batched_tokens = 16384
gpu_memory_utilization = 0.9
```

Launch using the same TOML interface as pretraining:

```bash
uv run --extra cloud modal run --detach src/synth-gen/generate_synthetic_modal.py \
  --config src/synth-gen/configs/generation.toml \
  --override 'name=qwen35-4b-toml-02;engine.max_num_seqs=512;sampling.max_tokens=2048' \
  --gpu L40S
```

`name`, `model`, and `prompts` are mandatory. Choose a unique name per run.
The former model, prompt, batch, and JSON-option CLI flags are replaced by
TOML fields and `--override`.
The prompt path resolves relative to the process working directory, like
training paths; it is not relative to the TOML file. A config file or preset
name may omit `.toml`. Bundled presets also work directly:

```bash
uv run --extra cloud modal run src/synth-gen/generate_synthetic_modal.py \
  --config synthetic_qwen35_4b \
  --override 'name=small-trial;prompts=artifacts/synthetic/prompts_100.jsonl'
```

Authored `extends` lists compose recursively relative to the declaring file.
CLI `--extends 'first;second.toml'` appends includes, searching beside the main
file, then the working directory, then bundled presets. The main file's own
values win over includes; typed `--override` assignments win last. As with the
Modal training launcher, each flag takes one semicolon-separated string.
Quote the entire argument so the shell passes semicolons intact. Source TOML
files are never rewritten, and saved configs contain no `extends` directives.

Existing input/output files are never overwritten. Failed runs retain partial
output; automatic resume is not provided. No cloud job is launched merely by
importing the module or running the unit tests.

After an image/startup failure, rerun with a new `name`, for example through
`--override 'name=retry-02'`. The previous input and failure metadata are retained.

## Performance and generation controls

The typed `[engine]` TOML section configures `vllm.LLM`.
Defaults are `performance_mode=throughput`, `dtype=auto`, `max_model_len=8192`,
`max_num_seqs=256`, `max_num_batched_tokens=16384`,
`gpu_memory_utilization=0.9`, `enable_chunked_prefill=true`, and
`language_model_only=true`. Other typed controls are `tensor_parallel_size`
(default 1), `kv_cache_dtype` (default `auto`), `quantization`, and `revision`.
Empty `quantization` and `revision` strings mean to leave these arguments to
vLLM; the saved config retains the empty strings. Set `engine.revision` to pin
the teacher weights. Native arguments without typed fields belong in the
explicit `[engine.extra_kwargs]` table:

```toml
[engine.extra_kwargs]
enable_prefix_caching = true
enforce_eager = false
```

Or use `--override 'engine.extra_kwargs={enable_prefix_caching=true}'`.
These dictionaries merge recursively with authored values. New schema fields
automatically forward to vLLM, without a separate argument forwarding list.
Unknown TOML settings are warned about and retained, but not forwarded.
Explicit overrides reject unknown typed paths such as `engine.max_num_seq`.
Use the whole `extra_kwargs` table for advanced native options, rather than
unvalidated dotted paths inside it. Typed fields cannot be duplicated there.

The typed `[sampling]` section configures `vllm.SamplingParams`.
Defaults are `max_tokens=1024`, `temperature=0.7`, `top_p=0.9`, and `seed=42`.
Other typed fields are `n` (1), `top_k` (-1), and `stop` (an empty string list).
Use `sampling.n` for multiple sequences per prompt, or `sampling.stop` for stop
strings. Additional native sampling options go in `[sampling.extra_kwargs]`.
TOML has no null value: omit optional native arguments or use their documented
typed default, rather than inserting JSON nulls. The
teacher's tokenizer is necessarily used inside vLLM, but token IDs are not
saved and no throughput metric uses Pavullmo's tokenizer. Detokenization cannot
be disabled. Native engine and sampling argument validity is checked by vLLM;
unsupported engine/model/quantization combinations fail rather than silently
falling back to different authored options.

Chat mode is the default: each prompt becomes a user message, using the
teacher's chat template. The root `system_prompt` adds a system message.
`--override 'chat_template_kwargs.enable_thinking=true'` enables thinking; the default
is false for models such as Qwen3.5. This is a template argument, so whether it
disables reasoning depends on the teacher. Generated text is saved as returned,
including reasoning text if the selected model emits it.
Use `prompt_mode="raw"` for base models or preformatted prompts; no chat template
or thinking switch is applied in that mode.

The root `batch_size` controls how many prompts are handed to vLLM at once (default
4096), bounding host memory and the amount of unsaved work. This differs from
`max_num_seqs`, the active GPU sequence limit. Each submission is continuously
batched internally but drains completely before the next submission. Keep the
submission size well above active concurrency for a meaningful bulk benchmark;
small files and tiny batches underutilize the GPU.

The root `warmup_prompts` defaults to 8; these prompts are generated and discarded before
the measured pass. All input prompts are subsequently generated and saved.
Warmup uses the same sampling settings and can warm prefix caches; set it to 0
to measure the first inference pass. Model initialization is always excluded
from the generation-only timer.

Infrastructure options are `--gpu` (L40S), `--cpu` (4), `--memory-mb` (32768),
`--timeout-seconds` (86400 maximum), `--output-volume` (`pavullmo-training`), and
`--cache-volume` (`pavullmo-inference-cache`). For tensor parallelism, request
matching resources, for example `--gpu H100:2` and
`--override 'engine.tensor_parallel_size=2'`. Infrastructure flags remain outside
`SyntheticGenerationConfig`, like the Modal training launcher. For gated/private models, use
`--hf-secret <Modal-secret-name>` with a secret containing `HF_TOKEN`.

## Outputs and throughput

Each run writes `/synthetic/<name>/` in the output Volume:

- `prompts.jsonl`: normalized input, including source metadata.
- `config.toml`: complete resolved generation settings, including defaults,
  retained unknown settings, and CLI overrides, without include directives.
- `generations.jsonl`: one decoded completion per row, with `text`, source
  record, zero-based prompt/sample indices, and finish/stop reasons.
- `metrics.json`: settings provenance, versions, prompt hash, status, counts,
  timing, and throughput.

During generation, output is called `generations.jsonl.partial`. Each completed
submission is flushed, metrics are updated, and the Volume is committed. On
success the partial file is renamed; on failure the partial data and failed
status are committed. `length_limited_sequences` counts completions ending at
the generation limit so that truncation is visible.

`chars_per_second = generated_chars / generation_seconds`. Characters are Python
Unicode code points (`len(text)`), including whitespace and newlines. Only
completion text counts: prompts, JSON framing, and warmup output are excluded.
Time includes chat formatting, teacher tokenization, scheduler work, prefill,
decode, and detokenization. CUDA synchronization brackets each submission.
File writing, Volume commits, model loading, and warmup are excluded.

`job_chars_per_second` includes remote setup/model loading, warmup, input reads,
output writing, and Volume commits through publication of the final output.
It excludes image build, container startup, local input upload, and the final
metrics-file publication. Both metrics include the final partially occupied GPU
batch; neither is a peak-window estimate. The remote function and Python local
entry point return the metrics dictionary; both also print metrics to logs.

Download results, for example:

```bash
uv run --extra cloud modal volume get pavullmo-training \
  /synthetic/qwen35-4b-toml-01 ./artifacts/synthetic/qwen35-4b-toml-01
```

Use sequential jobs against a shared output Volume, consistent with pretraining's
Volume publication policy. Keep validation/test documents out of prompt inputs.

API references: [vLLM offline inference](https://docs.vllm.ai/en/v0.32.0/serving/offline_inference/),
[vLLM LLM API](https://docs.vllm.ai/en/v0.32.0/api/vllm/entrypoints/llm/),
and [Modal Volumes](https://modal.com/docs/guide/volumes).
For the inference toolchain, see [Modal's CUDA image guidance](https://modal.com/docs/guide/cuda)
and [vLLM 0.32.0 release artifacts](https://github.com/vllm-project/vllm/releases/tag/v0.32.0).
