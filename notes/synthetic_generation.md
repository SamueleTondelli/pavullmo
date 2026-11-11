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

## First FP8 size comparison

The initial representatives use official prequantized Qwen3 checkpoints:

| Approximate bucket | Checkpoint | Reported parameters | Config preset |
| --- | --- | --- | --- |
| 1B | [Qwen3-1.7B-FP8](https://huggingface.co/Qwen/Qwen3-1.7B-FP8) | 1.7B | `throughput_qwen3_1_7b_fp8` |
| 4B | [Qwen3-4B-Instruct-2507-FP8](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8) | 4.0B | `throughput_qwen3_4b_fp8` |
| 7B | [Qwen3-8B-FP8](https://huggingface.co/Qwen/Qwen3-8B-FP8) | 8.2B | `throughput_qwen3_8b_fp8` |

These are practical throughput representatives within approximate size buckets,
all below 10B parameters. Using one dense architecture family reduces variation
in inference support and kernel behavior. They are not a controlled quality
scaling study: the 4B checkpoint has newer instruction tuning than the other two.
The 1.7B and 8B models have thinking disabled through their templates; the 4B
checkpoint is already non-thinking. Italian output quality still needs review.

The presets extend `throughput_fp8.toml`, sharing prompts, output limit, sampling,
seed, context limit, submission size, and initial concurrency. They explicitly
select FP8 quantization while retaining BF16 for nonquantized operations and
`kv_cache_dtype="auto"` (BF16 for these checkpoints). FP8 KV-cache storage is a
separate experiment. The official checkpoints avoid quantizing BF16 weights
during startup. vLLM documents native FP8 W8A8 on Ada/Hopper/Blackwell GPUs,
including the L4/L40S/H100 hardware classes; Ampere GPUs use a different
weight-only FP8 execution path. See [vLLM FP8 support](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/).

`throughput_10k.toml` is a ready-to-run config for the 10,000-prompt input. It
defaults to the 1.7B FP8 model and submits all 10,000 prompts in one call,
avoiding intermediate submission drains. Select another model with CLI
`--extends`, and set the run name and GPU concurrency with `--override`:

```bash
uv run --extra cloud modal run --detach src/synth-gen/generate_synthetic_modal.py \
  --config throughput_10k \
  --extends throughput_qwen3_4b_fp8 \
  --override 'name=qwen3-4b-fp8-l40s-seqs512-01;engine.max_num_seqs=512' \
  --gpu L40S
```

Use `throughput_qwen3_1_7b_fp8` or `throughput_qwen3_8b_fp8` for the other sizes.
Here `batch_size=10000` is the submission size; `engine.max_num_seqs` is the
active GPU sequence limit to sweep.

For a first comparison, run each model on the same GPU, starting with L40S:

```bash
uv run --extra cloud modal run --detach src/synth-gen/generate_synthetic_modal.py \
  --config throughput_qwen3_1_7b_fp8 --override 'name=qwen3-1p7b-fp8-l40s-01' --gpu L40S

uv run --extra cloud modal run --detach src/synth-gen/generate_synthetic_modal.py \
  --config throughput_qwen3_4b_fp8 --override 'name=qwen3-4b-fp8-l40s-01' --gpu L40S

uv run --extra cloud modal run --detach src/synth-gen/generate_synthetic_modal.py \
  --config throughput_qwen3_8b_fp8 --override 'name=qwen3-8b-fp8-l40s-01' --gpu L40S
```

Wait for each job to complete before starting the next on the shared output
Volume. Sweep `engine.max_num_seqs` through 128, 256, and 512 after this common
baseline; these are requested upper bounds, not guaranteed active batch sizes.
Prefix caching is disabled because the 10,000-row input repeats 100 prompts.
For quality inspection use `--override 'name=quality-trial;prompts=artifacts/synthetic/prompts_100.jsonl'`.
Repeated inputs with a fixed sampling seed can produce repeated completions;
they do not supply 10,000 independent quality examples.

Parameter count approximates weight memory, but total serving VRAM and throughput
do not transfer automatically between equally sized models. KV-head count,
layer count, attention type, context lengths, precision, vocabulary, and runtime
workspaces matter. For example, the Qwen3 4B and 8B configs both specify 36 layers,
8 KV heads, and head dimension 128. With BF16 KV, their calculated cache footprint
is the same 144 KiB per cached token, excluding allocator overhead, despite
their different weight sizes. Sources: [4B config](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8/blob/main/config.json),
[8B config](https://huggingface.co/Qwen/Qwen3-8B-FP8/blob/main/config.json).
Treat these measurements as initial capacity estimates when switching model
families, then check the new model's memory and batching behavior.

API references: [vLLM offline inference](https://docs.vllm.ai/en/v0.32.0/serving/offline_inference/),
[vLLM LLM API](https://docs.vllm.ai/en/v0.32.0/api/vllm/entrypoints/llm/),
and [Modal Volumes](https://modal.com/docs/guide/volumes).
For the inference toolchain, see [Modal's CUDA image guidance](https://modal.com/docs/guide/cuda)
and [vLLM 0.32.0 release artifacts](https://github.com/vllm-project/vllm/releases/tag/v0.32.0).

## Gemma 4 FP8 comparison

`throughput_gemma4_e2b_fp8.toml` and `throughput_gemma4_e4b_fp8.toml`
select the community [E2B](https://huggingface.co/axteam/gemma-4-E2B-it-FP8-dynamic)
and [E4B](https://huggingface.co/axteam/gemma-4-E4B-it-FP8-dynamic)
instruction checkpoints from axteam, pinned to their verified revisions. Their
linear weights use per-channel FP8 with dynamic per-token FP8 activations;
embeddings, norms, and excluded layers remain BF16. The presets override
`engine.quantization="compressed-tensors"` to match the checkpoint format.
This still executes FP8 W8A8; `"fp8"` is the loader for a different serialized
format and must not be forced on these checkpoints.

Use the same 10,000-prompt workload and L40S/concurrency settings as the Qwen
sweep, changing only the model preset and unique run name:

```bash
uv run --extra cloud modal run src/synth-gen/generate_synthetic_modal.py \
  --config src/synth-gen/configs/throughput_10k.toml \
  --extends src/synth-gen/configs/throughput_gemma4_e2b_fp8.toml \
  --override 'name=gemma4-e2b-fp8-l40s-seqs1024-01;engine.max_num_seqs=1024' \
  --gpu l40s
```

Swap `e2b` for `e4b` for the second run, after the first completes. Both inherit
the same sampling settings, 2,048-token output cap, BF16 KV cache, disabled
thinking and prefix caching, and text-only engine setting. A reusable sequential
launcher is saved locally at `artifacts/scripts/sweep_throughput_gemma4_synth_modal.sh`;
its optional first argument sets the run-name suffix (default: UTC timestamp).

Measured on 2026-10-25 with one L40S and `engine.max_num_seqs=1024`:

| FP8 model | Generation chars/s | Remote job chars/s | Generated characters | Generation seconds |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 E2B | 63,163.95 | 37,080.79 | 22,811,878 | 361.153 |
| Gemma 4 E4B | 35,209.01 | 23,979.16 | 22,103,355 | 627.776 |

Each run completed all 10,000 prompts with nonempty outputs and no length-limit
finishes. Recounting the saved JSONL reproduced both character totals exactly;
the normalized prompt hashes and resolved workload settings match. These are
one-run whole-submission averages. Generation chars/s excludes model loading,
warmup and persistence; remote job chars/s includes those costs (as defined above).
E2B delivered 1.79 times E4B's generation chars/s at this configuration.
Quality has not been scored.

The outputs and standalone TOML configs are saved in `pavullmo-training` under
`/synthetic/synth_l40s_gemma4_e2b_1024_20261025T1616Z/` and
`/synthetic/synth_l40s_gemma4_e4b_1024_20261025T1616Z/`. Local logs, downloaded
outputs, config snapshots, metrics, validation counts and a report are in
`artifacts/results/synthetic/gemma4_20261025T1616Z/`.

### L40S concurrency and memory sweep

The follow-up sweep on 2026-10-25 retained the same 10,000 prompts, submission
size, sampling, output cap, model revisions, BF16 KV cache and token budget.
It enabled statistics with `engine.extra_kwargs={disable_log_stats=false}`.
The dictionary-valued override merges with the existing prefix-cache setting;
individual keys inside `extra_kwargs` are not typed dotted override paths.

| Model | `max_num_seqs` | `gpu_memory_utilization` | Generation chars/s | Outcome |
| --- | ---: | ---: | ---: | --- |
| E2B | 1,024 | 0.90 | 61,930.49 | Completed |
| E2B | 1,536 | 0.90 | 63,116.43 | Completed |
| E2B | 2,048 | 0.90 | **64,826.19** | Completed |
| E4B | 512 | 0.90 | 34,771.36 | Completed |
| E4B | 768 | 0.90 | 35,835.04 | Completed |
| E4B | 1,024 | 0.90 | **36,451.50** | Completed |
| E4B | 1,536 | 0.90 | 33,927.11 | Completed |
| E2B | 2,048 | 0.95 | — | CUDA OOM during sampling initialization |
| E4B | 1,024 | 0.95 | — | CUDA OOM during sampling initialization |

Best measured settings were E2B at 2,048 and E4B at 1,024, both with a 0.90
memory budget. E2B gained 4.68% over its new 1,024 baseline. E4B lost 6.93%
when increasing from 1,024 to 1,536. At the 1,536 ceiling, E4B's KV cache was
at least 99% full in 91.5% of samples with queued requests; average logged
active requests were only 1,092. At 1,024, those figures were 1.8% and 1,020.
Exact preemption counters were not emitted in the logs.

Both 0.95 attempts failed before measured generation. E2B's FlashInfer
logits-mask buffer needed 2 GiB with 1.15 GiB free; E4B's softmax needed 1 GiB
with about 756 MiB free. Retain 0.90 for these tested settings so temporary
sampling buffers have room. Larger E2B runs at 0.90 also logged allocation
failures, then continued and completed; their logs are preserved.

These are single measurements per point, and small differences need caution.
Startup profiling produced different KV byte allocations across launches;
the sweep compares the actual configurations rather than forcing fixed cache
sizes. The seven completed runs produced 70,000 nonempty completions with
verified character totals and complete prompt coverage. One E2B/2,048 output
hit the 2,048-token cap. Quality has not been scored.

The reusable sequential launcher is
`artifacts/scripts/sweep_gemma4_concurrency.py`. Full logs, metrics, standalone
TOML configs, downloaded outputs, CSV/JSON results and plots are under
`artifacts/results/synthetic/gemma4_concurrency_20261025T155540Z/`; its
`report.md` contains the exact Volume run paths and analysis. All outputs and
failed-run metadata remain in `pavullmo-training:/synthetic/`. No benchmark
containers were left running.
