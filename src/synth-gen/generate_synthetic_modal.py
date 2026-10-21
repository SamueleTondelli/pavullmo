"""Offline vLLM generation into a Modal Volume; see notes/synthetic_generation.md."""

from __future__ import annotations

import hashlib
from itertools import islice
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Iterator

import modal

from config import SyntheticGenerationConfig, configure_logging
from pavullmo.config_cli import parse_extends, parse_overrides, resolve_config_path

VLLM_VERSION = "0.32.0"
CUDA_IMAGE = "nvidia/cuda:13.0.2-devel-ubuntu24.04"
OUTPUT_MOUNT = "/outputs"
CACHE_MOUNT = "/cache"
SOURCE_ROOT = "/workspace/src"
PRESET_DIR = Path(__file__).resolve().parent / "configs"

# Inference has its own PyTorch/CUDA dependencies. Never install it into the
# locked pretraining environment or change that environment's torch pin.
# FlashInfer can JIT-compile sampling/attention kernels during engine startup;
# CUDA runtime wheels alone do not provide nvcc or the development headers.
image = modal.Image.debian_slim(python_version="3.12")
if modal.is_local():
    image = (
        modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
        .entrypoint([])
        .apt_install("libnuma1", "build-essential", "ninja-build")
        .env({"CUDA_HOME": "/usr/local/cuda"})
        .run_commands("nvcc --version", "c++ --version")
        .uv_pip_install(f"vllm=={VLLM_VERSION}", "pydantic>=2.12,<3", "tomlkit>=0.13,<1", "loguru>=0.7,<1")
        .env({
            "PYTHONPATH": SOURCE_ROOT,
            "HF_HOME": f"{CACHE_MOUNT}/huggingface",
            "VLLM_CACHE_ROOT": f"{CACHE_MOUNT}/vllm",
        })
        .add_local_dir(Path(__file__).resolve().parents[1] / "pavullmo", remote_path=f"{SOURCE_ROOT}/pavullmo")
        .add_local_dir(Path(__file__).resolve().parents[1] / "config", remote_path=f"{SOURCE_ROOT}/config")
    )

app = modal.App("pavullmo-synthetic", image=image)


def read_prompts(path: Path) -> Iterator[dict[str, Any]]:
    """Read JSONL strings/objects or one prompt per nonblank .txt line."""
    plain = path.suffix.lower() == ".txt"
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = line.rstrip("\r\n") if plain else json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if isinstance(value, str):
                value = {"prompt": value}
            if not isinstance(value, dict) or not isinstance(value.get("prompt"), str) or not value["prompt"].strip():
                raise ValueError(f"{path}:{line_number}: expected a nonempty string or an object with a string 'prompt'")
            yield value


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def generate_batch(llm: Any, records: list[dict[str, Any]], sampling: Any,
                   config: SyntheticGenerationConfig) -> list[Any]:
    if config.prompt_mode == "raw":
        return llm.generate([row["prompt"] for row in records], sampling, use_tqdm=False)
    conversations = []
    for row in records:
        messages = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})
        messages.append({"role": "user", "content": row["prompt"]})
        conversations.append(messages)
    return llm.chat(conversations, sampling, use_tqdm=False,
                    chat_template_kwargs=config.chat_template_kwargs.to_kwargs())


def run_generation(
    config: SyntheticGenerationConfig, prompts_path: Path, output_dir: Path, *,
    llm_factory: Callable[..., Any], sampling_factory: Callable[..., Any],
    synchronize: Callable[[], None], publish: Callable[[], None],
    provenance: dict[str, Any], clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Persist bounded batches; count Unicode characters of completions only.

    Generation time includes chat formatting, teacher tokenization, scheduling,
    prefill, decode, and detokenization. It excludes load/warmup, output writing,
    and Volume commits. Job time includes these through output publication,
    excluding container startup and local upload (outside this function).
    """
    started = clock()
    output_dir.mkdir(parents=True, exist_ok=True)
    partial = output_dir / "generations.jsonl.partial"
    if any((output_dir / name).exists() for name in (partial.name, "generations.jsonl", "metrics.json")):
        raise FileExistsError(f"generation output already exists: {output_dir}")
    metrics: dict[str, Any] = {
        "status": "running", "generated_chars": 0, "generated_sequences": 0,
        "processed_prompts": 0, "generation_seconds": 0.0,
        "length_limited_sequences": 0, "output_path": str(partial),
        "character_unit": "Unicode code points; completion text only",
        **provenance,
    }
    config.save(output_dir / "config.toml")
    write_json(output_dir / "metrics.json", metrics)
    try:
        # Validate SamplingParams before paying for model loading.
        sampling = sampling_factory(**config.sampling.to_kwargs())
        llm = llm_factory(model=config.model, **config.engine.to_kwargs())
        with prompts_path.open("rb") as source:
            metrics["prompts_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
        warmup = list(islice(read_prompts(prompts_path), config.warmup_prompts))
        if warmup:
            generate_batch(llm, warmup, sampling, config)
            synchronize()
        records = read_prompts(prompts_path)
        with partial.open("x", encoding="utf-8") as destination:
            while batch := list(islice(records, config.batch_size)):
                synchronize()
                batch_started = clock()
                outputs = generate_batch(llm, batch, sampling, config)
                synchronize()
                elapsed = clock() - batch_started
                if len(outputs) != len(batch) or any(not output.outputs for output in outputs):
                    raise RuntimeError("vLLM did not return completions for every prompt")
                metrics["generation_seconds"] += elapsed
                offset = metrics["processed_prompts"]
                for index, (record, output) in enumerate(zip(batch, outputs, strict=True)):
                    for completion in output.outputs:
                        row = {
                            "prompt_index": offset + index,
                            "sample_index": completion.index,
                            "source": record,
                            "text": completion.text,
                            "finish_reason": completion.finish_reason,
                            "stop_reason": completion.stop_reason,
                        }
                        destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                        metrics["generated_chars"] += len(completion.text)
                        metrics["generated_sequences"] += 1
                        metrics["length_limited_sequences"] += completion.finish_reason == "length"
                metrics["processed_prompts"] += len(batch)
                destination.flush()
                write_json(output_dir / "metrics.json", metrics)
                publish()
                print(f"Saved {metrics['processed_prompts']} prompts, {metrics['generated_chars']} characters", flush=True)
        if not metrics["processed_prompts"]:
            raise ValueError("prompt file contains no prompts")
        final = output_dir / "generations.jsonl"
        partial.replace(final)
        metrics["output_path"] = str(final)
        publish()
        metrics["job_seconds"] = clock() - started
        metrics["chars_per_second"] = metrics["generated_chars"] / metrics["generation_seconds"]
        metrics["job_chars_per_second"] = metrics["generated_chars"] / metrics["job_seconds"]
        metrics["status"] = "completed"
    except Exception as error:
        metrics["status"] = "failed"
        metrics["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(output_dir / "metrics.json", metrics)
        publish()
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


@app.function(max_containers=1, single_use_containers=True)
def generate_remote(config_toml: str, output_volume_name: str,
                    provenance: dict[str, Any]) -> dict[str, Any]:
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    config = SyntheticGenerationConfig.from_toml(config_toml)
    volume = modal.Volume.from_name(output_volume_name)
    directory = Path(OUTPUT_MOUNT) / "synthetic" / config.name
    return run_generation(
        config, directory / "prompts.jsonl", directory,
        llm_factory=LLM, sampling_factory=SamplingParams,
        synchronize=torch.cuda.synchronize, publish=volume.commit,
        provenance={**provenance, "model": config.model, "vllm_version": vllm.__version__,
                    "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
                    "cuda_image": CUDA_IMAGE, "gpu_name": torch.cuda.get_device_name()},
    )


@app.local_entrypoint()
def main(
    config: str, override: str = "", extends: str = "", gpu: str = "L40S",
    output_volume: str = "pavullmo-training", cache_volume: str = "pavullmo-inference-cache",
    cpu: float = 4.0, memory_mb: int = 32768, timeout_seconds: int = 86400,
    hf_secret: str = "",
) -> dict[str, Any]:
    """--config, --extends and --override use the training launchers' TOML syntax."""
    configure_logging()
    path = resolve_config_path(config, preset_dir=PRESET_DIR)
    includes = [resolve_config_path(item, relative_to=path.parent, preset_dir=PRESET_DIR)
                for item in parse_extends([extends])]
    run = SyntheticGenerationConfig.load(path, overrides=parse_overrides([override]), extends=includes)
    # Reject settings that TOML cannot snapshot before creating cloud resources.
    config_toml = run.to_toml()
    run_name = run.name
    if cpu <= 0 or memory_mb <= 0 or not 0 < timeout_seconds <= 86400:
        raise ValueError("CPU/memory must be positive; timeout must be between 1 and 86400 seconds")
    if output_volume == cache_volume:
        raise ValueError("output and inference cache Volumes must differ")
    if run.engine.tensor_parallel_size > int(gpu.rsplit(":", 1)[1] if ":" in gpu else 1):
        raise ValueError("tensor_parallel_size exceeds the requested GPU count")

    # Normalize and validate the whole input before provisioning any cloud work.
    # Upload through a Volume rather than serializing a large corpus into an RPC.
    with tempfile.TemporaryDirectory(prefix="pavullmo-prompts-") as temporary:
        normalized = Path(temporary) / "prompts.jsonl"
        prompt_count = 0
        with normalized.open("w", encoding="utf-8") as destination:
            for record in read_prompts(Path(run.prompts).expanduser()):
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                prompt_count += 1
        if not prompt_count:
            raise ValueError("prompt file contains no prompts")
        outputs = modal.Volume.from_name(output_volume, create_if_missing=True)
        cache = modal.Volume.from_name(cache_volume, create_if_missing=True)
        with outputs.batch_upload() as upload:
            upload.put_file(normalized, f"/synthetic/{run_name}/prompts.jsonl")

    function = generate_remote.with_options(
        gpu=gpu, cpu=cpu, memory=memory_mb, timeout=timeout_seconds,
        volumes={OUTPUT_MOUNT: outputs, CACHE_MOUNT: cache},
        secrets=[modal.Secret.from_name(hf_secret)] if hf_secret else [],
    )
    print(f"Generating {prompt_count} prompts into {output_volume}:/synthetic/{run_name}/", flush=True)
    result = function.remote(config_toml, output_volume, {
        "run_name": run_name, "output_volume": output_volume, "prompt_count": prompt_count,
        "infrastructure": {"gpu": gpu, "cpu": cpu, "memory_mb": memory_mb,
                           "timeout_seconds": timeout_seconds, "cache_volume": cache_volume},
    })
    print(f"Generation: {result['chars_per_second']:,.2f} chars/s; "
          f"job: {result['job_chars_per_second']:,.2f} chars/s", flush=True)
    print(f"Saved to {output_volume}:/synthetic/{run_name}/generations.jsonl", flush=True)
    return result
