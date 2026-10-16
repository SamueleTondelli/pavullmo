"""Launch TOML-configured pretraining on Modal with persistent checkpoints."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from typing import Any

import modal
from loguru import logger

from config import PretrainingConfig, configure_logging
from pavullmo.modal_start_tensorboard import ensure_tensorboard
from pavullmo.pretrain import parse_extends, parse_overrides, prepare_config, run_training

DATASET_MOUNT = "/datasets"
OUTPUT_MOUNT = "/outputs"
SOURCE_ROOT = "/workspace/src"
OUTPUT_COMMIT_SECONDS = 15

image = modal.Image.debian_slim(python_version="3.12")
if modal.is_local():
    project_root = Path(__file__).resolve().parents[2]
    image = (
        image.uv_sync(str(project_root), extras=["cloud"], frozen=True)
        .env({"PYTHONPATH": SOURCE_ROOT})
        .add_local_dir(project_root / "src", remote_path=SOURCE_ROOT)
    )

app = modal.App("pavullmo-train", image=image)


@contextmanager
def _publish_outputs(output_volume: modal.Volume):
    """Publish live events even when periodic checkpoints are disabled."""
    stopped = Event()

    def publish() -> None:
        while not stopped.wait(OUTPUT_COMMIT_SECONDS):
            try:
                output_volume.commit()
            except Exception:
                logger.exception("Unable to publish live outputs; retrying next interval")

    worker = Thread(target=publish, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join()


@app.function(max_containers=1, single_use_containers=True)
def train_remote(
    config_toml: str, resume: str, overrides: dict[str, Any],
    output_volume_name: str,
) -> dict[str, Any]:
    """Use configured paths directly; mounts are provided by the Modal launcher."""
    configure_logging()
    output_volume = modal.Volume.from_name(output_volume_name)
    try:
        config = PretrainingConfig.from_toml(config_toml) if config_toml else None
        with _publish_outputs(output_volume):
            return run_training(
                config, resume=Path(resume) if resume else None, overrides=overrides,
                checkpoint_saved=output_volume.commit,
            )
    finally:
        # Also retain partial metrics when training fails.
        output_volume.commit()
        logger.info("Committed outputs to Modal Volume '{}'", output_volume_name)


@app.local_entrypoint()
def main(
    config: str = "", override: str = "", resume: str = "", extends: str = "",
    gpu: str = "L4", cpu: float = 4.0, memory_mb: int = 32768,
    timeout_seconds: int = 86400,
    dataset_volume: str = "pavullmo-datasets", output_volume: str = "pavullmo-training",
    tensorboard: bool = True,
) -> None:
    """--config/--resume and --override match local training; other flags configure Modal."""
    configure_logging()
    overrides = parse_overrides([override])
    run = prepare_config(config, resume, overrides, extends=parse_extends([extends]))
    if cpu <= 0 or memory_mb <= 0 or timeout_seconds <= 0:
        raise ValueError("Modal CPU, memory and timeout must be positive")
    datasets = modal.Volume.from_name(dataset_volume)
    outputs = modal.Volume.from_name(output_volume, create_if_missing=True)
    function = train_remote.with_options(
        gpu=gpu, cpu=cpu, memory=memory_mb, timeout=timeout_seconds,
        volumes={DATASET_MOUNT: datasets.with_mount_options(read_only=True), OUTPUT_MOUNT: outputs},
    )
    if tensorboard:
        logger.info("TensorBoard (public, persists across runs): {}", ensure_tensorboard(output_volume))
    logger.info("Starting Modal run with dataset Volume '{}' and output Volume '{}'", dataset_volume, output_volume)
    result = function.remote(run.to_toml() if run else "", resume, overrides if resume else {},
                             output_volume)
    logger.info("Run '{}' reached {}/{}; checkpoint on '{}': {}", result["name"],
                result["global_step"], result["total_steps"], output_volume, result["checkpoint"])
