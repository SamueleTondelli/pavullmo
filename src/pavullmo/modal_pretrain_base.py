"""Run ``pretrain_base.py`` on a single Modal L4 with live TensorBoard.

The pre-tokenized dataset Volume is mounted read-only. TensorBoard events,
final model weights, and the run CSV are stored on a separate persistent
Volume and explicitly committed when the remote function exits.

Launch from the project root with:

    uv run --extra cloud modal run src/pavullmo/modal_pretrain_base.py

Set the same environment variables accepted by ``pretrain_base.py`` before
the command to override its hyperparameters. Use Modal's ``--detach`` option
for a training run that should survive the local terminal closing.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal


REMOTE_SOURCE_ROOT = "/workspace/src"
DATASET_MOUNT_PATH = "/datasets"
OUTPUT_MOUNT_PATH = "/outputs"
TENSORBOARD_PORT = 6006

APP_NAME = "pavullmo-pretrain-base"
DATASET_VOLUME_NAME = os.environ.get(
    "MODAL_DATASET_VOLUME_NAME", "pavullmo-datasets"
)
OUTPUT_VOLUME_NAME = os.environ.get(
    "MODAL_OUTPUT_VOLUME_NAME", "pavullmo-training"
)
MODAL_GPU = os.environ.get("MODAL_GPU", "L4")
MODAL_CPU = float(os.environ.get("MODAL_CPU", "4"))
MODAL_MEMORY_MB = int(os.environ.get("MODAL_MEMORY_MB", "32768"))
MODAL_TIMEOUT_SECONDS = int(
    os.environ.get("MODAL_TIMEOUT_SECONDS", str(24 * 60 * 60))
)

PRETRAIN_ENVIRONMENT_VARIABLES = (
    "EXPERIMENT_NAME",
    "VOCAB_SIZE",
    "N_BLOCKS",
    "EMBED_DIM",
    "ATTN_HEADS",
    "FFN_DIM",
    "SEQ_LEN",
    "ROPE_BASE",
    "LR",
    "MIN_LR",
    "ADAM_BETA1",
    "ADAM_BETA2",
    "ADAM_EPS",
    "WARMUP_STEPS",
    "BATCH_SIZE",
    "GRAD_ACCUM_STEPS",
    "EPOCHS",
    "MAX_STEPS",
    "DROPOUT",
    "WEIGHT_DECAY",
    "MAX_GRAD_NORM",
    "SEED",
    "DATASET_PREFIX",
    "DATASET_VARIANT",
    "NUM_WORKERS",
    "PIN_MEMORY",
    "VALIDATION_INTERVAL",
    "VALIDATION_STEPS",
    "TENSORBOARD_FLUSH_SECS",
    "COMPILE_MODEL",
    "COMPILE_MODE",
)

image = modal.Image.debian_slim(python_version="3.12")
if modal.is_local():
    project_root = Path(__file__).resolve().parents[2]
    image = (
        image.uv_sync(str(project_root), extras=["cloud"])
        .env({"PYTHONPATH": REMOTE_SOURCE_ROOT})
        .add_local_dir(project_root / "src", remote_path=REMOTE_SOURCE_ROOT)
    )

dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)

app = modal.App(APP_NAME, image=image)


def _wait_for_tensorboard(process: object, timeout_seconds: float = 30.0) -> None:
    import socket
    import time

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"TensorBoard exited during startup with code {process.returncode}"
            )
        try:
            with socket.create_connection(("127.0.0.1", TENSORBOARD_PORT), timeout=0.5):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"TensorBoard did not open port {TENSORBOARD_PORT} in time")


def _stop_process(process: object) -> None:
    import subprocess

    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


@app.function(
    gpu=MODAL_GPU,
    cpu=MODAL_CPU,
    memory=MODAL_MEMORY_MB,
    timeout=MODAL_TIMEOUT_SECONDS,
    volumes={
        DATASET_MOUNT_PATH: dataset_volume.with_mount_options(read_only=True),
        OUTPUT_MOUNT_PATH: output_volume,
    },
)
def train(pretrain_environment: dict[str, str]) -> None:
    import subprocess
    import sys

    os.environ.update(pretrain_environment)
    os.environ.update(
        {
            "DATASET_DIR": DATASET_MOUNT_PATH,
            "LOG_DIR": f"{OUTPUT_MOUNT_PATH}/runs",
            "RUNS_CSV": f"{OUTPUT_MOUNT_PATH}/pretrain_runs.csv",
            "MODEL_OUTPUT_DIR": f"{OUTPUT_MOUNT_PATH}/models",
            "TRAIN_SCRIPT": Path(__file__).name,
        }
    )

    log_dir = Path(os.environ["LOG_DIR"])
    log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            str(log_dir),
            "--host",
            "0.0.0.0",
            "--port",
            str(TENSORBOARD_PORT),
            "--reload_interval",
            "5",
        ]
    )

    try:
        _wait_for_tensorboard(tensorboard_process)
        with modal.forward(TENSORBOARD_PORT) as tunnel:
            print(f"Live TensorBoard: {tunnel.url}", flush=True)
            print(
                "The TensorBoard tunnel is public and remains active for this "
                "training run.",
                flush=True,
            )
            from pavullmo.pretrain_base import main as pretrain_main

            pretrain_main()
    finally:
        _stop_process(tensorboard_process)
        output_volume.commit()
        print(
            f"Committed training outputs to Volume {OUTPUT_VOLUME_NAME!r}",
            flush=True,
        )


@app.local_entrypoint()
def main() -> None:
    pretrain_environment = {
        name: os.environ[name]
        for name in PRETRAIN_ENVIRONMENT_VARIABLES
        if name in os.environ
    }
    print(
        f"Starting Modal training with dataset Volume {DATASET_VOLUME_NAME!r} "
        f"and output Volume {OUTPUT_VOLUME_NAME!r}"
    )
    train.remote(pretrain_environment)
