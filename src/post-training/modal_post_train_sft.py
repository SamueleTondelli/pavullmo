"""Run PavuLLMo supervised fine-tuning on Modal with persistent Volumes."""

from __future__ import annotations

import os
from pathlib import Path

import modal


REMOTE_SOURCE_ROOT = "/workspace/src"
POST_TRAINING_SOURCE = f"{REMOTE_SOURCE_ROOT}/post-training"
INPUT_MOUNT_PATH = "/inputs"
OUTPUT_MOUNT_PATH = "/outputs"
REMOTE_DATASET_DIR = f"{INPUT_MOUNT_PATH}/dataset"
TENSORBOARD_PORT = 6006

APP_NAME = "pavullmo-post-train-sft"
INPUT_VOLUME_NAME = os.environ.get(
    "MODAL_POST_INPUT_VOLUME_NAME", "pavullmo-post-training-inputs"
)
OUTPUT_VOLUME_NAME = os.environ.get(
    "MODAL_POST_OUTPUT_VOLUME_NAME", "pavullmo-post-training-outputs"
)
BASE_CHECKPOINT_NAME = os.environ.get(
    "BASE_CHECKPOINT_NAME", "35m_tok16k_tuned_4b.pt.zip"
)
if Path(BASE_CHECKPOINT_NAME).name != BASE_CHECKPOINT_NAME:
    raise ValueError("BASE_CHECKPOINT_NAME must be a filename without directories")

MODAL_GPU = os.environ.get("MODAL_GPU", "L4")
MODAL_CPU = float(os.environ.get("MODAL_CPU", "4"))
MODAL_MEMORY_MB = int(os.environ.get("MODAL_MEMORY_MB", "32768"))
MODAL_TIMEOUT_SECONDS = int(
    os.environ.get("MODAL_TIMEOUT_SECONDS", str(24 * 60 * 60))
)

SFT_ENVIRONMENT_VARIABLES = (
    "EXPERIMENT_NAME",
    "LR",
    "MIN_LR",
    "ADAM_BETA1",
    "ADAM_BETA2",
    "ADAM_EPS",
    "WEIGHT_DECAY",
    "WARMUP_STEPS",
    "BATCH_SIZE",
    "GRAD_ACCUM_STEPS",
    "EPOCHS",
    "MAX_STEPS",
    "MAX_GRAD_NORM",
    "VALIDATION_INTERVAL",
    "SEED",
    "NUM_WORKERS",
    "COMPILE_MODEL",
    "COMPILE_MODE",
    "EVALUATE_TEST",
    "TENSORBOARD_FLUSH_SECS",
)

image = modal.Image.debian_slim(python_version="3.12")
if modal.is_local():
    project_root = Path(__file__).resolve().parents[2]
    image = (
        image.uv_sync(str(project_root), extras=["cloud"])
        .env({"PYTHONPATH": REMOTE_SOURCE_ROOT})
        .add_local_dir(project_root / "src", remote_path=REMOTE_SOURCE_ROOT)
    )

input_volume = modal.Volume.from_name(INPUT_VOLUME_NAME)
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
            with socket.create_connection(
                ("127.0.0.1", TENSORBOARD_PORT), timeout=0.5
            ):
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
    max_containers=1,
    single_use_containers=True,
    volumes={
        INPUT_MOUNT_PATH: input_volume.with_mount_options(read_only=True),
        OUTPUT_MOUNT_PATH: output_volume,
    },
)
def train(sft_environment: dict[str, str]) -> None:
    import subprocess
    import sys

    os.environ.update(sft_environment)
    os.environ.update(
        {
            "BASE_CHECKPOINT": (
                f"{INPUT_MOUNT_PATH}/checkpoints/{BASE_CHECKPOINT_NAME}"
            ),
            "POST_DATASET_DIR": REMOTE_DATASET_DIR,
            "LOG_DIR": f"{OUTPUT_MOUNT_PATH}/runs",
            "MODEL_OUTPUT_DIR": f"{OUTPUT_MOUNT_PATH}/models",
            "RUNS_CSV": f"{OUTPUT_MOUNT_PATH}/post_training_runs.csv",
            "DEVICE": "cuda",
            "MPLCONFIGDIR": "/tmp/matplotlib",
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
                "The TensorBoard tunnel is public and remains active only for "
                "this SFT run.",
                flush=True,
            )
            sys.path.insert(0, POST_TRAINING_SOURCE)
            from post_train_sft import main as sft_main

            sft_main()
    finally:
        _stop_process(tensorboard_process)
        output_volume.commit()
        print(
            f"Committed SFT outputs to Volume {OUTPUT_VOLUME_NAME!r}",
            flush=True,
        )


@app.local_entrypoint()
def main() -> None:
    sft_environment = {
        name: os.environ[name]
        for name in SFT_ENVIRONMENT_VARIABLES
        if name in os.environ
    }
    print(
        f"Starting Modal SFT with input Volume {INPUT_VOLUME_NAME!r}, output "
        f"Volume {OUTPUT_VOLUME_NAME!r}, and checkpoint {BASE_CHECKPOINT_NAME!r}"
    )
    train.remote(sft_environment)
