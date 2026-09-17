"""Run ``pretrain_base_muon.py`` on a single Modal GPU with live TensorBoard.

The pre-tokenized dataset Volume is mounted read-only. TensorBoard events,
final model weights, and the run CSV are stored on a separate persistent
Volume and explicitly committed when the remote function exits. After a
successful run, its remote CSV row is also appended to the local run CSV.

Launch from the project root with:

    uv run --extra cloud modal run src/pavullmo/modal_pretrain_base_muon.py

Set the same environment variables accepted by ``pretrain_base_muon.py``
before the command to override its hyperparameters. For a fire-and-forget run
that should survive the local terminal closing, submit the training input with
the dedicated detached entrypoint:

    uv run --extra cloud modal run --detach \
        src/pavullmo/modal_pretrain_base_muon.py::detached
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import modal


REMOTE_SOURCE_ROOT = "/workspace/src"
DATASET_MOUNT_PATH = "/datasets"
OUTPUT_MOUNT_PATH = "/outputs"
TENSORBOARD_PORT = 6006
REMOTE_RUNS_CSV = Path(OUTPUT_MOUNT_PATH) / "pretrain_runs.csv"
LOCAL_RUNS_CSV = (
    Path(__file__).resolve().parents[2] / "tmp" / "results" / "pretrain_runs.csv"
    if modal.is_local()
    else REMOTE_RUNS_CSV
)
RUN_RESULT_FIELDS = (
    "experiment_name",
    "dataset_variant",
    "train_loss",
    "validation_loss",
    "train_script",
    "model_path",
    "config",
)

APP_NAME = "pavullmo-pretrain-base-muon"
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
    "INITIALIZATION",
    "INITIALIZATION_STD",
    "QK_NORM",
    "SPLIT_QKV_PROJECTIONS",
    "LR",
    "MIN_LR",
    "LR_SCHEDULER",
    "WSD_DECAY_FRACTION",
    "MUON_LR_MULTIPLIER",
    "ADAM_BETA1",
    "ADAM_BETA2",
    "ADAM_EPS",
    "NORM_LR_MULTIPLIER",
    "WARMUP_STEPS",
    "BATCH_SIZE",
    "GRAD_ACCUM_STEPS",
    "EPOCHS",
    "MAX_STEPS",
    "DROPOUT",
    "WEIGHT_DECAY",
    "MUON_WEIGHT_DECAY",
    "MUON_MOMENTUM",
    "MUON_NESTEROV",
    "MUON_ADJUST_LR_FN",
    "MUON_EPS",
    "MUON_NS_ITERS",
    "MUON_NS_COEFFICIENT_A",
    "MUON_NS_COEFFICIENT_B",
    "MUON_NS_COEFFICIENT_C",
    "Z_LOSS_COEFFICIENT",
    "MAX_GRAD_NORM",
    "SEED",
    "DATASET_PREFIX",
    "DATASET_VARIANT",
    "NUM_WORKERS",
    "PIN_MEMORY",
    "VALIDATION_INTERVAL",
    "VALIDATION_STEPS",
    "TENSORBOARD_FLUSH_SECS",
    "DIAGNOSTICS_INTERVAL",
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


def _read_last_run_result(csv_path: Path) -> dict[str, str]:
    with csv_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != RUN_RESULT_FIELDS:
            raise ValueError(
                f"unexpected run CSV columns in {csv_path}: {reader.fieldnames}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"run CSV has no result rows: {csv_path}")
    return {field: rows[-1][field] for field in RUN_RESULT_FIELDS}


def _append_local_run_result(csv_path: Path, run_result: dict[str, str]) -> None:
    if set(run_result) != set(RUN_RESULT_FIELDS):
        raise ValueError(f"unexpected run result fields: {sorted(run_result)}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    if not write_header:
        with csv_path.open(encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            header = tuple(next(reader, ()))
        if header != RUN_RESULT_FIELDS:
            raise ValueError(f"unexpected local run CSV columns in {csv_path}: {header}")

    with csv_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RUN_RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(run_result)


@app.function(
    gpu=MODAL_GPU,
    cpu=MODAL_CPU,
    memory=MODAL_MEMORY_MB,
    timeout=MODAL_TIMEOUT_SECONDS,
    max_containers=1,
    single_use_containers=True,
    volumes={
        DATASET_MOUNT_PATH: dataset_volume.with_mount_options(read_only=True),
        OUTPUT_MOUNT_PATH: output_volume,
    },
)
def train(pretrain_environment: dict[str, str]) -> dict[str, str]:
    import subprocess
    import sys

    os.environ.update(pretrain_environment)
    os.environ.update(
        {
            "DATASET_DIR": DATASET_MOUNT_PATH,
            "LOG_DIR": f"{OUTPUT_MOUNT_PATH}/runs",
            "RUNS_CSV": str(REMOTE_RUNS_CSV),
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
            from pavullmo.pretrain_base_muon import main as pretrain_main

            pretrain_main()
            run_result = _read_last_run_result(REMOTE_RUNS_CSV)
    finally:
        _stop_process(tensorboard_process)
        output_volume.commit()
        print(
            f"Committed training outputs to Volume {OUTPUT_VOLUME_NAME!r}",
            flush=True,
        )

    return run_result


@app.local_entrypoint()
def main() -> None:
    pretrain_environment = {
        name: os.environ[name]
        for name in PRETRAIN_ENVIRONMENT_VARIABLES
        if name in os.environ
    }
    print(
        f"Starting Modal Muon training with dataset Volume {DATASET_VOLUME_NAME!r} "
        f"and output Volume {OUTPUT_VOLUME_NAME!r}"
    )
    run_result = train.remote(pretrain_environment)
    _append_local_run_result(LOCAL_RUNS_CSV, run_result)
    print(f"Appended remote run result to local CSV {LOCAL_RUNS_CSV}")


@app.local_entrypoint()
def detached() -> None:
    pretrain_environment = {
        name: os.environ[name]
        for name in PRETRAIN_ENVIRONMENT_VARIABLES
        if name in os.environ
    }
    print(
        f"Submitting detached Modal Muon training with dataset Volume "
        f"{DATASET_VOLUME_NAME!r} and output Volume {OUTPUT_VOLUME_NAME!r}"
    )
    function_call = train.spawn(pretrain_environment)
    print(f"Submitted training FunctionCall {function_call.object_id}")
    print(
        f"Results will be written remotely to {REMOTE_RUNS_CSV}; "
        "the local CSV is not updated by a detached run."
    )
