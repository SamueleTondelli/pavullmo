"""Run ``pretrain_base.py`` on a single Modal L4 with live TensorBoard.

The pre-tokenized dataset Volume is mounted read-only. TensorBoard events,
final model weights, and the run CSV are stored on a separate persistent
Volume and explicitly committed when the remote function exits. After a
successful run, its remote CSV row is also appended to the local run CSV.

Launch from the project root with:

    uv run --extra cloud modal run src/pavullmo/modal_pretrain_base.py

Set the same environment variables accepted by ``pretrain_base.py`` before
the command to override its hyperparameters. Use Modal's ``--detach`` option
for a training run that should survive the local terminal closing.

Launch the sequential detached AdamW/batch-size sweep with:

    scripts/adamw_batch_sweep.sh
"""

from __future__ import annotations

import csv
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import modal


REMOTE_SOURCE_ROOT = "/workspace/src"
DATASET_MOUNT_PATH = "/datasets"
OUTPUT_MOUNT_PATH = "/outputs"
TENSORBOARD_PORT = 6006
REMOTE_RUNS_CSV = Path(OUTPUT_MOUNT_PATH) / "pretrain_runs.csv"
LOCAL_RUNS_CSV = Path(__file__).resolve().with_name("pretrain_runs.csv")
RUN_RESULT_FIELDS = (
    "experiment_name",
    "dataset_variant",
    "train_loss",
    "validation_loss",
    "train_script",
    "model_path",
    "config",
)

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
    "INITIALIZATION",
    "INITIALIZATION_STD",
    "QK_NORM",
    "SPLIT_QKV_PROJECTIONS",
    "LR",
    "MIN_LR",
    "LR_SCHEDULER",
    "WSD_DECAY_FRACTION",
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


def _append_local_run_result(
    csv_path: Path, run_result: dict[str, str]
) -> None:
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
            from pavullmo.pretrain_base import main as pretrain_main

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


def build_adamw_batch_sweep_configurations(
    sweep_id: str,
) -> list[dict[str, str]]:
    """Build the 20-run 195M-token AdamW and batch-size sweep."""

    base = {
        "VOCAB_SIZE": "4096",
        "N_BLOCKS": "8",
        "EMBED_DIM": "256",
        "ATTN_HEADS": "4",
        "FFN_DIM": "704",
        "SEQ_LEN": "1024",
        "ROPE_BASE": "10000.0",
        "INITIALIZATION": "gpt_scaled",
        "INITIALIZATION_STD": "0.02",
        "MIN_LR": "0.0",
        "LR_SCHEDULER": "wsd",
        "WSD_DECAY_FRACTION": "0.20",
        "ADAM_EPS": "1e-8",
        "GRAD_ACCUM_STEPS": "1",
        "EPOCHS": "1",
        "MAX_STEPS": "0",
        "DROPOUT": "0.0",
        "MAX_GRAD_NORM": "1.0",
        "SEED": "42",
        "DATASET_PREFIX": "4k",
        "DATASET_VARIANT": "195m",
        "NUM_WORKERS": "4",
        "PIN_MEMORY": "true",
        "TENSORBOARD_FLUSH_SECS": "5",
        "COMPILE_MODEL": "true",
        "COMPILE_MODE": "default",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    batch_runtime = {
        "8": {
            "WARMUP_STEPS": "732",
            "VALIDATION_INTERVAL": "500",
            "VALIDATION_STEPS": "24",
            "DIAGNOSTICS_INTERVAL": "100",
        },
        "12": {
            "WARMUP_STEPS": "488",
            "VALIDATION_INTERVAL": "333",
            "VALIDATION_STEPS": "16",
            "DIAGNOSTICS_INTERVAL": "67",
        },
        "16": {
            "WARMUP_STEPS": "366",
            "VALIDATION_INTERVAL": "250",
            "VALIDATION_STEPS": "12",
            "DIAGNOSTICS_INTERVAL": "50",
        },
    }
    candidates: list[tuple[str, str, str, str, str, str]] = []

    # Batch-8 controls around the winning 1.5e-3 peak LR.
    for label, learning_rate in (
        ("125", "0.00125"),
        ("150", "0.00150"),
        ("175", "0.00175"),
    ):
        candidates.append(
            (f"b8_lr{label}_base", "8", learning_rate, "0.9", "0.95", "0.0")
        )

    # Larger batches: compare the original beta2 with beta2^(batch / 8),
    # which keeps its exponential-memory horizon approximately fixed in tokens.
    for batch_size, learning_rates, scaled_beta2 in (
        (
            "12",
            (("175", "0.00175"), ("200", "0.00200"), ("225", "0.00225")),
            "0.925945",
        ),
        (
            "16",
            (("210", "0.00210"), ("250", "0.00250"), ("300", "0.00300")),
            "0.9025",
        ),
    ):
        for lr_label, learning_rate in learning_rates:
            candidates.append(
                (
                    f"b{batch_size}_lr{lr_label}_b2base",
                    batch_size,
                    learning_rate,
                    "0.9",
                    "0.95",
                    "0.0",
                )
            )
            candidates.append(
                (
                    f"b{batch_size}_lr{lr_label}_b2tok",
                    batch_size,
                    learning_rate,
                    "0.9",
                    scaled_beta2,
                    "0.0",
                )
            )

    # At each larger batch's center LR, also scale beta1's token horizon.
    candidates.extend(
        [
            (
                "b12_lr200_b1b2tok",
                "12",
                "0.00200",
                "0.853815",
                "0.925945",
                "0.0",
            ),
            (
                "b16_lr250_b1b2tok",
                "16",
                "0.00250",
                "0.81",
                "0.9025",
                "0.0",
            ),
        ]
    )

    # Grouped AdamW decay on the batch-8 winning base. The trainer excludes
    # the tied embedding/output matrix, RMSNorm gains, and biases from decay.
    for label, weight_decay in (
        ("001", "0.01"),
        ("005", "0.05"),
        ("010", "0.10"),
    ):
        candidates.append(
            (f"b8_lr150_wd{label}", "8", "0.00150", "0.9", "0.95", weight_decay)
        )

    configurations: list[dict[str, str]] = []
    for label, batch_size, learning_rate, beta1, beta2, weight_decay in candidates:
        experiment_name = f"10m_tok4k_195m_adambs_{sweep_id}_{label}"
        configurations.append(
            base
            | batch_runtime[batch_size]
            | {
                "EXPERIMENT_NAME": experiment_name,
                "BATCH_SIZE": batch_size,
                "LR": learning_rate,
                "ADAM_BETA1": beta1,
                "ADAM_BETA2": beta2,
                "WEIGHT_DECAY": weight_decay,
            }
        )

    if len(configurations) != 20:
        raise AssertionError(f"expected 20 sweep runs, got {len(configurations)}")
    experiment_names = {
        configuration["EXPERIMENT_NAME"] for configuration in configurations
    }
    if len(experiment_names) != len(configurations):
        raise AssertionError("sweep experiment names must be unique")
    return configurations


@app.local_entrypoint()
def adamw_batch_sweep() -> None:
    sweep_id = (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid4().hex[:6]
    )
    configurations = build_adamw_batch_sweep_configurations(sweep_id)
    print(f"Submitting {len(configurations)} sequential detached runs:")
    for index, configuration in enumerate(configurations, start=1):
        print(
            f"  {index:02d}/{len(configurations)} "
            f"{configuration['EXPERIMENT_NAME']}"
        )

    train.spawn_map(configurations)
    print("All sweep inputs were submitted to Modal.")
    print(
        f"Results will be appended remotely to {REMOTE_RUNS_CSV}; "
        "the local CSV is not updated by a detached sweep."
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
    run_result = train.remote(pretrain_environment)
    _append_local_run_result(LOCAL_RUNS_CSV, run_result)
    print(f"Appended remote run result to local CSV {LOCAL_RUNS_CSV}")
