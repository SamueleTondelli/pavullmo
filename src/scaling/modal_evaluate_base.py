"""Run ``evaluate_base.py`` on Modal and mirror its result locally.

The validation datasets are read from the dataset Volume and checkpoints are
read from the training output Volume. The evaluator appends its result to the
remote ``scaling_loss.csv``; after a successful run, the same row is appended
to a local CSV (``scaling_loss.csv`` beside this file by default).

Launch from the project root with a checkpoint already stored below
``/outputs/models`` on the output Volume:

    uv run --extra cloud modal run src/pavullmo/modal_evaluate_base.py \
        --model my_experiment.pt

``--model`` may also be written as ``models/my_experiment.pt`` or as an
absolute ``/outputs/models/my_experiment.pt`` path. Use ``--csv`` to choose a
different local result CSV. Evaluation settings accepted by
``evaluate_base.py`` can be overridden with environment variables.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path, PurePosixPath

import modal


REMOTE_SOURCE_ROOT = "/workspace/src"
DATASET_MOUNT_PATH = "/datasets"
OUTPUT_MOUNT_PATH = "/outputs"
REMOTE_MODELS_DIR = PurePosixPath(OUTPUT_MOUNT_PATH) / "models"
REMOTE_EVALUATIONS_CSV = Path(OUTPUT_MOUNT_PATH) / "scaling_loss.csv"
LOCAL_EVALUATIONS_CSV = Path(__file__).resolve().with_name("scaling_loss.csv")
EVALUATION_RESULT_FIELDS = (
    "tokenizer",
    "parameters",
    "train_tokens",
    "validation_loss",
)

APP_NAME = "pavullmo-evaluate-base"
DATASET_VOLUME_NAME = os.environ.get("MODAL_DATASET_VOLUME_NAME", "pavullmo-datasets")
OUTPUT_VOLUME_NAME = os.environ.get("MODAL_OUTPUT_VOLUME_NAME", "pavullmo-training")
MODAL_GPU = os.environ.get("MODAL_GPU", "L4")
MODAL_CPU = float(os.environ.get("MODAL_CPU", "4"))
MODAL_MEMORY_MB = int(os.environ.get("MODAL_MEMORY_MB", "32768"))
MODAL_TIMEOUT_SECONDS = int(os.environ.get("MODAL_TIMEOUT_SECONDS", str(24 * 60 * 60)))

EVALUATION_ENVIRONMENT_VARIABLES = (
    "BATCH_SIZE",
    "SEED",
    "NUM_WORKERS",
    "PIN_MEMORY",
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
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME)

app = modal.App(APP_NAME, image=image)


def _remote_model_path(model: str) -> str:
    """Resolve a checkpoint name to a path inside the output Volume."""
    model_path = PurePosixPath(model)
    if not model or model_path.name in {"", ".", ".."}:
        raise ValueError("model must name a checkpoint")
    if ".." in model_path.parts:
        raise ValueError("model must not contain '..'")

    output_root = PurePosixPath(OUTPUT_MOUNT_PATH)
    if model_path.is_absolute():
        try:
            model_path.relative_to(output_root)
        except ValueError as error:
            raise ValueError(
                f"absolute model path must be below {OUTPUT_MOUNT_PATH}"
            ) from error
        return str(model_path)

    if model_path.parts[0] == "outputs":
        return str(PurePosixPath("/") / model_path)
    if model_path.parts[0] == "models":
        return str(output_root / model_path)
    return str(REMOTE_MODELS_DIR / model_path)


def _read_last_evaluation_result(csv_path: Path) -> dict[str, str]:
    with csv_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != EVALUATION_RESULT_FIELDS:
            raise ValueError(
                f"unexpected evaluation CSV columns in {csv_path}: {reader.fieldnames}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"evaluation CSV has no result rows: {csv_path}")
    return {field: rows[-1][field] for field in EVALUATION_RESULT_FIELDS}


def _validate_evaluation_csv(csv_path: Path) -> None:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return
    with csv_path.open(encoding="utf-8", newline="") as file:
        header = tuple(next(csv.reader(file), ()))
    if header != EVALUATION_RESULT_FIELDS:
        raise ValueError(
            f"unexpected local evaluation CSV columns in {csv_path}: {header}"
        )


def _append_local_evaluation_result(
    csv_path: Path, evaluation_result: dict[str, str]
) -> None:
    if set(evaluation_result) != set(EVALUATION_RESULT_FIELDS):
        raise ValueError(
            f"unexpected evaluation result fields: {sorted(evaluation_result)}"
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _validate_evaluation_csv(csv_path)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=EVALUATION_RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(evaluation_result)


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
def evaluate(model: str, evaluation_environment: dict[str, str]) -> dict[str, str]:
    import subprocess
    import sys

    remote_model = _remote_model_path(model)
    os.environ.update(evaluation_environment)
    os.environ["DATASET_DIR"] = DATASET_MOUNT_PATH

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scaling.evaluate_base",
                "--model",
                remote_model,
                "--csv",
                str(REMOTE_EVALUATIONS_CSV),
            ],
            check=True,
        )
        evaluation_result = _read_last_evaluation_result(REMOTE_EVALUATIONS_CSV)
    finally:
        output_volume.commit()
        print(
            f"Committed evaluation outputs to Volume {OUTPUT_VOLUME_NAME!r}",
            flush=True,
        )

    return evaluation_result


@app.local_entrypoint()
def main(model: str, csv: str = str(LOCAL_EVALUATIONS_CSV)) -> None:
    local_csv = Path(csv).expanduser()
    _validate_evaluation_csv(local_csv)
    evaluation_environment = {
        name: os.environ[name]
        for name in EVALUATION_ENVIRONMENT_VARIABLES
        if name in os.environ
    }

    print(
        f"Starting Modal evaluation of {_remote_model_path(model)} with dataset "
        f"Volume {DATASET_VOLUME_NAME!r} and output Volume {OUTPUT_VOLUME_NAME!r}"
    )
    evaluation_result = evaluate.remote(model, evaluation_environment)
    _append_local_evaluation_result(local_csv, evaluation_result)
    print(f"Appended remote evaluation result to local CSV {local_csv}")
