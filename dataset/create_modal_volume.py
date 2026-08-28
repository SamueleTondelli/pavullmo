"""Create a Modal Volume and upload the pre-tokenized datasets once.

Run this script locally after ``dataset/build_dataset.py`` has finished:

    uv run --extra cloud python dataset/create_modal_volume.py

The Volume root contains every tokenizer-specific artifact so it can later be
mounted directly at the training script's ``DATASET_DIR``. Existing remote
files are not overwritten unless ``--force`` is passed explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


DEFAULT_VOLUME_NAME = "pavullmo-datasets"
DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "ds"
REMOTE_DATASET_DIR = "/"
TOKENIZER_PREFIXES = ("4k", "8k", "16k")
DATASET_VARIANTS = ("195m",)
EXPECTED_ARTIFACTS = tuple(
    [
        f"train_{prefix}_{variant}"
        for prefix in TOKENIZER_PREFIXES
        for variant in DATASET_VARIANTS
    ]
    + [f"validation_{prefix}" for prefix in TOKENIZER_PREFIXES]
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a named Modal Volume if needed and upload the local "
            "pre-tokenized dataset artifacts to its root."
        )
    )
    parser.add_argument(
        "--volume-name",
        default=DEFAULT_VOLUME_NAME,
        help=f"Modal Volume name (default: {DEFAULT_VOLUME_NAME})",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"local artifact directory (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--environment",
        help=(
            "Modal environment name; by default Modal uses MODAL_ENVIRONMENT, "
            "the active profile, or the workspace default"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite remote files that already exist",
    )
    return parser.parse_args(argv)


def validate_artifact(artifact_dir: Path) -> tuple[int, int]:
    """Validate one artifact and return its token and byte counts."""

    metadata_path = artifact_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"artifact metadata not found: {metadata_path}")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {metadata_path}: {error}") from error

    token_count = metadata.get("token_count")
    if not isinstance(token_count, int) or isinstance(token_count, bool):
        raise ValueError(f"invalid token_count in {metadata_path}")
    if token_count <= 0:
        raise ValueError(f"token_count must be positive in {metadata_path}")

    shards = metadata.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"missing shard list in {metadata_path}")

    declared_shards: set[str] = set()
    total_bytes = 0
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError(f"invalid shard entry in {metadata_path}")

        filename = shard.get("file")
        byte_count = shard.get("bytes")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
        ):
            raise ValueError(f"invalid shard filename in {metadata_path}: {filename!r}")
        if filename in declared_shards:
            raise ValueError(f"duplicate shard {filename!r} in {metadata_path}")
        if not isinstance(byte_count, int) or isinstance(byte_count, bool):
            raise ValueError(f"invalid byte count for {filename!r} in {metadata_path}")

        shard_path = artifact_dir / filename
        if not shard_path.is_file():
            raise FileNotFoundError(f"dataset shard not found: {shard_path}")
        actual_bytes = shard_path.stat().st_size
        if actual_bytes != byte_count:
            raise ValueError(
                f"size mismatch for {shard_path}: metadata says {byte_count:,} "
                f"bytes, found {actual_bytes:,}"
            )

        declared_shards.add(filename)
        total_bytes += byte_count

    actual_shards = {path.name for path in artifact_dir.glob("tokens-*.bin")}
    if actual_shards != declared_shards:
        missing = sorted(declared_shards - actual_shards)
        extra = sorted(actual_shards - declared_shards)
        raise ValueError(
            f"shard inventory mismatch in {artifact_dir}: "
            f"missing={missing}, extra={extra}"
        )
    if total_bytes != token_count * 2:
        raise ValueError(
            f"byte total mismatch in {metadata_path}: {token_count:,} uint16 "
            f"tokens require {token_count * 2:,} bytes, metadata lists "
            f"{total_bytes:,}"
        )

    return token_count, total_bytes


def validate_dataset_dir(dataset_dir: Path) -> tuple[int, int, int]:
    """Ensure all expected artifacts are complete before starting an upload."""

    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"dataset directory not found: {dataset_dir}; run "
            "dataset/build_dataset.py first"
        )

    total_tokens = 0
    total_bytes = 0
    total_files = 0
    for artifact_name in EXPECTED_ARTIFACTS:
        artifact_dir = dataset_dir / artifact_name
        if not artifact_dir.is_dir():
            raise FileNotFoundError(f"dataset artifact not found: {artifact_dir}")
        token_count, byte_count = validate_artifact(artifact_dir)
        total_tokens += token_count
        total_bytes += byte_count
        total_files += 1 + len(list(artifact_dir.glob("tokens-*.bin")))

    return total_tokens, total_bytes, total_files


def upload_dataset(
    dataset_dir: Path,
    volume_name: str,
    environment: str | None,
    force: bool,
) -> None:
    try:
        import modal
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Modal is not installed; run this command with `uv run --extra cloud`"
        ) from error

    volume = modal.Volume.from_name(
        volume_name,
        environment_name=environment,
        create_if_missing=True,
    )
    with volume.batch_upload(force=force) as upload:
        for artifact_name in EXPECTED_ARTIFACTS:
            upload.put_directory(
                dataset_dir / artifact_name,
                f"{REMOTE_DATASET_DIR}{artifact_name}",
            )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.volume_name.strip():
        raise ValueError("--volume-name cannot be empty")

    dataset_dir = args.dataset_dir.expanduser().resolve()
    total_tokens, total_bytes, total_files = validate_dataset_dir(dataset_dir)
    print(
        f"Validated {total_files} files in {dataset_dir}: "
        f"{total_tokens:,} tokens, {total_bytes / (1024**3):.2f} GiB"
    )
    print(f"Uploading to Modal Volume {args.volume_name!r} at {REMOTE_DATASET_DIR}...")
    upload_dataset(
        dataset_dir=dataset_dir,
        volume_name=args.volume_name,
        environment=args.environment,
        force=args.force,
    )
    print(f"Modal Volume {args.volume_name!r} is ready")


if __name__ == "__main__":
    main()
