"""Validate and upload SFT inputs to a persistent Modal Volume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_CHECKPOINT = PROJECT_DIR.parent / "35m_tok16k_tuned_4b.pt.zip"
DEFAULT_DATASET_DIR = SCRIPT_DIR / "dataset"
DEFAULT_VOLUME_NAME = "pavullmo-post-training-inputs"
REMOTE_CHECKPOINT_DIR = "/checkpoints"
REMOTE_DATASET_DIR = "/dataset"
SPLITS = ("train", "validation", "test")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and upload a base checkpoint and built SFT dataset."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--volume-name", default=DEFAULT_VOLUME_NAME)
    parser.add_argument("--environment")
    parser.add_argument(
        "--component",
        choices=("all", "checkpoint", "dataset"),
        default="all",
        help="upload everything, or only one component",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite files already present in the Volume",
    )
    return parser.parse_args(argv)


def load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"base checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("base checkpoint must contain a dictionary")
    hyperparameters = checkpoint.get("hyperparameters")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(hyperparameters, dict) or not isinstance(state_dict, dict):
        raise ValueError("base checkpoint lacks hyperparameters or model_state_dict")
    required = {
        "VOCAB_SIZE",
        "N_BLOCKS",
        "EMBED_DIM",
        "ATTN_HEADS",
        "FFN_DIM",
        "SEQ_LEN",
    }
    missing = sorted(required - hyperparameters.keys())
    if missing:
        raise ValueError("base checkpoint missing: " + ", ".join(missing))
    if not state_dict:
        raise ValueError("base checkpoint model_state_dict is empty")
    return {
        "experiment_name": checkpoint.get("experiment_name"),
        "vocab_size": int(hyperparameters["VOCAB_SIZE"]),
        "context_length": int(hyperparameters["SEQ_LEN"]),
        "parameters_tensors": len(state_dict),
        "bytes": path.stat().st_size,
    }


def validate_dataset(dataset_dir: Path) -> dict[str, Any]:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"SFT metadata not found: {metadata_path}; run build_post_dataset.py first"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1:
        raise ValueError("unsupported SFT metadata format")

    summaries: dict[str, Any] = {}
    for split in SPLITS:
        artifact_path = dataset_dir / f"{split}.pt"
        if not artifact_path.is_file():
            raise FileNotFoundError(f"missing SFT artifact: {artifact_path}")
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
        if artifact.get("format_version") != 1 or artifact.get("split") != split:
            raise ValueError(f"invalid {split} artifact header")
        examples = artifact.get("examples")
        if not isinstance(examples, list) or not examples:
            raise ValueError(f"{split} artifact contains no examples")

        ids: list[str] = []
        tokens = 0
        assistant_targets = 0
        for example in examples:
            example_id = example.get("id")
            token_ids = example.get("token_ids")
            loss_mask = example.get("loss_mask")
            if not isinstance(example_id, str) or not example_id:
                raise ValueError(f"{split} contains an invalid example id")
            if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 1:
                raise ValueError(f"{example_id!r} has invalid token_ids")
            if not isinstance(loss_mask, torch.Tensor) or loss_mask.shape != token_ids.shape:
                raise ValueError(f"{example_id!r} has invalid loss_mask")
            if token_ids.numel() < 2 or not loss_mask.bool().any():
                raise ValueError(f"{example_id!r} has no usable assistant targets")
            ids.append(example_id)
            tokens += token_ids.numel()
            assistant_targets += int(loss_mask.sum().item())

        declared = metadata.get("splits", {}).get(split, {})
        if ids != declared.get("ids"):
            raise ValueError(f"{split} IDs do not match metadata")
        if len(examples) != declared.get("examples") or tokens != declared.get("tokens"):
            raise ValueError(f"{split} counts do not match metadata")
        if assistant_targets != declared.get("assistant_targets"):
            raise ValueError(f"{split} assistant target count does not match metadata")
        summaries[split] = {
            "examples": len(examples),
            "tokens": tokens,
            "assistant_targets": assistant_targets,
            "bytes": artifact_path.stat().st_size,
        }

    summaries["metadata_bytes"] = metadata_path.stat().st_size
    summaries["vocab_size"] = int(metadata["vocab_size"])
    summaries["context_length"] = int(metadata["context_length"])
    return summaries


def validate_compatibility(
    checkpoint: dict[str, Any], dataset: dict[str, Any]
) -> None:
    if checkpoint["vocab_size"] != dataset["vocab_size"]:
        raise ValueError(
            "checkpoint and SFT dataset vocabulary sizes differ: "
            f"{checkpoint['vocab_size']} != {dataset['vocab_size']}"
        )
    if dataset["context_length"] > checkpoint["context_length"]:
        raise ValueError(
            "SFT dataset context length exceeds the checkpoint limit: "
            f"{dataset['context_length']} > {checkpoint['context_length']}"
        )


def upload_inputs(
    *,
    checkpoint_path: Path,
    dataset_dir: Path,
    volume_name: str,
    environment: str | None,
    component: str,
    force: bool,
) -> None:
    try:
        import modal
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Modal is not installed; run with `uv run --extra cloud`"
        ) from error

    volume = modal.Volume.from_name(
        volume_name,
        environment_name=environment,
        create_if_missing=True,
    )
    with volume.batch_upload(force=force) as upload:
        if component in {"all", "checkpoint"}:
            upload.put_file(
                checkpoint_path,
                f"{REMOTE_CHECKPOINT_DIR}/{checkpoint_path.name}",
            )
        if component in {"all", "dataset"}:
            upload.put_directory(dataset_dir, REMOTE_DATASET_DIR)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.volume_name.strip():
        raise ValueError("--volume-name cannot be empty")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    checkpoint_summary = load_checkpoint_metadata(checkpoint_path)
    dataset_summary = validate_dataset(dataset_dir)
    validate_compatibility(checkpoint_summary, dataset_summary)
    print(
        json.dumps(
            {
                "checkpoint": checkpoint_summary,
                "dataset": dataset_summary,
                "volume": args.volume_name,
                "component": args.component,
            },
            indent=2,
        )
    )
    upload_inputs(
        checkpoint_path=checkpoint_path,
        dataset_dir=dataset_dir,
        volume_name=args.volume_name,
        environment=args.environment,
        component=args.component,
        force=args.force,
    )
    print(f"Modal Volume {args.volume_name!r} is ready")


if __name__ == "__main__":
    main()
