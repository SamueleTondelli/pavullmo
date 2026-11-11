"""Migrate a legacy run to config.toml, checkpoints/ and tensorboard/."""

from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from components.checkpoint import FORMAT_VERSION, model_state_from_checkpoint
from config import ModelConfig
from config.serialization import dump_toml, save_toml
from model import DecoderTransformer


def legacy_model_config(
    hyperparameters: Mapping[str, Any], state: dict[str, torch.Tensor],
) -> ModelConfig:
    """Translate recorded settings; infer missing layout flags from weight keys."""
    settings = {}
    for name, field in ModelConfig.model_fields.items():
        key = name.upper()
        if key not in hyperparameters:
            continue
        value = hyperparameters[key]
        if isinstance(value, str):
            value = value.strip()
            if field.annotation is bool:
                booleans = {
                    "true": True, "yes": True, "on": True, "1": True,
                    "false": False, "no": False, "off": False, "0": False,
                }
                if value.lower() not in booleans:
                    raise ValueError(f"invalid boolean for {key}: {value!r}")
                value = booleans[value.lower()]
            elif field.annotation is int:
                value = int(value)
            elif field.annotation is float:
                value = float(value)
        elif field.annotation is bool and type(value) is int and value in (0, 1):
            value = bool(value)
        settings[name] = value

    # These settings affect inference and were recorded by the old trainers.
    # In particular, heads/context/RoPE cannot be recovered safely from shapes.
    required = ("vocab_size", "n_blocks", "embed_dim", "attn_heads", "ffn_dim",
                "seq_len", "rope_base")
    missing = [name.upper() for name in required if name not in settings]
    if missing:
        raise ValueError("checkpoint is missing architecture settings: " + ", ".join(missing))

    inferred: dict[str, bool | int] = {
        "split_qkv_projections": any(".attn.q_proj.weight" in name for name in state),
        "qk_norm": any(".attn.q_norm.weight" in name for name in state),
        "canon_layers": any(".ca.weight" in name for name in state),
        "attn_bias": any(name.endswith((".attn.c_attn.bias", ".attn.q_proj.bias",
                                       ".attn.c_proj.bias")) for name in state),
    }
    kernels = {value.shape[-1] for name, value in state.items()
               if name.endswith((".ca.weight", ".attn.cb.weight", ".cc.weight", ".ffn.cd.weight"))}
    if kernels:
        if len(kernels) != 1:
            raise ValueError("checkpoint Canon layers have inconsistent kernel sizes")
        inferred["canon_kernel_size"] = kernels.pop()
    for name, value in inferred.items():
        if name not in settings:
            settings[name] = value
            logger.debug("Recovered model.{}={} from checkpoint weights", name, value)
    return ModelConfig.from_dict(settings)


def migrate_checkpoint_config(
    run_dir: str | Path, *, models_dir: str | Path | None = None, force: bool = False,
) -> Path:
    """Publish a weights-only save and relocate events, preserving the source .pt."""
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f"run directory not found: {run_dir}")
    models_dir = (run_dir.parent.parent / "models" if models_dir is None
                  else Path(models_dir).expanduser().resolve())
    checkpoint = models_dir / f"{run_dir.name}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"model checkpoint not found: {checkpoint}")
    destination = run_dir / "config.toml"
    if destination.exists() and not force:
        raise FileExistsError(f"config already exists: {destination}; use --force to replace it")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "trainer_state" in payload:
        raise ValueError("expected a legacy checkpoint with hyperparameters and model_state_dict")
    hyperparameters = payload.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        raise ValueError("checkpoint does not contain a hyperparameters mapping")
    state = model_state_from_checkpoint(payload)
    config = legacy_model_config(hyperparameters, state)
    step = payload.get("global_step")
    if type(step) is not int or step < 0:
        raise ValueError("legacy checkpoint must contain a non-negative integer global_step")
    del payload

    # Check names and shapes before publishing. Meta parameters avoid allocating
    # a second copy of large weights; assign loads the actual CPU tensors.
    with torch.device("meta"):
        model = DecoderTransformer(config)
    model.load_state_dict(state, strict=True, assign=True)

    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    migrated_checkpoint = checkpoint_dir / f"step_{step:08d}.pt"
    for directory in (checkpoint_dir, tensorboard_dir):
        if directory.exists() and not directory.is_dir():
            raise NotADirectoryError(f"expected a directory: {directory}")
    if migrated_checkpoint.exists() and not force:
        raise FileExistsError(f"checkpoint already exists: {migrated_checkpoint}; use --force to replace it")
    if destination.is_dir() or migrated_checkpoint.is_dir():
        raise IsADirectoryError("config and checkpoint destinations must be files")
    events = sorted(path for path in run_dir.glob("events.out.tfevents.*") if path.is_file())
    for event in events:
        target = tensorboard_dir / event.name
        if target.exists():
            raise FileExistsError(f"TensorBoard event already exists: {target}")

    # Legacy saves have no optimizer/cursor/RNG state for a full resume. Use the
    # current weights-only envelope with the same partial config as the sidecar.
    data = {"model": config.to_dict()}
    converted = {
        "format_version": FORMAT_VERSION,
        "config_toml": dump_toml(data),
        "source_checkpoint": str(checkpoint),
        "full_state": False,
        "trainer_state": {"model": state},
    }
    checkpoint_dir.mkdir(exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=checkpoint_dir, prefix=f".{migrated_checkpoint.name}.",
            suffix=".tmp", delete=False,
        ) as file:
            temporary = Path(file.name)
            torch.save(converted, file)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(migrated_checkpoint)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    save_toml(data, destination)
    tensorboard_dir.mkdir(exist_ok=True)
    for event in events:
        event.rename(tensorboard_dir / event.name)
    logger.info("Migrated {} to {}; supports generation/evaluation, not training resume",
                checkpoint, run_dir)
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="legacy run directory containing TensorBoard events")
    parser.add_argument(
        "--models-dir", type=Path,
        help="legacy weights directory (default: <run_dir>/../../models); selects <run_name>.pt",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing config.toml and migrated checkpoint")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    migrate_checkpoint_config(args.run_dir, models_dir=args.models_dir, force=args.force)


if __name__ == "__main__":
    main()
