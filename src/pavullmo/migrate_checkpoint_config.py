"""Write a generation-only TOML model config beside a legacy checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from components.checkpoint import model_state_from_checkpoint
from config import ModelConfig
from config.serialization import save_toml
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
        if name == "num_kv_heads" and value is None:
            continue  # Legacy metadata uses None when K/V heads were omitted.
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
    config = ModelConfig.from_dict(settings)
    if "num_kv_heads" not in settings:
        if config.split_qkv_projections:
            projection = state.get("layers.0.attn.k_proj.weight")
            if projection is None:
                raise ValueError("checkpoint is missing its key projection")
            kv_width = projection.shape[0]
        else:
            projection = state.get("layers.0.attn.c_attn.weight")
            if projection is None:
                raise ValueError("checkpoint is missing its QKV projection")
            kv_rows = projection.shape[0] - config.embed_dim
            if kv_rows <= 0 or kv_rows % 2:
                raise ValueError("fused QKV projection must have equal K and V widths")
            kv_width = kv_rows // 2
        head_dim = config.embed_dim // config.attn_heads
        kv_heads, remainder = divmod(kv_width, head_dim)
        if remainder or kv_heads <= 0 or config.attn_heads % kv_heads:
            raise ValueError("checkpoint K/V width is incompatible with model.attn_heads")
        config = config.with_overrides({
            "num_kv_heads": 0 if kv_heads == config.attn_heads else kv_heads,
        })
    return config


def migrate_checkpoint_config(checkpoint: str | Path, *, force: bool = False) -> Path:
    """Validate legacy weights and publish config.toml without changing the .pt."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"model checkpoint not found: {checkpoint}")
    destination = checkpoint.parent / "config.toml"
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
    del payload

    # Check names and shapes before publishing. Meta parameters avoid allocating
    # a second copy of large weights; assign loads the actual CPU tensors.
    with torch.device("meta"):
        model = DecoderTransformer(config)
    model.load_state_dict(state, strict=True, assign=True)
    save_toml({"model": config.to_dict()}, destination)
    logger.info("Migrated {} to {}; this config supports generation only", checkpoint, destination)
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="legacy .pt checkpoint")
    parser.add_argument("--force", action="store_true", help="replace an existing config.toml")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    migrate_checkpoint_config(args.checkpoint, force=args.force)


if __name__ == "__main__":
    main()
