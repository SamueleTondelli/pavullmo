"""Reconstruct evaluation models from checkpoint metadata and saved weights."""

from __future__ import annotations

from pathlib import Path

import torch

from model import DecoderTransformer


ARCHITECTURE_KEYS = (
    "VOCAB_SIZE",
    "N_BLOCKS",
    "EMBED_DIM",
    "ATTN_HEADS",
    "FFN_DIM",
    "SEQ_LEN",
    "ROPE_BASE",
    "DROPOUT",
)


def _num_kv_heads(
    hyperparameters: dict[str, object],
    state_dict: dict[str, torch.Tensor],
    split_qkv_projections: bool,
) -> int:
    recorded = hyperparameters.get("NUM_KV_HEADS")
    if recorded is not None:
        return int(recorded)

    # Older checkpoints omit NUM_KV_HEADS. Projection widths distinguish GQA
    # from MHA for both parameter layouts without assuming full-width K/V.
    embed_dim = int(hyperparameters["EMBED_DIM"])
    attn_heads = int(hyperparameters["ATTN_HEADS"])
    if attn_heads <= 0 or embed_dim <= 0 or embed_dim % attn_heads:
        raise ValueError("EMBED_DIM must be positive and divisible by ATTN_HEADS")
    if int(hyperparameters["N_BLOCKS"]) == 0:
        return attn_heads
    if split_qkv_projections:
        kv_dimension = state_dict["layers.0.attn.k_proj.weight"].shape[0]
    else:
        kv_rows = state_dict["layers.0.attn.c_attn.weight"].shape[0] - embed_dim
        if kv_rows % 2:
            raise ValueError("fused QKV projection must have equal K and V widths")
        kv_dimension = kv_rows // 2

    head_dim = embed_dim // attn_heads
    num_kv_heads, remainder = divmod(kv_dimension, head_dim)
    if remainder or num_kv_heads <= 0 or attn_heads % num_kv_heads:
        raise ValueError(
            "checkpoint K/V projection width is incompatible with ATTN_HEADS"
        )
    return num_kv_heads


def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[DecoderTransformer, dict[str, object]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")

    hyperparameters = checkpoint.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        raise ValueError("checkpoint does not contain a hyperparameters mapping")
    missing = [key for key in ARCHITECTURE_KEYS if key not in hyperparameters]
    if missing:
        raise ValueError(
            "checkpoint is missing architecture settings: " + ", ".join(missing)
        )

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint does not contain a model_state_dict")
    split_qkv_projections = bool(
        hyperparameters.get(
            "SPLIT_QKV_PROJECTIONS",
            any(".attn.q_proj." in name for name in state_dict),
        )
    )
    canon_layers = bool(
        hyperparameters.get(
            "CANON_LAYERS",
            any(".ca.weight" in name for name in state_dict),
        )
    )
    qk_norm = bool(
        hyperparameters.get(
            "QK_NORM",
            any(".attn.q_norm.weight" in name for name in state_dict),
        )
    )

    model = DecoderTransformer(
        vocab_size=int(hyperparameters["VOCAB_SIZE"]),
        n_blocks=int(hyperparameters["N_BLOCKS"]),
        embed_dim=int(hyperparameters["EMBED_DIM"]),
        attn_heads=int(hyperparameters["ATTN_HEADS"]),
        num_kv_heads=_num_kv_heads(hyperparameters, state_dict, split_qkv_projections),
        ffn_dim=int(hyperparameters["FFN_DIM"]),
        dropout=float(hyperparameters["DROPOUT"]),
        seq_len=int(hyperparameters["SEQ_LEN"]),
        rope_base=float(hyperparameters["ROPE_BASE"]),
        qk_norm=qk_norm,
        split_qkv_projections=split_qkv_projections,
        canon_layers=canon_layers,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, hyperparameters
