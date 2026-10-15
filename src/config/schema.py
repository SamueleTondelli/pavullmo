"""Run settings, independent of training imports and execution providers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal, Self

from loguru import logger
from pydantic import Field, model_validator

from .base import Config


class ModelConfig(Config):
    vocab_size: int = Field(default=16000, gt=0)
    n_blocks: int = Field(default=12, gt=0)
    embed_dim: int = Field(default=512, gt=0)
    attn_heads: int = Field(default=8, gt=0)
    num_kv_heads: int = Field(default=0, ge=0, description="Zero uses attn_heads (MHA).")
    ffn_dim: int = Field(default=1536, gt=0)
    seq_len: int = Field(default=1024, gt=0)
    dropout: float = Field(default=0.0, ge=0, le=1)
    rope_base: float = Field(default=10000.0, gt=0)
    initialization: Literal["pytorch_default", "olmo", "gpt_scaled"] = "pytorch_default"
    initialization_std: float = Field(default=0.02, gt=0)
    qk_norm: bool = False
    split_qkv_projections: bool = False
    canon_layers: bool = False
    attn_bias: bool = False
    canon_kernel_size: int = Field(default=4, gt=0)

    @model_validator(mode="after")
    def validate_heads(self) -> Self:
        if self.embed_dim % self.attn_heads:
            raise ValueError("model.embed_dim must be divisible by model.attn_heads")
        if (self.embed_dim // self.attn_heads) % 2:
            raise ValueError("each attention head must have an even dimension for RoPE")
        if self.num_kv_heads and self.attn_heads % self.num_kv_heads:
            raise ValueError("model.num_kv_heads must divide model.attn_heads")
        return self


class TokenDatasetConfig(Config):
    """One token artifact, relative to runtime.dataset_dir or absolute."""

    path: str = Field(min_length=1)


class AdamWConfig(Config):
    beta1: float = Field(default=0.9, ge=0, lt=1)
    beta2: float = Field(default=0.95, ge=0, lt=1)
    eps: float = Field(default=1e-8, gt=0)
    weight_decay: float = Field(default=1e-5, ge=0)
    fused: bool = True


class MuonConfig(Config):
    lr_multiplier: float = Field(default=1.0, gt=0)
    weight_decay: float = Field(default=0.05, ge=0)
    momentum: float = Field(default=0.95, ge=0, lt=1)
    nesterov: bool = True
    adjust_lr_fn: Literal["original", "match_rms_adamw"] = "match_rms_adamw"
    eps: float = Field(default=1e-7, gt=0)
    ns_iters: int = Field(default=5, gt=0, lt=100)
    ns_coefficients: list[float] = Field(
        default_factory=lambda: [3.4445, -4.775, 2.0315],
        min_length=3,
        max_length=3,
    )


class ScheduleConfig(Config):
    kind: Literal["cosine", "wsd"] = "cosine"
    min_lr: float = Field(default=0.0, ge=0)
    warmup_steps: int = Field(default=10, ge=0)
    decay_fraction: float = Field(default=0.2, gt=0, le=1)


class OptimizerConfig(Config):
    kind: Literal["adamw", "muon_adamw"] = "adamw"
    lr: float = Field(default=1e-3, gt=0)
    norm_lr_multiplier: float = Field(default=1.0, ge=0)
    adamw: AdamWConfig = Field(default_factory=AdamWConfig)
    muon: MuonConfig = Field(default_factory=MuonConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)

    @model_validator(mode="after")
    def validate_minimum_lr(self) -> Self:
        if self.schedule.min_lr > self.lr:
            raise ValueError("optimizer.schedule.min_lr must not exceed optimizer.lr")
        return self


class TrainingConfig(Config):
    batch_size: int = Field(default=32, gt=0)
    grad_accum_steps: int = Field(default=1, gt=0)
    epochs: int = Field(default=1, gt=0)
    max_steps: int = Field(default=0, ge=0, description="Zero means no step cap.")
    max_grad_norm: float = Field(default=1.0, gt=0)
    z_loss_coefficient: float = Field(default=0.0, ge=0)
    seed: int = Field(default=42, ge=0)


class EvaluationConfig(Config):
    interval_steps: int = Field(default=20, gt=0)
    steps: int = Field(default=20, gt=0)


class RuntimeConfig(Config):
    dataset_dir: str = Field(default="artifacts/datasets", min_length=1)
    output_dir: str = Field(default="artifacts/runs", min_length=1)
    runs_csv: str = Field(default="artifacts/results/pretrain_runs.csv", min_length=1)
    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = Field(default=2, gt=0)
    compile_model: bool = True
    compile_mode: Literal[
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    ] = "default"


class LoggingConfig(Config):
    level: Literal[
        "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    ] = "INFO"
    console: bool = True
    file: str = ""
    rotation: str = "10 MB"
    retention: str = "7 days"
    enqueue: bool = False
    tensorboard_flush_secs: int = Field(default=5, gt=0)
    diagnostics_interval: int = Field(default=100, gt=0)


class CheckpointConfig(Config):
    interval_steps: int = Field(
        default=0, ge=0, description="Zero disables periodic saves."
    )
    keep_last: int = Field(
        default=2,
        gt=0,
        description="Number of most recently saved checkpoint files to retain per run, including periodic, final and graceful-stop saves.",
    )
    save_training_state: bool = True


class TrainerConfig(Config):
    """Common components; each training stage defines its own root subclass."""

    name: str
    model: ModelConfig = Field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    @model_validator(mode="after")
    def validate_name(self) -> Self:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.name):
            raise ValueError(
                "name must be filename-safe and start with a letter or digit"
            )
        return self


class PretrainingConfig(TrainerConfig):
    """Root configuration for next-token pretraining."""

    train_dataset: TokenDatasetConfig = Field(
        default_factory=lambda: TokenDatasetConfig(path="train_10m"),
    )
    validation_dataset: TokenDatasetConfig = Field(
        default_factory=lambda: TokenDatasetConfig(path="validation"),
    )
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="before")
    @classmethod
    def move_legacy_data(cls, data: Any) -> Any:
        """Load former combined data sections; explicit dataset fields win."""
        if not isinstance(data, Mapping) or not isinstance(data.get("data"), Mapping):
            return data
        legacy = dict(data["data"])
        if not any(name in legacy for name in ("train", "validation")):
            return data
        migrated = dict(data)
        for name in ("train", "validation"):
            if name not in legacy:
                continue
            path = legacy.pop(name)
            section = f"{name}_dataset"
            if section not in migrated:
                migrated[section] = {"path": path}
            elif isinstance(migrated[section], Mapping):
                migrated[section] = {"path": path, **migrated[section]}
        if legacy:
            migrated["data"] = legacy
        else:
            migrated.pop("data")
        logger.warning(
            "Moved deprecated config section 'data' to 'train_dataset'/'validation_dataset'."
        )
        return migrated

    @model_validator(mode="before")
    @classmethod
    def move_legacy_schedule(cls, data: Any) -> Any:
        """Reload pre-move TOML snapshots; explicitly nested settings win."""
        if isinstance(data, Mapping) and "schedule" in data:
            from .serialization import merge_dicts

            migrated = dict(data)
            schedule = migrated.pop("schedule")
            optimizer = migrated.get("optimizer", {})
            if isinstance(optimizer, Mapping):
                optimizer = dict(optimizer)
                if "schedule" not in optimizer:
                    optimizer["schedule"] = schedule
                elif isinstance(schedule, Mapping) and isinstance(
                    optimizer["schedule"], Mapping
                ):
                    optimizer["schedule"] = merge_dicts(schedule, optimizer["schedule"])
                migrated["optimizer"] = optimizer
                logger.warning(
                    "Moved deprecated config section 'schedule' to 'optimizer.schedule'."
                )
                return migrated
        return data

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        if (
            self.training.max_steps
            and self.optimizer.schedule.warmup_steps >= self.training.max_steps
        ):
            raise ValueError(
                "optimizer.schedule.warmup_steps must be smaller than training.max_steps"
            )
        return self
