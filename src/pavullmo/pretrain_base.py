from __future__ import annotations

import bisect
import csv
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
MODEL_DIR = SCRIPT_DIR.parent / "model"
sys.path.insert(0, str(MODEL_DIR))

from model import DecoderTransformer


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "pavullmo_base")

# Architecture hyperparameters.
VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", 16000))
N_BLOCKS = int(os.environ.get("N_BLOCKS", 12))
EMBED_DIM = int(os.environ.get("EMBED_DIM", 512))
ATTN_HEADS = int(os.environ.get("ATTN_HEADS", 8))
FFN_DIM = int(os.environ.get("FFN_DIM", 1536))
SEQ_LEN = int(os.environ.get("SEQ_LEN", 1024))
ROPE_BASE = float(os.environ.get("ROPE_BASE", 10000.0))
INITIALIZATION = os.environ.get("INITIALIZATION", "pytorch_default").strip().lower()
INITIALIZATION_STD = float(os.environ.get("INITIALIZATION_STD", 0.02))
QK_NORM = env_bool("QK_NORM", False)

# Training hyperparameters.
LR = float(os.environ.get("LR", 1e-3))
MIN_LR = float(os.environ.get("MIN_LR", 0.0))
LR_SCHEDULER = os.environ.get("LR_SCHEDULER", "cosine").strip().lower()
WSD_DECAY_FRACTION = float(os.environ.get("WSD_DECAY_FRACTION", 0.2))
ADAM_BETA1 = float(os.environ.get("ADAM_BETA1", 0.9))
ADAM_BETA2 = float(os.environ.get("ADAM_BETA2", 0.95))
ADAM_EPS = float(os.environ.get("ADAM_EPS", 1e-8))
NORM_LR_MULTIPLIER = float(os.environ.get("NORM_LR_MULTIPLIER", 1.0))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", 10))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", 1))
EPOCHS = int(os.environ.get("EPOCHS", 1))
MAX_STEPS = int(os.environ.get("MAX_STEPS", 0))
DROPOUT = float(os.environ.get("DROPOUT", 0.0))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 1e-5))
Z_LOSS_COEFFICIENT = float(os.environ.get("Z_LOSS_COEFFICIENT", 0.0))
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", 1.0))
SEED = int(os.environ.get("SEED", 42))

# Data, evaluation, and runtime settings.
DATASET_VARIANT = os.environ.get("DATASET_VARIANT", "10m").lower()
DATASET_PREFIX = os.environ.get("DATASET_PREFIX", "").strip()
if DATASET_PREFIX and not re.fullmatch(
    r"[A-Za-z0-9][A-Za-z0-9._-]*", DATASET_PREFIX
):
    raise ValueError(
        "DATASET_PREFIX must start with an ASCII letter or digit and contain "
        "only letters, digits, '.', '_', and '-'"
    )
DATASET_DIR = Path(
    os.environ.get("DATASET_DIR", str(PROJECT_DIR / "dataset" / "ds"))
).expanduser()
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 4))
PIN_MEMORY = env_bool("PIN_MEMORY", True)
VALIDATION_INTERVAL = int(os.environ.get("VALIDATION_INTERVAL", 20))
VALIDATION_STEPS = int(os.environ.get("VALIDATION_STEPS", 20))
LOG_DIR = Path(os.environ.get("LOG_DIR", str(PROJECT_DIR / "runs"))).expanduser()
RUNS_CSV = Path(
    os.environ.get("RUNS_CSV", str(SCRIPT_DIR / "pretrain_runs.csv"))
).expanduser()
MODEL_OUTPUT_DIR = Path(
    os.environ.get("MODEL_OUTPUT_DIR", str(PROJECT_DIR / "models"))
).expanduser()
TRAIN_SCRIPT = os.environ.get("TRAIN_SCRIPT", Path(__file__).name)
TENSORBOARD_FLUSH_SECS = int(os.environ.get("TENSORBOARD_FLUSH_SECS", 5))
DIAGNOSTICS_INTERVAL = int(os.environ.get("DIAGNOSTICS_INTERVAL", 100))
COMPILE_MODEL = env_bool("COMPILE_MODEL", True)
COMPILE_MODE = os.environ.get("COMPILE_MODE", "default")


class TokenBlockDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Non-overlapping next-token blocks over sharded uint16 token data."""

    def __init__(self, artifact_dir: Path, sequence_length: int) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        metadata_path = artifact_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"dataset metadata not found: {metadata_path}")

        with metadata_path.open(encoding="utf-8") as file:
            self.metadata: dict[str, Any] = json.load(file)

        storage = self.metadata.get("storage", {})
        if storage.get("dtype") != "uint16":
            raise ValueError(f"unsupported dataset dtype: {storage.get('dtype')!r}")
        if storage.get("endianness") != "little":
            raise ValueError(
                f"unsupported dataset endianness: {storage.get('endianness')!r}"
            )
        if storage.get("layout") != "flat_token_stream":
            raise ValueError(f"unsupported dataset layout: {storage.get('layout')!r}")
        if sys.byteorder != "little":
            raise RuntimeError(
                "uint16 dataset loading currently requires a little-endian host"
            )

        self.sequence_length = sequence_length
        self.shard_paths: list[Path] = []
        self.shard_token_counts: list[int] = []
        self.shard_ends: list[int] = []
        self._mapped_shards: dict[int, torch.Tensor] = {}

        total_tokens = 0
        for shard in self.metadata.get("shards", []):
            path = artifact_dir / shard["file"]
            token_count = int(shard["tokens"])
            expected_bytes = token_count * torch.uint16.itemsize
            if not path.is_file():
                raise FileNotFoundError(f"dataset shard not found: {path}")
            if path.stat().st_size != expected_bytes:
                raise ValueError(
                    f"{path} has {path.stat().st_size} bytes, expected {expected_bytes}"
                )
            self.shard_paths.append(path)
            self.shard_token_counts.append(token_count)
            total_tokens += token_count
            self.shard_ends.append(total_tokens)

        metadata_token_count = int(self.metadata.get("token_count", -1))
        if total_tokens != metadata_token_count:
            raise ValueError(
                f"shards contain {total_tokens:,} tokens, metadata declares "
                f"{metadata_token_count:,}"
            )
        if total_tokens <= sequence_length:
            raise ValueError(
                f"dataset has {total_tokens:,} tokens, but a block needs "
                f"{sequence_length + 1:,}"
            )

        self.total_tokens = total_tokens
        # Adjacent blocks share the boundary token: it is the final target of one
        # block and the first input of the next, so no next-token transition is lost.
        self.block_count = (total_tokens - 1) // sequence_length

    def __len__(self) -> int:
        return self.block_count

    def _mapped_shard(self, index: int) -> torch.Tensor:
        shard = self._mapped_shards.get(index)
        if shard is None:
            shard = torch.from_file(
                str(self.shard_paths[index]),
                shared=False,
                size=self.shard_token_counts[index],
                dtype=torch.uint16,
            )
            self._mapped_shards[index] = shard
        return shard

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += self.block_count
        if index < 0 or index >= self.block_count:
            raise IndexError(index)

        start = index * self.sequence_length
        remaining = self.sequence_length + 1
        shard_index = bisect.bisect_right(self.shard_ends, start)
        pieces: list[torch.Tensor] = []

        while remaining:
            shard_start = 0 if shard_index == 0 else self.shard_ends[shard_index - 1]
            local_start = start - shard_start
            take = min(remaining, self.shard_token_counts[shard_index] - local_start)
            pieces.append(
                self._mapped_shard(shard_index)[local_start : local_start + take]
            )
            start += take
            remaining -= take
            shard_index += 1

        tokens = (pieces[0] if len(pieces) == 1 else torch.cat(pieces)).to(torch.long)
        return tokens[:-1], tokens[1:]


def validate_configuration(train_dataset: TokenBlockDataset) -> None:
    positive_values = {
        "VOCAB_SIZE": VOCAB_SIZE,
        "N_BLOCKS": N_BLOCKS,
        "EMBED_DIM": EMBED_DIM,
        "ATTN_HEADS": ATTN_HEADS,
        "FFN_DIM": FFN_DIM,
        "SEQ_LEN": SEQ_LEN,
        "LR": LR,
        "BATCH_SIZE": BATCH_SIZE,
        "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
        "EPOCHS": EPOCHS,
        "VALIDATION_INTERVAL": VALIDATION_INTERVAL,
        "VALIDATION_STEPS": VALIDATION_STEPS,
        "DIAGNOSTICS_INTERVAL": DIAGNOSTICS_INTERVAL,
    }
    invalid = {name: value for name, value in positive_values.items() if value <= 0}
    if invalid:
        raise ValueError(f"these settings must be positive: {invalid}")
    if WARMUP_STEPS < 0:
        raise ValueError("WARMUP_STEPS must be non-negative")
    if LR_SCHEDULER not in {"cosine", "wsd"}:
        raise ValueError("LR_SCHEDULER must be either 'cosine' or 'wsd'")
    if not 0.0 < WSD_DECAY_FRACTION <= 1.0:
        raise ValueError("WSD_DECAY_FRACTION must be in (0, 1]")
    if INITIALIZATION_STD <= 0.0:
        raise ValueError("INITIALIZATION_STD must be positive")
    if MAX_STEPS < 0:
        raise ValueError("MAX_STEPS must be non-negative")
    if NUM_WORKERS < 0:
        raise ValueError("NUM_WORKERS must be non-negative")
    if TENSORBOARD_FLUSH_SECS <= 0:
        raise ValueError("TENSORBOARD_FLUSH_SECS must be positive")
    if not 0.0 <= DROPOUT <= 1.0:
        raise ValueError("DROPOUT must be between 0 and 1")
    if WEIGHT_DECAY < 0.0:
        raise ValueError("WEIGHT_DECAY must be non-negative")
    if Z_LOSS_COEFFICIENT < 0.0:
        raise ValueError("Z_LOSS_COEFFICIENT must be non-negative")
    if not 0.0 <= MIN_LR <= LR:
        raise ValueError("MIN_LR must be between zero and LR")
    if not 0.0 <= ADAM_BETA1 < 1.0 or not 0.0 <= ADAM_BETA2 < 1.0:
        raise ValueError("ADAM_BETA1 and ADAM_BETA2 must be in [0, 1)")
    if ADAM_EPS <= 0.0:
        raise ValueError("ADAM_EPS must be positive")
    if not math.isfinite(NORM_LR_MULTIPLIER) or NORM_LR_MULTIPLIER < 0.0:
        raise ValueError("NORM_LR_MULTIPLIER must be finite and non-negative")
    if len(train_dataset) < BATCH_SIZE * GRAD_ACCUM_STEPS:
        raise ValueError("training dataset is too small for one optimizer step")

    dataset_vocab_size = int(train_dataset.metadata["tokenizer"]["vocab_size"])
    if dataset_vocab_size != VOCAB_SIZE:
        raise ValueError(
            f"model VOCAB_SIZE={VOCAB_SIZE} does not match dataset vocabulary "
            f"size {dataset_vocab_size}"
        )


def build_loader(
    dataset: TokenBlockDataset,
    *,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle,
        "drop_last": True,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "persistent_workers": NUM_WORKERS > 0,
        "generator": generator,
    }
    if NUM_WORKERS > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
) -> LambdaLR:
    if WARMUP_STEPS >= total_steps:
        raise ValueError(
            f"WARMUP_STEPS ({WARMUP_STEPS}) must be smaller than total training "
            f"steps ({total_steps})"
        )

    minimum_ratio = MIN_LR / LR
    post_warmup_steps = total_steps - WARMUP_STEPS

    if LR_SCHEDULER == "wsd":
        wsd_stable_steps, wsd_decay_steps = wsd_phase_steps(total_steps)

    def lr_multiplier(step: int) -> float:
        if WARMUP_STEPS and step < WARMUP_STEPS:
            return (step + 1) / WARMUP_STEPS

        if LR_SCHEDULER == "cosine":
            progress = (step - WARMUP_STEPS) / max(1, post_warmup_steps - 1)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return minimum_ratio + (1.0 - minimum_ratio) * cosine

        decay_start = WARMUP_STEPS + wsd_stable_steps
        if step < decay_start:
            return 1.0
        if wsd_decay_steps == 1:
            return minimum_ratio

        progress = (step - decay_start) / (wsd_decay_steps - 1)
        progress = min(max(progress, 0.0), 1.0)
        return 1.0 - (1.0 - minimum_ratio) * progress

    return LambdaLR(optimizer, lr_lambda=lr_multiplier)


def wsd_phase_steps(total_steps: int) -> tuple[int, int]:
    """Return stable and decay steps after warmup for the configured WSD run."""

    post_warmup_steps = total_steps - WARMUP_STEPS
    decay_steps = max(1, math.ceil(post_warmup_steps * WSD_DECAY_FRACTION))
    return post_warmup_steps - decay_steps, decay_steps


def is_norm_parameter_name(parameter_name: str) -> bool:
    return "_norm." in parameter_name or parameter_name == "norm.weight"


def build_adamw_parameter_groups(
    model: torch.nn.Module,
) -> list[dict[str, Any]]:
    """Build decay, no-decay, and norm-specific AdamW parameter groups."""

    decay_parameters: list[torch.nn.Parameter] = []
    no_decay_parameters: list[torch.nn.Parameter] = []
    norm_parameters: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if is_norm_parameter_name(name):
            norm_parameters.append(parameter)
        elif parameter.ndim >= 2 and name != "embeddings.weight":
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)

    return [
        {"params": decay_parameters, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_parameters, "weight_decay": 0.0},
        {
            "params": norm_parameters,
            "weight_decay": 0.0,
            "lr": LR * NORM_LR_MULTIPLIER,
        },
    ]


def tensor_collection_stats(
    tensors: list[torch.Tensor],
) -> tuple[float, float, int]:
    """Return the global L2 norm, RMS, and element count of tensors."""

    if not tensors:
        return 0.0, 0.0, 0

    sum_of_squares = torch.zeros((), device=tensors[0].device, dtype=torch.float32)
    element_count = 0
    for tensor in tensors:
        detached = tensor.detach()
        sum_of_squares = sum_of_squares + detached.float().square().sum()
        element_count += detached.numel()

    norm = math.sqrt(sum_of_squares.item())
    rms = norm / math.sqrt(element_count)
    return norm, rms, element_count


def gradient_group_name(parameter_name: str) -> str:
    if parameter_name == "embeddings.weight":
        return "embeddings"
    if ".attn.c_attn." in parameter_name:
        return "attention_qkv"
    if ".attn.c_proj." in parameter_name:
        return "attention_output"
    if ".ffn.w1." in parameter_name or ".ffn.w3." in parameter_name:
        return "ffn_input_gate"
    if ".ffn.w2." in parameter_name:
        return "ffn_output"
    if is_norm_parameter_name(parameter_name):
        return "norms"
    return "other"


def gradient_norms_by_group(model: torch.nn.Module) -> dict[str, float]:
    grouped_gradients: dict[str, list[torch.Tensor]] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group_name = gradient_group_name(name)
        grouped_gradients.setdefault(group_name, []).append(parameter.grad)

    return {
        group_name: tensor_collection_stats(gradients)[0]
        for group_name, gradients in grouped_gradients.items()
    }


def sampled_logit_stats(logits: torch.Tensor) -> tuple[float, float]:
    """Measure a small, evenly strided logit sample to limit diagnostics cost."""

    sequence_stride = max(1, logits.size(-2) // 16)
    vocabulary_stride = max(1, logits.size(-1) // 1024)
    sample = logits.detach()[
        ..., ::sequence_stride, ::vocabulary_stride
    ].float()
    rms = sample.square().mean().sqrt().item()
    absolute_max = sample.abs().max().item()
    return rms, absolute_max


def next_token_training_objective(
    logits: torch.Tensor,
    targets: torch.Tensor,
    z_loss_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the total objective, cross-entropy, and unweighted z-loss."""

    cross_entropy = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )
    if z_loss_coefficient == 0.0:
        z_loss = cross_entropy.new_zeros(())
        return cross_entropy, cross_entropy, z_loss

    log_normalizer = torch.logsumexp(logits.float(), dim=-1)
    z_loss = log_normalizer.square().mean()
    objective = cross_entropy + z_loss_coefficient * z_loss
    return objective, cross_entropy, z_loss


@torch.no_grad()
def adamw_update_stats(
    optimizer: torch.optim.AdamW,
) -> tuple[float, float]:
    """Reconstruct the last AdamW parameter update without copying parameters."""

    sum_of_squares: torch.Tensor | None = None
    element_count = 0
    for group in optimizer.param_groups:
        parameters = [
            parameter
            for parameter in group["params"]
            if optimizer.state.get(parameter, {}).get("step") is not None
        ]
        if not parameters:
            continue

        step = int(optimizer.state[parameters[0]]["step"].item())
        beta1, beta2 = group["betas"]
        step_size = group["lr"] / (1.0 - beta1**step)
        square_root_bias_correction2 = math.sqrt(1.0 - beta2**step)
        decay_factor = 1.0 - group["lr"] * group["weight_decay"]

        for parameter in parameters:
            state = optimizer.state[parameter]
            denominator = (
                state["exp_avg_sq"].sqrt() / square_root_bias_correction2
            ).add(group["eps"])
            adaptive_update = state["exp_avg"] * (step_size / denominator)
            update = -(
                parameter.detach() * (group["lr"] * group["weight_decay"])
                + adaptive_update
            ) / decay_factor
            update_sum_of_squares = update.float().square().sum()
            sum_of_squares = (
                update_sum_of_squares
                if sum_of_squares is None
                else sum_of_squares + update_sum_of_squares
            )
            element_count += parameter.numel()

    if sum_of_squares is None or element_count == 0:
        return 0.0, 0.0
    update_norm = math.sqrt(sum_of_squares.item())
    return update_norm, update_norm / math.sqrt(element_count)


def config_string(hyperparameters: dict[str, object]) -> str:
    return ";".join(f"{name}={value}" for name, value in hyperparameters.items())


def save_final_model(
    model: torch.nn.Module,
    output_dir: Path,
    *,
    experiment_name: str,
    dataset_variant: str,
    global_step: int,
    train_loss: float,
    val_loss: float,
    hyperparameters: dict[str, object],
) -> Path:
    """Atomically save final model weights and the configuration that made them."""

    if Path(experiment_name).name != experiment_name or experiment_name in {"", "."}:
        raise ValueError(
            "EXPERIMENT_NAME must be a non-empty filename-safe name without slashes"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{experiment_name}.pt"
    temporary_path = output_dir / f".{experiment_name}.{os.getpid()}.tmp"
    checkpoint = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "experiment_name": experiment_name,
        "dataset_variant": dataset_variant,
        "global_step": global_step,
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "hyperparameters": hyperparameters,
        "config": config_string(hyperparameters),
    }

    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(model_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return model_path


def append_run_result(
    csv_path: Path,
    *,
    experiment_name: str,
    dataset_variant: str,
    train_loss: float,
    val_loss: float,
    train_script: str,
    model_path: Path,
    hyperparameters: dict[str, object],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "experiment_name",
                "dataset_variant",
                "train_loss",
                "validation_loss",
                "train_script",
                "model_path",
                "config",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "experiment_name": experiment_name,
                "dataset_variant": dataset_variant,
                "train_loss": format(train_loss, ".10g"),
                "validation_loss": format(val_loss, ".10g"),
                "train_script": train_script,
                "model_path": str(model_path),
                "config": config_string(hyperparameters),
            }
        )


@torch.no_grad()
def validation_loss(
    train_model: torch.nn.Module,
    model: torch.nn.Module,
    validation_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    model.eval()
    loss_sum = 0.0
    token_count = 0

    for batch_index, (input_ids, targets) in enumerate(validation_loader):
        if batch_index >= VALIDATION_STEPS:
            break
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = train_model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="sum",
            )

        loss_sum += loss.item()
        token_count += targets.numel()

    model.train()
    if token_count == 0:
        raise RuntimeError("validation loader did not produce any tokens")
    return loss_sum / token_count


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("BF16 pretraining requires a CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA GPU does not support BF16")

    device = torch.device("cuda")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    if DATASET_PREFIX:
        train_dir = DATASET_DIR / f"train_{DATASET_PREFIX}_{DATASET_VARIANT}"
        validation_dir = DATASET_DIR / f"validation_{DATASET_PREFIX}"
    else:
        train_dir = DATASET_DIR / f"train_{DATASET_VARIANT}"
        validation_dir = DATASET_DIR / "validation"
    train_dataset = TokenBlockDataset(train_dir, SEQ_LEN)
    validation_dataset = TokenBlockDataset(validation_dir, SEQ_LEN)
    validate_configuration(train_dataset)

    validation_vocab_size = int(validation_dataset.metadata["tokenizer"]["vocab_size"])
    if validation_vocab_size != VOCAB_SIZE:
        raise ValueError(
            f"validation vocabulary size {validation_vocab_size} does not match "
            f"VOCAB_SIZE={VOCAB_SIZE}"
        )

    data_generator = torch.Generator().manual_seed(SEED)
    train_loader = build_loader(
        train_dataset,
        shuffle=True,
        generator=data_generator,
    )
    validation_loader = build_loader(validation_dataset, shuffle=False)

    optimizer_steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
    available_steps = EPOCHS * optimizer_steps_per_epoch
    total_steps = min(MAX_STEPS, available_steps) if MAX_STEPS else available_steps
    if WARMUP_STEPS >= total_steps:
        raise ValueError(
            f"WARMUP_STEPS={WARMUP_STEPS} must be smaller than the "
            f"{total_steps} optimizer steps in this run"
        )
    if LR_SCHEDULER == "wsd":
        wsd_stable_steps, wsd_decay_steps = wsd_phase_steps(total_steps)
    else:
        wsd_stable_steps = None
        wsd_decay_steps = None

    model = DecoderTransformer(
        vocab_size=VOCAB_SIZE,
        n_blocks=N_BLOCKS,
        embed_dim=EMBED_DIM,
        attn_heads=ATTN_HEADS,
        ffn_dim=FFN_DIM,
        dropout=DROPOUT,
        seq_len=SEQ_LEN,
        rope_base=ROPE_BASE,
        initialization=INITIALIZATION,
        initialization_std=INITIALIZATION_STD,
        qk_norm=QK_NORM,
    ).to(device)

    optimizer_parameter_groups = build_adamw_parameter_groups(model)
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups,
        lr=LR,
        betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPS,
        weight_decay=0.0,
        fused=True,
    )
    scheduler = build_scheduler(optimizer, total_steps)
    train_model = (
        torch.compile(model, mode=COMPILE_MODE, dynamic=False)
        if COMPILE_MODEL
        else model
    )

    run_dir = LOG_DIR / EXPERIMENT_NAME
    writer = SummaryWriter(
        log_dir=run_dir,
        flush_secs=TENSORBOARD_FLUSH_SECS,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    decayed_parameter_count = sum(
        parameter.numel() for parameter in optimizer_parameter_groups[0]["params"]
    )
    no_decay_parameter_count = sum(
        parameter.numel() for parameter in optimizer_parameter_groups[1]["params"]
    )
    norm_parameter_count = sum(
        parameter.numel() for parameter in optimizer_parameter_groups[2]["params"]
    )
    hyperparameters: dict[str, object] = {
        "VOCAB_SIZE": VOCAB_SIZE,
        "N_BLOCKS": N_BLOCKS,
        "EMBED_DIM": EMBED_DIM,
        "ATTN_HEADS": ATTN_HEADS,
        "FFN_DIM": FFN_DIM,
        "SEQ_LEN": SEQ_LEN,
        "ROPE_BASE": ROPE_BASE,
        "INITIALIZATION": INITIALIZATION,
        "INITIALIZATION_STD": INITIALIZATION_STD,
        "QK_NORM": QK_NORM,
        "LR": LR,
        "MIN_LR": MIN_LR,
        "LR_SCHEDULER": LR_SCHEDULER,
        "WSD_DECAY_FRACTION": WSD_DECAY_FRACTION,
        "ADAM_BETA1": ADAM_BETA1,
        "ADAM_BETA2": ADAM_BETA2,
        "ADAM_EPS": ADAM_EPS,
        "NORM_LR_MULTIPLIER": NORM_LR_MULTIPLIER,
        "WARMUP_STEPS": WARMUP_STEPS,
        "BATCH_SIZE": BATCH_SIZE,
        "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
        "EPOCHS": EPOCHS,
        "MAX_STEPS": MAX_STEPS,
        "DROPOUT": DROPOUT,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "Z_LOSS_COEFFICIENT": Z_LOSS_COEFFICIENT,
        "MAX_GRAD_NORM": MAX_GRAD_NORM,
        "SEED": SEED,
        "DATASET_PREFIX": DATASET_PREFIX,
        "DATASET_VARIANT": DATASET_VARIANT,
        "VALIDATION_INTERVAL": VALIDATION_INTERVAL,
        "VALIDATION_STEPS": VALIDATION_STEPS,
        "DIAGNOSTICS_INTERVAL": DIAGNOSTICS_INTERVAL,
        "COMPILE_MODEL": COMPILE_MODEL,
        "COMPILE_MODE": COMPILE_MODE,
    }
    settings = {
        "experiment_name": EXPERIMENT_NAME,
        "dataset_prefix": DATASET_PREFIX,
        "dataset_variant": DATASET_VARIANT,
        "train_tokens": train_dataset.total_tokens,
        "validation_tokens": validation_dataset.total_tokens,
        "parameters": parameter_count,
        "initialization": INITIALIZATION,
        "initialization_std": INITIALIZATION_STD,
        "qk_norm": QK_NORM,
        "sequence_length": SEQ_LEN,
        "micro_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
        "epochs": EPOCHS,
        "max_steps": MAX_STEPS,
        "total_steps": total_steps,
        "learning_rate": LR,
        "minimum_learning_rate": MIN_LR,
        "lr_scheduler": LR_SCHEDULER,
        "wsd_decay_fraction": WSD_DECAY_FRACTION,
        "wsd_stable_steps": wsd_stable_steps,
        "wsd_decay_steps": wsd_decay_steps,
        "adam_beta1": ADAM_BETA1,
        "adam_beta2": ADAM_BETA2,
        "adam_epsilon": ADAM_EPS,
        "norm_learning_rate_multiplier": NORM_LR_MULTIPLIER,
        "norm_learning_rate": LR * NORM_LR_MULTIPLIER,
        "warmup_steps": WARMUP_STEPS,
        "weight_decay": WEIGHT_DECAY,
        "z_loss_coefficient": Z_LOSS_COEFFICIENT,
        "decayed_parameters": decayed_parameter_count,
        "no_decay_parameters": no_decay_parameter_count,
        "norm_parameters": norm_parameter_count,
        "max_gradient_norm": MAX_GRAD_NORM,
        "validation_interval": VALIDATION_INTERVAL,
        "validation_steps": VALIDATION_STEPS,
        "diagnostics_interval": DIAGNOSTICS_INTERVAL,
        "tensorboard_flush_seconds": TENSORBOARD_FLUSH_SECS,
        "runs_csv": str(RUNS_CSV),
        "model_output_dir": str(MODEL_OUTPUT_DIR),
        "train_script": TRAIN_SCRIPT,
        "compile_model": COMPILE_MODEL,
        "compile_mode": COMPILE_MODE,
        "seed": SEED,
    }
    print(json.dumps(settings, indent=2), flush=True)
    writer.add_text("configuration", json.dumps(settings, indent=2), 0)
    writer.add_scalar("model/parameter_count", parameter_count, 0)

    initial_parameter_norm, initial_parameter_rms, _ = tensor_collection_stats(
        list(model.parameters())
    )
    embedding_norm, embedding_rms, _ = tensor_collection_stats(
        [model.embeddings.weight]
    )
    writer.add_scalar("diagnostics/parameter_norm", initial_parameter_norm, 0)
    writer.add_scalar("diagnostics/parameter_rms", initial_parameter_rms, 0)
    writer.add_scalar("diagnostics/embedding_norm", embedding_norm, 0)
    writer.add_scalar("diagnostics/embedding_rms", embedding_rms, 0)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\tTrainable: {trainable_params}")

    global_step = 0
    tokens_seen = 0
    clipped_steps = 0
    tokens_per_step = BATCH_SIZE * GRAD_ACCUM_STEPS * SEQ_LEN
    throughput_window_tokens = 0
    throughput_window_steps = 0
    throughput_window_start = time.perf_counter()
    uniform_loss = math.log(VOCAB_SIZE)
    final_train_loss: float | None = None
    final_validation_loss: float | None = None
    model.train()
    try:
        for epoch in range(EPOCHS):
            train_iterator = iter(train_loader)
            for _ in range(optimizer_steps_per_epoch):
                if global_step >= total_steps:
                    break
                optimizer.zero_grad(set_to_none=True)
                accumulated_loss = 0.0
                accumulated_objective = 0.0
                accumulated_z_loss = 0.0

                for _ in range(GRAD_ACCUM_STEPS):
                    input_ids, targets = next(train_iterator)
                    input_ids = input_ids.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = train_model(input_ids)
                        objective, loss, z_loss = next_token_training_objective(
                            logits,
                            targets,
                            Z_LOSS_COEFFICIENT,
                        )

                    if not torch.isfinite(objective):
                        raise FloatingPointError(
                            f"non-finite training objective at step "
                            f"{global_step + 1}: {objective.item()}"
                        )
                    accumulated_loss += loss.detach().item()
                    accumulated_objective += objective.detach().item()
                    accumulated_z_loss += z_loss.detach().item()
                    (objective / GRAD_ACCUM_STEPS).backward()

                next_step = global_step + 1
                should_log_diagnostics = (
                    next_step == 1
                    or next_step % DIAGNOSTICS_INTERVAL == 0
                    or next_step == total_steps
                )
                if should_log_diagnostics:
                    gradient_group_norms = gradient_norms_by_group(model)
                    parameter_norm, parameter_rms, _ = tensor_collection_stats(
                        list(model.parameters())
                    )
                    embedding_norm, embedding_rms, _ = tensor_collection_stats(
                        [model.embeddings.weight]
                    )
                    logit_rms, logit_absolute_max = sampled_logit_stats(logits)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=MAX_GRAD_NORM if MAX_GRAD_NORM > 0.0 else float("inf"),
                    error_if_nonfinite=True,
                )
                learning_rate = optimizer.param_groups[0]["lr"]
                norm_learning_rate = optimizer.param_groups[2]["lr"]
                optimizer.step()
                if should_log_diagnostics:
                    update_norm, update_rms = adamw_update_stats(optimizer)
                scheduler.step()
                global_step += 1
                tokens_seen += tokens_per_step
                throughput_window_tokens += tokens_per_step
                throughput_window_steps += 1

                train_loss = accumulated_loss / GRAD_ACCUM_STEPS
                train_objective = accumulated_objective / GRAD_ACCUM_STEPS
                train_z_loss = accumulated_z_loss / GRAD_ACCUM_STEPS
                final_train_loss = train_loss
                gradient_norm = grad_norm.item()
                if MAX_GRAD_NORM > 0.0:
                    clip_coefficient = min(
                        1.0,
                        MAX_GRAD_NORM / (gradient_norm + 1e-6),
                    )
                    gradient_clipped = gradient_norm > MAX_GRAD_NORM
                else:
                    clip_coefficient = 1.0
                    gradient_clipped = False
                clipped_steps += int(gradient_clipped)

                writer.add_scalar("train/loss", train_loss, global_step)
                writer.add_scalar("train/objective", train_objective, global_step)
                writer.add_scalar("train/z_loss", train_z_loss, global_step)
                writer.add_scalar(
                    "train/z_loss_contribution",
                    Z_LOSS_COEFFICIENT * train_z_loss,
                    global_step,
                )
                writer.add_scalar(
                    "train/loss_vs_uniform",
                    train_loss - uniform_loss,
                    global_step,
                )
                if train_loss < math.log(sys.float_info.max):
                    writer.add_scalar(
                        "train/perplexity",
                        math.exp(train_loss),
                        global_step,
                    )
                writer.add_scalar("train/gradient_norm", gradient_norm, global_step)
                writer.add_scalar(
                    "train/gradient_rms",
                    gradient_norm / math.sqrt(trainable_params),
                    global_step,
                )
                writer.add_scalar(
                    "train/clip_coefficient", clip_coefficient, global_step
                )
                writer.add_scalar(
                    "train/gradient_clipped", float(gradient_clipped), global_step
                )
                writer.add_scalar(
                    "train/clip_fraction", clipped_steps / global_step, global_step
                )
                writer.add_scalar("train/learning_rate", learning_rate, global_step)
                writer.add_scalar(
                    "train/norm_learning_rate", norm_learning_rate, global_step
                )
                writer.add_scalar("train/tokens_seen", tokens_seen, global_step)

                if should_log_diagnostics:
                    writer.add_scalar(
                        "diagnostics/parameter_norm", parameter_norm, global_step
                    )
                    writer.add_scalar(
                        "diagnostics/parameter_rms", parameter_rms, global_step
                    )
                    writer.add_scalar(
                        "diagnostics/embedding_norm", embedding_norm, global_step
                    )
                    writer.add_scalar(
                        "diagnostics/embedding_rms", embedding_rms, global_step
                    )
                    writer.add_scalar(
                        "diagnostics/gradient_to_parameter_norm",
                        gradient_norm / max(parameter_norm, 1e-12),
                        global_step,
                    )
                    writer.add_scalar(
                        "diagnostics/update_norm", update_norm, global_step
                    )
                    writer.add_scalar(
                        "diagnostics/update_rms", update_rms, global_step
                    )
                    writer.add_scalar(
                        "diagnostics/update_to_parameter_norm",
                        update_norm / max(parameter_norm, 1e-12),
                        global_step,
                    )
                    writer.add_scalar(
                        "diagnostics/logit_rms_sample", logit_rms, global_step
                    )
                    writer.add_scalar(
                        "diagnostics/logit_absolute_max_sample",
                        logit_absolute_max,
                        global_step,
                    )
                    for group_name, group_norm in gradient_group_norms.items():
                        writer.add_scalar(
                            f"gradient_groups/{group_name}",
                            group_norm,
                            global_step,
                        )
                    gibibyte = 1024**3
                    writer.add_scalar(
                        "system/cuda_memory_allocated_gib",
                        torch.cuda.memory_allocated(device) / gibibyte,
                        global_step,
                    )
                    writer.add_scalar(
                        "system/cuda_memory_reserved_gib",
                        torch.cuda.memory_reserved(device) / gibibyte,
                        global_step,
                    )
                    writer.add_scalar(
                        "system/cuda_peak_memory_allocated_gib",
                        torch.cuda.max_memory_allocated(device) / gibibyte,
                        global_step,
                    )
                print(
                    f"step={global_step}/{total_steps} epoch={epoch + 1}/{EPOCHS} "
                    f"loss={train_loss:.6f} grad_norm={gradient_norm:.6f} "
                    f"lr={learning_rate:.8g}"
                    + (
                        f" z_loss={train_z_loss:.6f} "
                        f"objective={train_objective:.6f}"
                        if Z_LOSS_COEFFICIENT > 0.0
                        else ""
                    ),
                    flush=True,
                )

                should_validate = (
                    global_step % VALIDATION_INTERVAL == 0 or global_step == total_steps
                )
                if should_log_diagnostics or should_validate:
                    torch.cuda.synchronize(device)
                    throughput_elapsed = time.perf_counter() - throughput_window_start
                    writer.add_scalar(
                        "performance/tokens_per_second",
                        throughput_window_tokens / throughput_elapsed,
                        global_step,
                    )
                    writer.add_scalar(
                        "performance/optimizer_step_seconds",
                        throughput_elapsed / throughput_window_steps,
                        global_step,
                    )
                    throughput_window_tokens = 0
                    throughput_window_steps = 0
                    throughput_window_start = time.perf_counter()
                if should_validate:
                    val_loss = validation_loss(
                        train_model,
                        model,
                        validation_loader,
                        device,
                    )
                    final_validation_loss = val_loss
                    writer.add_scalar("validation/loss", val_loss, global_step)
                    writer.add_scalar(
                        "validation/loss_vs_uniform",
                        val_loss - uniform_loss,
                        global_step,
                    )
                    if val_loss < math.log(sys.float_info.max):
                        writer.add_scalar(
                            "validation/perplexity",
                            math.exp(val_loss),
                            global_step,
                        )
                    writer.flush()
                    print(
                        f"step={global_step}/{total_steps} "
                        f"validation_loss={val_loss:.6f}",
                        flush=True,
                    )
                    throughput_window_start = time.perf_counter()
            if global_step >= total_steps:
                break
    finally:
        writer.close()

    if final_train_loss is None or final_validation_loss is None:
        raise RuntimeError("training completed without final loss values")
    model_path = save_final_model(
        model,
        MODEL_OUTPUT_DIR,
        experiment_name=EXPERIMENT_NAME,
        dataset_variant=DATASET_VARIANT,
        global_step=global_step,
        train_loss=final_train_loss,
        val_loss=final_validation_loss,
        hyperparameters=hyperparameters,
    )
    print(f"final model saved to {model_path}", flush=True)
    append_run_result(
        RUNS_CSV,
        experiment_name=EXPERIMENT_NAME,
        dataset_variant=DATASET_VARIANT,
        train_loss=final_train_loss,
        val_loss=final_validation_loss,
        train_script=TRAIN_SCRIPT,
        model_path=model_path,
        hyperparameters=hyperparameters,
    )
    print(f"run result appended to {RUNS_CSV}", flush=True)


if __name__ == "__main__":
    main()
