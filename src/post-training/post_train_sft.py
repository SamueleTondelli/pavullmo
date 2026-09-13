"""Full-parameter supervised fine-tuning with assistant-only cross-entropy."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
MODEL_DIR = PROJECT_DIR / "src" / "model"
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


BASE_CHECKPOINT_VALUE = os.environ.get("BASE_CHECKPOINT")
DATASET_DIR = Path(os.environ.get("POST_DATASET_DIR", str(SCRIPT_DIR / "dataset"))).expanduser()
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "pavullmo_sft")
LR = float(os.environ.get("LR", 1e-5))
MIN_LR = float(os.environ.get("MIN_LR", 0.0))
ADAM_BETA1 = float(os.environ.get("ADAM_BETA1", 0.9))
ADAM_BETA2 = float(os.environ.get("ADAM_BETA2", 0.95))
ADAM_EPS = float(os.environ.get("ADAM_EPS", 1e-8))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.01))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", 0))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 2))
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", 1))
EPOCHS = int(os.environ.get("EPOCHS", 3))
MAX_STEPS = int(os.environ.get("MAX_STEPS", 0))
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", 1.0))
VALIDATION_INTERVAL = int(os.environ.get("VALIDATION_INTERVAL", 1))
SEED = int(os.environ.get("SEED", 42))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 0))
DEVICE = os.environ.get("DEVICE", "auto").strip().lower()
COMPILE_MODEL = env_bool("COMPILE_MODEL", False)
COMPILE_MODE = os.environ.get("COMPILE_MODE", "default")
EVAL_ONLY = env_bool("EVAL_ONLY", False)
EVALUATE_TEST = env_bool("EVALUATE_TEST", False)
LOG_DIR = Path(os.environ.get("LOG_DIR", str(PROJECT_DIR / "post-training-runs"))).expanduser()
MODEL_OUTPUT_DIR = Path(
    os.environ.get("MODEL_OUTPUT_DIR", str(PROJECT_DIR / "post-training-models"))
).expanduser()
RUNS_CSV = Path(os.environ.get("RUNS_CSV", str(SCRIPT_DIR / "post_training_runs.csv"))).expanduser()
TENSORBOARD_FLUSH_SECS = int(os.environ.get("TENSORBOARD_FLUSH_SECS", 5))


class SFTDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: Path, sequence_length: int) -> None:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if artifact.get("format_version") != 1:
            raise ValueError(f"unsupported SFT artifact format in {path}")
        self.examples = artifact.get("examples")
        if not isinstance(self.examples, list) or not self.examples:
            raise ValueError(f"SFT artifact has no examples: {path}")
        self.sequence_length = sequence_length
        for example in self.examples:
            token_ids = example["token_ids"]
            loss_mask = example["loss_mask"]
            if token_ids.ndim != 1 or token_ids.numel() < 2:
                raise ValueError(f"invalid token_ids for {example['id']!r}")
            if token_ids.numel() > sequence_length:
                raise ValueError(
                    f"{example['id']!r} has {token_ids.numel()} tokens, exceeding "
                    f"checkpoint context length {sequence_length}"
                )
            if token_ids.shape != loss_mask.shape or not loss_mask.any():
                raise ValueError(f"invalid loss mask for {example['id']!r}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


def collate_sft(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    max_input_length = max(item["token_ids"].numel() - 1 for item in batch)
    input_ids = torch.full((len(batch), max_input_length), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_input_length), -100, dtype=torch.long)
    ids: list[str] = []
    for row, item in enumerate(batch):
        tokens = item["token_ids"].long()
        targets = tokens[1:]
        target_mask = item["loss_mask"][1:].bool()
        length = tokens.numel() - 1
        input_ids[row, :length] = tokens[:-1]
        labels[row, :length] = torch.where(target_mask, targets, -100)
        ids.append(item["id"])
    return {"input_ids": input_ids, "labels": labels, "ids": ids}


def select_device() -> torch.device:
    if DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DEVICE not in {"cpu", "cuda"}:
        raise ValueError("DEVICE must be 'auto', 'cpu', or 'cuda'")
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE=cuda requested, but CUDA is unavailable")
    return torch.device(DEVICE)


def load_base_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[DecoderTransformer, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    hyperparameters = checkpoint.get("hyperparameters")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(hyperparameters, dict) or not isinstance(state_dict, dict):
        raise ValueError("base checkpoint lacks hyperparameters or model_state_dict")
    required = ("VOCAB_SIZE", "N_BLOCKS", "EMBED_DIM", "ATTN_HEADS", "FFN_DIM", "SEQ_LEN", "ROPE_BASE", "DROPOUT")
    missing = [name for name in required if name not in hyperparameters]
    if missing:
        raise ValueError("base checkpoint missing: " + ", ".join(missing))
    split_qkv = bool(
        hyperparameters.get(
            "SPLIT_QKV_PROJECTIONS",
            any(".attn.q_proj." in name for name in state_dict),
        )
    )
    hyperparameters = dict(hyperparameters)
    hyperparameters["SPLIT_QKV_PROJECTIONS"] = split_qkv
    model = DecoderTransformer(
        vocab_size=int(hyperparameters["VOCAB_SIZE"]),
        n_blocks=int(hyperparameters["N_BLOCKS"]),
        embed_dim=int(hyperparameters["EMBED_DIM"]),
        attn_heads=int(hyperparameters["ATTN_HEADS"]),
        ffn_dim=int(hyperparameters["FFN_DIM"]),
        dropout=float(hyperparameters["DROPOUT"]),
        seq_len=int(hyperparameters["SEQ_LEN"]),
        rope_base=float(hyperparameters["ROPE_BASE"]),
        initialization=str(hyperparameters.get("INITIALIZATION", "pytorch_default")),
        initialization_std=float(hyperparameters.get("INITIALIZATION_STD", 0.02)),
        qk_norm=bool(hyperparameters.get("QK_NORM", False)),
        split_qkv_projections=split_qkv,
    )
    model.load_state_dict(state_dict)
    return model.to(device), checkpoint, hyperparameters


def build_loader(dataset: SFTDataset, *, shuffle: bool, pad_id: int) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator().manual_seed(SEED) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        generator=generator,
        collate_fn=partial(collate_sft, pad_id=pad_id),
    )


def autocast_context(device: torch.device):
    if device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA SFT requires BF16 support")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction=reduction
    )


@torch.no_grad()
def evaluate(model_for_forward: torch.nn.Module, model: torch.nn.Module, loader: DataLoader[dict[str, Any]], device: torch.device) -> tuple[float, int]:
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    target_count = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with autocast_context(device):
            logits = model_for_forward(input_ids)
            loss = masked_cross_entropy(logits, labels, reduction="sum")
        loss_sum += loss.item()
        target_count += int((labels != -100).sum().item())
    model.train(was_training)
    if target_count == 0:
        raise RuntimeError("evaluation split contains no assistant targets")
    return loss_sum / target_count, target_count


def scheduler_for(optimizer: torch.optim.Optimizer, total_steps: int) -> LambdaLR:
    if WARMUP_STEPS >= total_steps:
        raise ValueError("WARMUP_STEPS must be smaller than the number of optimizer steps")
    minimum_ratio = MIN_LR / LR

    def multiplier(step: int) -> float:
        if WARMUP_STEPS and step < WARMUP_STEPS:
            return (step + 1) / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return LambdaLR(optimizer, multiplier)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_training_history(
    run_dir: Path,
    train_history: list[tuple[int, float, float]],
    validation_history: list[tuple[int, float]],
) -> None:
    """Persist machine-readable metrics and a static loss curve."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["series", "optimizer_step", "epoch", "loss"])
        for step, epoch, loss in train_history:
            writer.writerow(["train", step, format(epoch, ".10g"), format(loss, ".10g")])
        epoch_by_step = {step: epoch for step, epoch, _ in train_history}
        for step, loss in validation_history:
            writer.writerow(
                ["validation", step, format(epoch_by_step.get(step, 0.0), ".10g"), format(loss, ".10g")]
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5))
    if train_history:
        axis.plot(
            [item[0] for item in train_history],
            [item[2] for item in train_history],
            label="Training loss",
            linewidth=1.2,
            alpha=0.8,
        )
    axis.plot(
        [item[0] for item in validation_history],
        [item[1] for item in validation_history],
        label="Validation loss",
        marker="o",
        linewidth=2,
    )
    axis.set(title="SFT loss", xlabel="Optimizer step", ylabel="Assistant-token cross-entropy")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(run_dir / "loss_curves.png", dpi=160)
    plt.close(figure)


def main() -> None:
    if not BASE_CHECKPOINT_VALUE:
        raise ValueError("set BASE_CHECKPOINT to the pretrained .pt or .pt.zip file")
    if Path(EXPERIMENT_NAME).name != EXPERIMENT_NAME or EXPERIMENT_NAME in {"", "."}:
        raise ValueError("EXPERIMENT_NAME must be a filename-safe name without slashes")
    positive = {"LR": LR, "BATCH_SIZE": BATCH_SIZE, "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS, "EPOCHS": EPOCHS, "VALIDATION_INTERVAL": VALIDATION_INTERVAL}
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"settings must be positive: {invalid}")
    if MAX_STEPS < 0 or WARMUP_STEPS < 0 or NUM_WORKERS < 0:
        raise ValueError("MAX_STEPS, WARMUP_STEPS, and NUM_WORKERS must be non-negative")
    if not 0.0 <= MIN_LR <= LR or WEIGHT_DECAY < 0 or MAX_GRAD_NORM < 0:
        raise ValueError("invalid optimizer or schedule setting")

    checkpoint_path = Path(BASE_CHECKPOINT_VALUE).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    device = select_device()
    random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    model, base_checkpoint, architecture = load_base_checkpoint(checkpoint_path, device)
    sequence_length = int(architecture["SEQ_LEN"])
    metadata = json.loads((DATASET_DIR / "metadata.json").read_text(encoding="utf-8"))
    if int(metadata["vocab_size"]) != int(architecture["VOCAB_SIZE"]):
        raise ValueError("SFT dataset vocabulary does not match base checkpoint")
    if int(metadata["context_length"]) > sequence_length:
        raise ValueError("SFT dataset context length exceeds base checkpoint limit")

    train_dataset = SFTDataset(DATASET_DIR / "train.pt", sequence_length)
    validation_dataset = SFTDataset(DATASET_DIR / "validation.pt", sequence_length)
    test_dataset = SFTDataset(DATASET_DIR / "test.pt", sequence_length)
    special_token_ids = metadata.get("special_token_ids", {})
    pad_id = int(special_token_ids.get("pad", -1))
    if pad_id < 0:
        raise ValueError("SFT metadata does not define a valid PAD token")
    train_loader = build_loader(train_dataset, shuffle=True, pad_id=pad_id)
    validation_loader = build_loader(validation_dataset, shuffle=False, pad_id=pad_id)
    test_loader = build_loader(test_dataset, shuffle=False, pad_id=pad_id)
    model_for_forward = torch.compile(model, mode=COMPILE_MODE, dynamic=True) if COMPILE_MODEL else model

    initial_validation_loss, validation_targets = evaluate(
        model_for_forward, model, validation_loader, device
    )
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,}")
    print(f"initial_validation_loss={initial_validation_loss:.6f} targets={validation_targets}")
    if EVAL_ONLY:
        if EVALUATE_TEST:
            test_loss, test_targets = evaluate(model_for_forward, model, test_loader, device)
            print(f"test_loss={test_loss:.6f} targets={test_targets}")
        return

    updates_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
    available_steps = EPOCHS * updates_per_epoch
    total_steps = min(MAX_STEPS, available_steps) if MAX_STEPS else available_steps
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(ADAM_BETA1, ADAM_BETA2), eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY, fused=device.type == "cuda"
    )
    scheduler = scheduler_for(optimizer, total_steps)
    run_dir = LOG_DIR / EXPERIMENT_NAME
    writer = SummaryWriter(run_dir, flush_secs=TENSORBOARD_FLUSH_SECS)
    global_step = 0
    final_train_loss = float("nan")
    final_validation_loss = initial_validation_loss
    train_history: list[tuple[int, float, float]] = []
    validation_history = [(0, initial_validation_loss)]
    writer.add_scalar("validation/loss", initial_validation_loss, 0)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch in range(EPOCHS):
            accumulated_batches = 0
            for batch_index, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                with autocast_context(device):
                    logits = model_for_forward(input_ids)
                    loss = masked_cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite SFT loss: {loss.item()}")
                (loss / GRAD_ACCUM_STEPS).backward()
                final_train_loss = loss.item()
                accumulated_batches += 1
                is_last_batch = batch_index + 1 == len(train_loader)
                if accumulated_batches < GRAD_ACCUM_STEPS and not is_last_batch:
                    continue
                if is_last_batch and accumulated_batches < GRAD_ACCUM_STEPS:
                    gradient_scale = GRAD_ACCUM_STEPS / accumulated_batches
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(gradient_scale)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), MAX_GRAD_NORM if MAX_GRAD_NORM > 0 else float("inf")
                )
                current_lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated_batches = 0
                global_step += 1
                epoch_progress = epoch + (batch_index + 1) / len(train_loader)
                train_history.append((global_step, epoch_progress, final_train_loss))
                writer.add_scalar("train/loss", final_train_loss, global_step)
                writer.add_scalar("train/gradient_norm", grad_norm.item(), global_step)
                writer.add_scalar("train/learning_rate", current_lr, global_step)
                writer.add_scalar("progress/epoch", epoch_progress, global_step)
                print(
                    f"step={global_step}/{total_steps} epoch={epoch + 1}/{EPOCHS} "
                    f"loss={final_train_loss:.6f} grad_norm={grad_norm.item():.6f} lr={current_lr:.8g}",
                    flush=True,
                )
                if global_step % VALIDATION_INTERVAL == 0 or global_step == total_steps:
                    final_validation_loss, validation_targets = evaluate(
                        model_for_forward, model, validation_loader, device
                    )
                    writer.add_scalar("validation/loss", final_validation_loss, global_step)
                    validation_history.append((global_step, final_validation_loss))
                    write_training_history(run_dir, train_history, validation_history)
                    print(f"step={global_step}/{total_steps} validation_loss={final_validation_loss:.6f}")
                if global_step >= total_steps:
                    break
            if global_step >= total_steps:
                break
    finally:
        write_training_history(run_dir, train_history, validation_history)
        writer.flush()
        writer.close()

    test_loss: float | None = None
    test_targets = 0
    if EVALUATE_TEST:
        test_loss, test_targets = evaluate(model_for_forward, model, test_loader, device)
    hyperparameters = {
        "BASE_CHECKPOINT": str(checkpoint_path), "LR": LR, "MIN_LR": MIN_LR,
        "ADAM_BETA1": ADAM_BETA1, "ADAM_BETA2": ADAM_BETA2, "ADAM_EPS": ADAM_EPS,
        "WEIGHT_DECAY": WEIGHT_DECAY, "WARMUP_STEPS": WARMUP_STEPS,
        "BATCH_SIZE": BATCH_SIZE, "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
        "EPOCHS": EPOCHS, "MAX_STEPS": MAX_STEPS, "MAX_GRAD_NORM": MAX_GRAD_NORM,
        "VALIDATION_INTERVAL": VALIDATION_INTERVAL, "SEED": SEED,
        "NUM_WORKERS": NUM_WORKERS, "DEVICE": DEVICE, "COMPILE_MODEL": COMPILE_MODEL,
        "COMPILE_MODE": COMPILE_MODE, "EVALUATE_TEST": EVALUATE_TEST,
        "TENSORBOARD_FLUSH_SECS": TENSORBOARD_FLUSH_SECS,
    }
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_OUTPUT_DIR / f"{EXPERIMENT_NAME}.pt"
    torch.save(
        {
            "format_version": 1,
            "training_stage": "sft",
            "model_state_dict": model.state_dict(),
            "hyperparameters": architecture,
            "sft_hyperparameters": hyperparameters,
            "base_checkpoint": str(checkpoint_path),
            "base_checkpoint_sha256": sha256_file(checkpoint_path),
            "base_experiment_name": base_checkpoint.get("experiment_name"),
            "global_step": global_step,
            "train_loss": final_train_loss,
            "validation_loss": final_validation_loss,
            "test_loss": test_loss,
            "dataset_metadata": metadata,
        },
        output_path,
    )
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RUNS_CSV.exists() or RUNS_CSV.stat().st_size == 0
    with RUNS_CSV.open("a", encoding="utf-8", newline="") as file:
        writer_csv = csv.DictWriter(
            file,
            fieldnames=["experiment_name", "base_checkpoint", "global_step", "train_loss", "validation_loss", "test_loss", "model_path", "config"],
        )
        if write_header:
            writer_csv.writeheader()
        writer_csv.writerow(
            {
                "experiment_name": EXPERIMENT_NAME,
                "base_checkpoint": str(checkpoint_path),
                "global_step": global_step,
                "train_loss": format(final_train_loss, ".10g"),
                "validation_loss": format(final_validation_loss, ".10g"),
                "test_loss": "" if test_loss is None else format(test_loss, ".10g"),
                "model_path": str(output_path),
                "config": ";".join(f"{key}={value}" for key, value in hyperparameters.items()),
            }
        )
    if test_loss is not None:
        print(f"test_loss={test_loss:.6f} targets={test_targets}")
    print(f"SFT checkpoint saved to {output_path}")


if __name__ == "__main__":
    main()
