from __future__ import annotations

import argparse
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from components.data import TokenBlockDataset
from pavullmo.generate import load_model


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


BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
SEED = int(os.environ.get("SEED", 42))

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 4))
PIN_MEMORY = env_bool("PIN_MEMORY", True)


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
        "drop_last": False,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "persistent_workers": NUM_WORKERS > 0,
        "generator": generator,
    }
    if NUM_WORKERS > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


@torch.no_grad()
def validation_loss(
    model: torch.nn.Module,
    validation_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    token_count = 0

    for input_ids, targets in validation_loader:
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else nullcontext()
        )
        with autocast_context:
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="sum",
            )

        loss_sum += loss.item()
        token_count += targets.numel()

    model.train(was_training)
    if token_count == 0:
        raise RuntimeError("validation loader did not produce any tokens")
    return loss_sum / token_count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved run on a complete token dataset")
    parser.add_argument(
        "--model", type=Path, required=True, help="saved run or migrated model directory",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        help="specific checkpoint, absolute or relative to the model directory",
    )
    parser.add_argument(
        "--dataset", type=Path, required=True, help="path to dataset to test",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model, _, sequence_length = load_model(args.model, device, checkpoint=args.checkpoint)

    validation_dataset = TokenBlockDataset(args.dataset.expanduser(), sequence_length)

    validation_loader = build_loader(validation_dataset, shuffle=False)

    val_loss = validation_loss(
        model,
        validation_loader,
        device,
    )

    print(f"Final loss: {val_loss:.6f}")


if __name__ == "__main__":
    main()
