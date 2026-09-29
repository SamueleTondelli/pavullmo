"""Optimizer-boundary checkpoint state shared by the two pretraining scripts."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Sampler


class EpochBatchSampler(Sampler[list[int]]):
    """Recreate an epoch's shuffle and skip consumed batches without reading them.

    The cursor is set by the training loop, never by iteration: DataLoader worker
    prefetching must not advance the position recorded in a checkpoint.
    """

    def __init__(self, dataset_size: int, batch_size: int, seed: int) -> None:
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.start_batch = 0

    def set_epoch(self, epoch: int, start_batch: int = 0) -> None:
        if epoch < 0 or not 0 <= start_batch <= self.dataset_size // self.batch_size:
            raise ValueError("invalid training data cursor")
        self.epoch = epoch
        self.start_batch = start_batch

    def __len__(self) -> int:
        return self.dataset_size // self.batch_size - self.start_batch

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        for batch in range(self.start_batch, self.dataset_size // self.batch_size):
            start = batch * self.batch_size
            yield indices[start : start + self.batch_size]


def optimizer_parameter_names(
    model: torch.nn.Module, optimizers: dict[str, torch.optim.Optimizer]
) -> dict[str, list[list[str]]]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return {
        name: [
            [names[id(parameter)] for parameter in group["params"]]
            for group in optimizer.param_groups
        ]
        for name, optimizer in optimizers.items()
    }


def capture_training_state(
    model: torch.nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    schedulers: dict[str, LambdaLR],
    *,
    global_step: int,
    total_steps: int,
    tokens_seen: int,
    clipped_steps: int,
    optimizer_steps_per_epoch: int,
    dataset_metadata: dict[str, Any],
    data_generator: torch.Generator,
    validation_generator: torch.Generator,
) -> dict[str, Any]:
    """Capture state after an optimizer/scheduler step, with no pending update."""
    next_epoch, next_step_in_epoch = divmod(global_step, optimizer_steps_per_epoch)
    return {
        "format_version": 1,
        "optimizer_types": {
            name: type(optimizer).__name__ for name, optimizer in optimizers.items()
        },
        "optimizer_parameter_names": optimizer_parameter_names(model, optimizers),
        "optimizer_state_dicts": {
            name: optimizer.state_dict() for name, optimizer in optimizers.items()
        },
        "scheduler_state_dicts": {
            name: scheduler.state_dict() for name, scheduler in schedulers.items()
        },
        # These are the last step's clipped gradients, already applied by the
        # optimizers. The loop clears them before the next accumulation window.
        "gradients": {
            name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for name, parameter in model.named_parameters()
        },
        "global_step": global_step,
        "total_steps": total_steps,
        "tokens_seen": tokens_seen,
        "clipped_steps": clipped_steps,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "next_epoch": next_epoch,
        "next_step_in_epoch": next_step_in_epoch,
        "dataset_metadata": dataset_metadata,
        "rng_state": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        },
        "data_generator_state": data_generator.get_state(),
        "validation_generator_state": validation_generator.get_state(),
    }


# Only settings that determine the model's parameter structure must match.
# Context length, RoPE base, dropout, optimizer, schedule, and data settings may
# change because RESUME_CHECKPOINT starts a new training phase.
ARCHITECTURE_SETTINGS = {
    "VOCAB_SIZE", "N_BLOCKS", "EMBED_DIM", "ATTN_HEADS", "FFN_DIM",
    "QK_NORM", "SPLIT_QKV_PROJECTIONS", "CANON_LAYERS",
}


def restore_training_state(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    *,
    hyperparameters: dict[str, object],
) -> dict[str, Any]:
    """Load weights, gradients, and optimizer history for a new training phase.

    Current optimizer options override the saved options, while moment estimates,
    momentum buffers, and optimizer step counters are retained. Schedulers, data
    iterators, RNG streams, and run counters start fresh under the current config.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("training_state"), dict
    ):
        raise ValueError(
            "RESUME_CHECKPOINT requires a full checkpoint saved with SAVE_TRAINING_STATE=true"
        )
    state = checkpoint["training_state"]
    if checkpoint.get("format_version") != 2 or state.get("format_version") != 1:
        raise ValueError("unsupported resumable checkpoint format")
    saved_hyperparameters = checkpoint.get("hyperparameters", {})
    mismatches = sorted(
        name for name in ARCHITECTURE_SETTINGS
        if saved_hyperparameters.get(name) != hyperparameters.get(name)
    )
    if mismatches:
        raise ValueError(f"resume architecture differs from checkpoint: {', '.join(mismatches)}")
    optimizer_types = {name: type(optimizer).__name__ for name, optimizer in optimizers.items()}
    if state["optimizer_types"] != optimizer_types:
        raise ValueError("resume optimizer types differ from checkpoint")
    if state["optimizer_parameter_names"] != optimizer_parameter_names(model, optimizers):
        raise ValueError("resume optimizer parameter groups differ from checkpoint")
    if set(state["optimizer_state_dicts"]) != set(optimizers):
        raise ValueError("checkpoint optimizer state is incomplete")
    global_step = state["global_step"]
    if (
        not isinstance(global_step, int)
        or global_step < 0
        or global_step != checkpoint["global_step"]
    ):
        raise ValueError("invalid checkpoint global step")
    parameters = dict(model.named_parameters())
    if set(state["gradients"]) != set(parameters):
        raise ValueError("checkpoint gradient parameter names differ from model")
    for name, gradient in state["gradients"].items():
        if gradient is not None and (
            gradient.shape != parameters[name].shape
            or gradient.dtype != parameters[name].dtype
        ):
            raise ValueError(f"checkpoint gradient shape or dtype differs for {name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    for name, optimizer in optimizers.items():
        # load_state_dict restores both buffers and the OLD group options. Keep
        # the newly configured options so saved LRs/betas/decay cannot override
        # this phase's environment, including norm/Muon LR multipliers.
        current_options = [
            copy.deepcopy({key: value for key, value in group.items() if key != "params"})
            for group in optimizer.param_groups
        ]
        optimizer.load_state_dict(state["optimizer_state_dicts"][name])
        for group, options in zip(optimizer.param_groups, current_options):
            group_parameters = group["params"]
            group.clear()
            group.update(options)
            group["params"] = group_parameters
    for name, parameter in parameters.items():
        gradient = state["gradients"][name]
        parameter.grad = None if gradient is None else gradient.to(parameter.device).clone()
    # Release the loaded CPU copies once restoration is complete. The saved step
    # is provenance only; this phase's run and scheduler counters start at zero.
    return {"global_step": global_step}
