"""Config-driven AdamW or Muon/AdamW with synchronized LR schedules."""

from __future__ import annotations

import math
from typing import Any

import torch
from loguru import logger
from torch.optim.lr_scheduler import LambdaLR

from config import OptimizerConfig


class Optimizer:
    """Construct optimizers and schedules without reading process settings.

    The trainer passes the computed step horizon separately. Named
    optimizers/schedulers remain accessible for diagnostics and checkpoints.
    Gradient clearing, accumulation and clipping belong to the training loop.
    """

    def __init__(self, config: OptimizerConfig, model: torch.nn.Module, *, total_steps: int):
        self.config = OptimizerConfig.model_validate(
            config.to_dict(), context={"warn_unknown_config": False},
        )
        config = self.config
        if type(total_steps) is not int or total_steps <= 0:
            raise ValueError("total_steps must be a positive integer")
        if config.schedule.warmup_steps >= total_steps:
            raise ValueError("schedule.warmup_steps must be smaller than total_steps")
        self.total_steps = total_steps

        self.parameter_groups = _parameter_groups(config, model)
        self.optimizers: dict[str, torch.optim.Optimizer] = {
            "adamw": torch.optim.AdamW(
                self.parameter_groups["adamw"], lr=config.lr,
                betas=(config.adamw.beta1, config.adamw.beta2),
                eps=config.adamw.eps, weight_decay=0.0, fused=config.adamw.fused,
            ),
        }
        if config.kind == "muon_adamw":
            muon = config.muon
            self.optimizers["muon"] = torch.optim.Muon(
                self.parameter_groups["muon"],
                lr=config.lr * muon.lr_multiplier,
                weight_decay=muon.weight_decay, momentum=muon.momentum,
                nesterov=muon.nesterov, eps=muon.eps,
                ns_steps=muon.ns_iters, ns_coefficients=tuple(muon.ns_coefficients),
                adjust_lr_fn=muon.adjust_lr_fn,
            )
        self.schedulers = {
            name: _build_schedule(optimizer, config, total_steps)
            for name, optimizer in self.optimizers.items()
        }
        logger.info(
            "Configured {} for {} optimizer steps", config.kind, total_steps,
        )

    def step_optimizer(self) -> None:
        """Apply the current gradients to all assigned parameter groups."""
        for optimizer in self.optimizers.values():
            optimizer.step()

    def step_schedule(self) -> None:
        """Advance every scheduler after the completed optimizer update."""
        for scheduler in self.schedulers.values():
            scheduler.step()


def _parameter_groups(
    config: OptimizerConfig, model: torch.nn.Module,
) -> dict[str, list[dict[str, Any]]]:
    embeddings = {
        id(parameter) for module in model.modules() if isinstance(module, torch.nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    norms = {
        id(parameter) for module in model.modules()
        if isinstance(module, (torch.nn.RMSNorm, torch.nn.LayerNorm))
        for parameter in module.parameters(recurse=False)
    }
    decay, no_decay, norm, muon = [], [], [], []
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("the model has no trainable parameters")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in norms or "_norm." in name or name == "norm.weight":
            norm.append(parameter)
        elif (config.kind == "muon_adamw" and parameter.ndim == 2
              and id(parameter) not in embeddings and name != "embeddings.weight"):
            muon.append(parameter)
        elif parameter.ndim >= 2 and id(parameter) not in embeddings and name != "embeddings.weight":
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    groups = {
        "adamw": [
            {"params": decay, "weight_decay": config.adamw.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
            {"params": norm, "weight_decay": 0.0, "lr": config.lr * config.norm_lr_multiplier},
        ],
    }
    if config.kind == "muon_adamw":
        if not muon:
            raise ValueError("muon_adamw requires at least one eligible hidden weight matrix")
        groups["muon"] = [{"params": muon, "weight_decay": config.muon.weight_decay}]
    assigned = [id(parameter) for entries in groups.values()
                for group in entries for parameter in group["params"]]
    if len(assigned) != len(set(assigned)) or set(assigned) != {id(p) for p in trainable}:
        raise RuntimeError("each trainable parameter must belong to exactly one optimizer group")
    return groups


def _build_schedule(optimizer: torch.optim.Optimizer, config: OptimizerConfig, total_steps: int) -> LambdaLR:
    schedule = config.schedule
    post_warmup_steps = total_steps - schedule.warmup_steps
    minimum_ratio = schedule.min_lr / config.lr
    decay_steps = max(1, math.ceil(post_warmup_steps * schedule.decay_fraction))
    stable_steps = post_warmup_steps - decay_steps

    def multiplier(step: int) -> float:
        if schedule.warmup_steps and step < schedule.warmup_steps:
            return (step + 1) / schedule.warmup_steps
        if schedule.kind == "cosine":
            progress = (step - schedule.warmup_steps) / max(1, post_warmup_steps - 1)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return minimum_ratio + (1.0 - minimum_ratio) * cosine
        decay_start = schedule.warmup_steps + stable_steps
        if step < decay_start:
            return 1.0
        if decay_steps == 1:
            return minimum_ratio
        progress = (step - decay_start) / (decay_steps - 1)
        progress = min(max(progress, 0.0), 1.0)
        return 1.0 - (1.0 - minimum_ratio) * progress

    return LambdaLR(optimizer, lr_lambda=multiplier)
