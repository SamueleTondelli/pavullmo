"""Common component setup and one microbatch's forward/backward operation."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Self

import torch
import torch._functorch.config as aot_config
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from config import TrainerConfig, configure_logging
from config.serialization import dump_toml
from model import DecoderTransformer

from .checkpoint import CheckpointManager
from .optimizer import Optimizer


def _select_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("BF16 training requires a CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA GPU does not support BF16")
    return torch.device("cuda")


class GenericTrainer[ConfigT: TrainerConfig]:
    """Initialize shared components without defining a training algorithm.

    The specialized trainer supplies its typed config, seed and computed
    optimizer horizon. It owns datasets, losses, accumulation, stepping,
    evaluation, progress, and checkpoint save/load behavior.
    """

    def __init__(self, config: ConfigT, *, total_steps: int, seed: int):
        self.config = type(config).model_validate(
            config.to_dict(),
            context={"warn_unknown_config": False},
        )
        self.device = _select_device()
        configure_logging(self.config.logging)
        settings = self.config.to_dict()
        for name in TrainerConfig.model_fields:
            logger.info(
                "GenericTrainer config.{}: {}",
                name,
                dump_toml({name: settings[name]}, compact=True),
            )
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.total_steps = total_steps
        self.output_dir = Path(self.config.runtime.output_dir).expanduser().resolve()
        self.run_dir = self.output_dir / self.config.name
        self.model = DecoderTransformer(self.config.model).to(
            self.device, dtype=torch.float32
        )
        parameter_count = sum(
            parameter.numel() for parameter in self.model.parameters()
        )
        trainable_parameter_count = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        logger.info(
            "Model parameters: {:,} total, {:,} trainable",
            parameter_count,
            trainable_parameter_count,
        )
        self.optimizer = Optimizer(
            self.config.optimizer, self.model, total_steps=total_steps
        )
        self.train_model = (
            torch.compile(
                self.model, mode=self.config.runtime.compile_mode, dynamic=False
            )
            if self.config.runtime.compile_model
            else self.model
        )
        self.checkpoints = CheckpointManager(
            self.config.checkpoint, self.run_dir / "checkpoints"
        )
        self.config.save(self.run_dir / "config.toml")
        self.writer = SummaryWriter(
            log_dir=self.run_dir / "tensorboard",
            flush_secs=self.config.logging.tensorboard_flush_secs,
        )
        self.writer.add_text(
            "config/resolved", f"```toml\n{self.config.to_toml()}\n```", 0
        )
        self.writer.add_scalar("model/parameter_count", parameter_count, 0)

    def forward_backward(
        self,
        microbatch: torch.Tensor | Mapping[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        *,
        loss_kwargs: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        """Run one forward and backward, returning the detached scalar loss.

        Inputs and loss arguments must already be on the appropriate device.
        A tensor is passed positionally; a mapping is expanded as model kwargs.
        loss_fn(outputs, **loss_kwargs) must supply any required loss weighting
        or accumulation scaling. Gradients are neither cleared nor stepped.
        """
        if self.config.runtime.compile_model:
            torch.compiler.cudagraph_mark_step_begin()
        # Compilation is lazy and backward may compile separately. Keep the
        # budget active for both, then restore it for other trainers/callers.
        with aot_config.patch(
            activation_memory_budget=self.config.runtime.act_checkpoint_mem_budget
        ):
            with torch.autocast(
                self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
            ):
                outputs = (
                    self.train_model(**microbatch)
                    if isinstance(microbatch, Mapping)
                    else self.train_model(microbatch)
                )
                loss = loss_fn(outputs, **(loss_kwargs or {}))
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                raise ValueError("loss_fn must return a scalar tensor")
            loss.backward()
        return loss.detach()

    def close(self) -> None:
        self.writer.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
