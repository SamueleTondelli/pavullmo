"""Synchronous, atomic checkpoints for specialized trainers."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import torch
from loguru import logger

from config import CheckpointConfig, TrainerConfig


class CheckpointTrainer(Protocol):
    """State owned by a specialized trainer; no shared loop assumptions."""

    config: TrainerConfig
    global_step: int
    source_checkpoint: str

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...

FORMAT_VERSION = 1


def model_state_from_checkpoint(payload: Any) -> dict[str, torch.Tensor]:
    """Extract generation weights from current or legacy checkpoint payloads."""
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a dictionary")
    if "trainer_state" in payload:
        if payload.get("format_version") != FORMAT_VERSION:
            raise ValueError("unsupported Trainer checkpoint format")
        trainer_state = payload["trainer_state"]
        state = trainer_state.get("model") if isinstance(trainer_state, dict) else None
    else:
        state = payload.get("model_state_dict")
    if not isinstance(state, dict) or not state or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError("checkpoint is missing a valid model state dictionary")
    return state


def optimizer_parameter_names(
    model: torch.nn.Module, optimizers: dict[str, torch.optim.Optimizer],
) -> dict[str, list[list[str]]]:
    """Identify optimizer groups by stable, deduplicated model parameter names."""
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return {
        name: [[names[id(parameter)] for parameter in group["params"]]
               for group in optimizer.param_groups]
        for name, optimizer in optimizers.items()
    }


class CheckpointManager:
    """Own file I/O and retention; the trainer owns the serialized state.

    The specialized trainer enforces its save boundary and owns all state;
    this manager handles publication, retention and optional persistence.
    """

    def __init__(self, config: CheckpointConfig, directory: str | Path):
        self.config = CheckpointConfig.model_validate(
            config.to_dict(), context={"warn_unknown_config": False},
        )
        self.directory = Path(directory).expanduser().resolve()
        # Runtime hook for cloud persistence, independent of config/providers.
        self.after_save: Callable[[], None] | None = None

    def _checkpoints(self) -> list[Path]:
        paths = [path for path in self.directory.glob("step_*.pt")
                 if path.is_file() and re.fullmatch(r"step_[0-9]+\.pt", path.name)]
        # Save order also handles resuming an older step in an existing run.
        return sorted(paths, key=lambda path: (path.stat().st_mtime_ns, path.name))

    @property
    def latest_checkpoint(self) -> Path:
        paths = self._checkpoints()
        if not paths:
            raise FileNotFoundError(f"no checkpoints in {self.directory}")
        return paths[-1]

    def save(self, trainer: CheckpointTrainer) -> Path:
        path = self.directory / f"step_{trainer.global_step:08d}.pt"
        temporary: Path | None = None
        try:
            state = trainer.state_dict()
            if not self.config.save_training_state:
                state = {"model": state["model"]}
            payload = {
                "format_version": FORMAT_VERSION,
                "config_toml": trainer.config.to_toml(),
                "source_checkpoint": trainer.source_checkpoint,
                "full_state": self.config.save_training_state,
                "trainer_state": state,
            }
            self.directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self.directory, prefix=f".{path.name}.",
                suffix=".tmp", delete=False,
            ) as file:
                temporary = Path(file.name)
                torch.save(payload, file)
                file.flush()
                os.fsync(file.fileno())
            temporary.replace(path)
        except Exception:
            logger.exception("Could not save checkpoint {}", path)
            raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        # Prune only after publishing successfully; unrelated files are untouched.
        old = [item for item in self._checkpoints() if item != path]
        for item in old[:max(0, len(old) - self.config.keep_last + 1)]:
            item.unlink()
            logger.debug("Removed old checkpoint {}", item)
        logger.info("Saved checkpoint {}", path)
        if self.after_save is not None:
            self.after_save()
        return path

    @staticmethod
    def read(path: str | Path) -> dict[str, Any]:
        """Read on CPU without instantiating models, optimizers or the Trainer."""
        path = Path(path).expanduser()
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
                raise ValueError("unsupported Trainer checkpoint format")
            if not isinstance(payload.get("config_toml"), str):
                raise ValueError("checkpoint is missing its TOML configuration")
            if not isinstance(payload.get("trainer_state"), dict):
                raise ValueError("checkpoint is missing trainer state")
            if type(payload.get("full_state")) is not bool:
                raise ValueError("checkpoint is missing its full-state flag")
        except Exception:
            logger.exception("Could not read checkpoint {}", path)
            raise
        return payload

    def load(self, trainer: CheckpointTrainer, path: str | Path | None = None) -> None:
        path = self.latest_checkpoint if path is None else Path(path).expanduser().resolve()
        payload = self.read(path)
        if not payload["full_state"]:
            raise ValueError("resuming requires a full checkpoint saved with checkpoint.save_training_state=true")
        trainer.load_state_dict(payload["trainer_state"])
        trainer.source_checkpoint = str(path)
        logger.info("Loaded checkpoint {} at step {}", path, trainer.global_step)
