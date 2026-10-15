"""Single-GPU pretraining and complete state restoration."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import signal
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from typing import Any, Self

import torch
import torch.nn.functional as F
from loguru import logger
from pydantic import BaseModel
from torch.utils.tensorboard import SummaryWriter

from config import PretrainingConfig, TrainerConfig, configure_logging
from config.serialization import dump_toml

from .checkpoint import CheckpointManager, optimizer_parameter_names
from .data import TokenBlockDataset, TrainDataLoader, ValidationDataLoader
from .diagnostics import gradient_group_name
from .generic_trainer import GenericTrainer


def _known_settings(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return {name: _known_settings(getattr(value, name)) for name in type(value).model_fields}
    if isinstance(value, list):
        return [_known_settings(item) for item in value]
    return value


def _training_signature(config: PretrainingConfig) -> dict[str, Any]:
    # Paths are checked through artifact identities, independently of mounts.
    # Runtime, logging, checkpoint retention and the label may change on resume.
    return {name: _known_settings(getattr(config, name))
            for name in ("model", "optimizer", "training", "evaluation")}


def _matches_training_signature(saved: Any, config: PretrainingConfig) -> bool:
    # Older refactor checkpoints duplicated the computed horizon in the config
    # signature. It remains checked independently through progress.total_steps.
    if isinstance(saved, dict):
        optimizer = saved.get("optimizer")
        if isinstance(optimizer, dict) and isinstance(optimizer.get("schedule"), dict):
            saved = {**saved, "optimizer": {
                **optimizer, "schedule": {name: value for name, value in optimizer["schedule"].items()
                                          if name != "total_steps"},
            }}
    # Refactor checkpoints from the MHA-only branch predate this field.
    if isinstance(saved, dict) and isinstance(saved.get("model"), dict):
        if "num_kv_heads" not in saved["model"]:
            saved = {**saved, "model": {**saved["model"], "num_kv_heads": 0}}
    return saved == _training_signature(config)


def _dataset_identity(dataset: TokenBlockDataset) -> str:
    metadata = dataset.metadata
    tokenizer = metadata.get("tokenizer", {})
    if not isinstance(tokenizer.get("sha256"), str) or not tokenizer["sha256"]:
        raise ValueError("dataset metadata must contain a tokenizer SHA256")
    shards = []
    for shard in metadata["shards"]:
        if not isinstance(shard.get("sha256"), str) or not shard["sha256"]:
            raise ValueError("dataset metadata must contain a SHA256 for every shard")
        shards.append({"tokens": shard["tokens"], "sha256": shard["sha256"]})
    identity = {
        "storage": metadata["storage"], "token_count": dataset.total_tokens,
        "tokenizer_sha256": tokenizer["sha256"], "vocab_size": tokenizer.get("vocab_size"),
        "shards": shards, "fixed_evaluation_sample": metadata.get("fixed_evaluation_sample"),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


class PretrainingTrainer(GenericTrainer[PretrainingConfig]):
    """Own the components, training loop and mutable progress of one run."""

    def __init__(self, config: PretrainingConfig):
        self.config = PretrainingConfig.model_validate(
            config.to_dict(), context={"warn_unknown_config": False},
        )
        # Loader setup logs before the common component constructor can run:
        # this trainer must inspect its data to compute the optimizer horizon.
        configure_logging(self.config.logging)
        self.dataset_dir = Path(self.config.runtime.dataset_dir).expanduser().resolve()
        self.runs_csv = Path(self.config.runtime.runs_csv).expanduser().resolve()
        self.train_loader = TrainDataLoader(self.dataset_dir / self.config.train_dataset.path, self.config)
        self.validation_loader = ValidationDataLoader(self.dataset_dir / self.config.validation_dataset.path, self.config)
        for loader in (self.train_loader, self.validation_loader):
            vocabulary = loader.dataset.metadata.get("tokenizer", {}).get("vocab_size")
            if vocabulary != self.config.model.vocab_size:
                raise ValueError("dataset tokenizer vocabulary must match model.vocab_size")
        train_tokenizer = self.train_loader.dataset.metadata["tokenizer"].get("sha256")
        validation_tokenizer = self.validation_loader.dataset.metadata["tokenizer"].get("sha256")
        if train_tokenizer != validation_tokenizer:
            raise ValueError("training and validation artifacts must use the same tokenizer")
        self.dataset_identities = {
            "train": _dataset_identity(self.train_loader.dataset),
            "validation": _dataset_identity(self.validation_loader.dataset),
        }

        self.optimizer_steps_per_epoch = len(self.train_loader) // self.config.training.grad_accum_steps
        available = self.optimizer_steps_per_epoch * self.config.training.epochs
        self.total_steps = min(available, self.config.training.max_steps) if self.config.training.max_steps else available
        super().__init__(
            self.config, total_steps=self.total_steps, seed=self.config.training.seed,
        )
        self.global_step = 0
        self.epoch = 0
        self.next_batch = 0
        self.tokens_seen = 0
        self.clipped_steps = 0
        self.train_loss: float | None = None
        self.validation_loss: float | None = None
        self.best_validation_loss: float | None = None
        self.last_validation_step = 0
        self.source_checkpoint = ""
        self._at_update_boundary = True
        self._running = False
        self._stop_requested = False
        self._epoch_loader_rng: torch.Tensor | None = None

        for name, value in self.config.to_dict().items():
            if name not in TrainerConfig.model_fields:
                logger.info("PretrainingTrainer config.{}: {}", name,
                            dump_toml({name: value}, compact=True))
        logger.info("Configured PretrainingTrainer '{}' for {} optimizer steps", self.config.name, self.total_steps)

    def _assert_update_boundary(self) -> None:
        accumulation = self.config.training.grad_accum_steps
        if (not self._at_update_boundary
                or not 0 <= self.global_step <= self.total_steps
                or self.next_batch % accumulation
                or not 0 <= self.next_batch < self.optimizer_steps_per_epoch * accumulation
                or self.global_step != self.epoch * self.optimizer_steps_per_epoch + self.next_batch // accumulation
                or any(scheduler.last_epoch != self.global_step for scheduler in self.optimizer.schedulers.values())):
            raise RuntimeError("PretrainingTrainer requires a completed optimizer and scheduler update; restore a checkpoint after a failed update")

    @staticmethod
    def _iterator(loader, start_rng: torch.Tensor | None = None):
        """Advance the loader RNG once per logical iterator, even with persistent workers.

        Rebuilding an iterator uses its original worker seed without consuming
        the saved RNG stream again. Token-shard reads themselves are deterministic.
        """
        current = loader.generator.get_state()
        if start_rng is None:
            start_rng = current
            torch.empty((), dtype=torch.int64).random_(generator=loader.generator)
            current = loader.generator.get_state()
        loader.generator.set_state(start_rng)
        try:
            return iter(loader)
        finally:
            loader.generator.set_state(current)

    def _train_iterator(self):
        self.train_loader.set_epoch(self.epoch, self.next_batch)
        if self._epoch_loader_rng is None:
            self._epoch_loader_rng = self.train_loader.generator.get_state()
            torch.empty((), dtype=torch.int64).random_(generator=self.train_loader.generator)
        return self._iterator(self.train_loader, self._epoch_loader_rng)

    def request_stop(self) -> None:
        """Ask train() to save and return after its current complete update."""
        self._stop_requested = True

    @contextmanager
    def _stop_signals(self):
        # Signal handlers can only be installed by the main thread. Wrappers
        # running training in a worker thread can call request_stop() instead.
        previous = {}
        if threading.current_thread() is threading.main_thread():
            def stop(signum, frame):
                self.request_stop()
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, stop)
        try:
            yield
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)

    def _diagnostics(self) -> dict[str, float]:
        groups = {}
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            group = gradient_group_name(name)
            squared = parameter.grad.detach().float().norm().square()
            groups[group] = groups.get(group, 0) + squared
            if group in {"attention_query", "attention_key", "attention_value"}:
                groups["attention_qkv"] = groups.get("attention_qkv", 0) + squared
        values = {f"gradient_groups/{name}": value.sqrt().item() for name, value in groups.items()}
        parameters = list(self.model.parameters())
        norm = torch.stack([parameter.detach().norm() for parameter in parameters]).norm().item()
        values["diagnostics/parameter_norm"] = norm
        values["diagnostics/parameter_rms"] = norm / math.sqrt(sum(p.numel() for p in parameters))
        values["diagnostics/embedding_norm"] = self.model.embeddings.weight.detach().norm().item()
        return values

    def compute_loss(
        self, logits: torch.Tensor, *, targets: torch.Tensor, metrics: dict[str, float],
    ) -> torch.Tensor:
        """Pretraining objective, scaled for this trainer's accumulation group."""
        training = self.config.training
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        z_loss = (logits.float().logsumexp(dim=-1).square().mean()
                  if training.z_loss_coefficient else loss.new_zeros(()))
        objective = loss + training.z_loss_coefficient * z_loss
        if not torch.isfinite(objective).item():
            raise FloatingPointError(f"non-finite training objective at step {self.global_step + 1}")
        metrics.update(loss=loss.detach().item(), objective=objective.detach().item(),
                       z_loss=z_loss.detach().item())
        return objective / training.grad_accum_steps

    def _update(self, iterator) -> dict[str, float]:
        """Consume one complete accumulation group and commit its progress."""
        self._at_update_boundary = False
        self.model.zero_grad(set_to_none=True)
        training = self.config.training
        loss_sum = objective_sum = z_loss_sum = 0.0
        tokens = 0
        for microbatch in range(training.grad_accum_steps):
            inputs, targets = next(iterator)
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            loss_metrics = {}
            self.forward_backward(
                inputs, self.compute_loss,
                loss_kwargs={"targets": targets, "metrics": loss_metrics},
            )
            loss_sum += loss_metrics["loss"]
            objective_sum += loss_metrics["objective"]
            z_loss_sum += loss_metrics["z_loss"]
            tokens += targets.numel()
            if self.config.runtime.compile_model and microbatch == 0:
                # Compiled backward can return gradients backed by CUDA-graph
                # storage. Own the accumulation buffers before the next replay.
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        parameter.grad = parameter.grad.clone()

        next_step = self.global_step + 1
        metrics = self._diagnostics() if (next_step == 1 or next_step == self.total_steps
                  or next_step % self.config.logging.diagnostics_interval == 0) else {}
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), training.max_grad_norm, error_if_nonfinite=True,
        ).item()
        for name, optimizer in self.optimizer.optimizers.items():
            for index, group in enumerate(optimizer.param_groups):
                metrics[f"learning_rates/{name}/group_{index}"] = group["lr"]
        metrics["train/learning_rate"] = self.optimizer.optimizers["adamw"].param_groups[0]["lr"]
        metrics["train/norm_learning_rate"] = self.optimizer.optimizers["adamw"].param_groups[2]["lr"]
        if "muon" in self.optimizer.optimizers:
            metrics["train/muon_learning_rate"] = self.optimizer.optimizers["muon"].param_groups[0]["lr"]
        self.optimizer.step_optimizer()
        self.optimizer.step_schedule()
        self.model.zero_grad(set_to_none=True)
        self.global_step = next_step
        self.next_batch += training.grad_accum_steps
        self.tokens_seen += tokens
        self.clipped_steps += int(gradient_norm > training.max_grad_norm)
        self.train_loss = loss_sum / training.grad_accum_steps
        if self.next_batch == self.optimizer_steps_per_epoch * training.grad_accum_steps:
            self.epoch += 1
            self.next_batch = 0
            self._epoch_loader_rng = None
        self._at_update_boundary = True
        metrics.update({
            "train/loss": self.train_loss,
            "train/objective": objective_sum / training.grad_accum_steps,
            "train/z_loss": z_loss_sum / training.grad_accum_steps,
            "train/z_loss_contribution": training.z_loss_coefficient * z_loss_sum / training.grad_accum_steps,
            "train/gradient_norm": gradient_norm,
            "train/clip_fraction": self.clipped_steps / self.global_step,
            "train/gradient_clipped": float(gradient_norm > training.max_grad_norm),
            "train/tokens_seen": self.tokens_seen,
        })
        return metrics

    @torch.no_grad()
    def validate(self) -> float:
        """Evaluate ordered batches, weighting cross-entropy by target tokens."""
        self._assert_update_boundary()
        was_training = self.model.training
        self.model.eval()
        loss_sum = 0.0
        tokens = 0
        try:
            iterator = self._iterator(self.validation_loader)
            for inputs, targets in islice(iterator, self.config.evaluation.steps):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                # Eager evaluation handles the partial final batch without a
                # second compiled graph or disturbing training CUDA graphs.
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    logits = self.model(inputs)
                    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="sum")
                loss_sum += loss.item()
                tokens += targets.numel()
        finally:
            self.model.train(was_training)
        if not tokens:
            raise RuntimeError("validation loader did not produce any tokens")
        value = loss_sum / tokens
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite validation loss at step {self.global_step}")
        self.validation_loss = value
        self.best_validation_loss = min(self.best_validation_loss, value) if self.best_validation_loss is not None else value
        self.last_validation_step = self.global_step
        self.writer.add_scalar("validation/loss", value, self.global_step)
        self.writer.flush()
        logger.info("step={}/{} validation_loss={:.6f}", self.global_step, self.total_steps, value)
        return value

    def _append_result(self, checkpoint: Path) -> None:
        # Retain the historical registry columns. dataset_variant now holds
        # the explicit training artifact path; config contains single-line TOML.
        path = self.runs_csv
        fields = ["experiment_name", "dataset_variant", "train_loss", "validation_loss", "train_script", "model_path", "config"]
        row = dict(zip(fields, [self.config.name, self.config.train_dataset.path, format(self.train_loss, ".10g"),
                               format(self.validation_loss, ".10g"), "components.pretraining_trainer.PretrainingTrainer",
                               str(checkpoint), self.config.to_string()]))
        existing = path.exists() and path.stat().st_size > 0
        if existing:
            with path.open(encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames != fields:
                    raise ValueError(f"run registry columns differ: {path}")
                # Avoid appending a duplicate when completing a run whose row
                # was written before compact config strings were introduced.
                config_strings = {row["config"], self.config.to_toml()}
                if any(item["model_path"] == row["model_path"] and item["config"] in config_strings for item in reader):
                    return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            if not existing:
                writer.writeheader()
            writer.writerow(row)
        logger.info("Recorded completed run in {}", path)

    def train(self) -> None:
        """Run to the configured horizon, or save and return on request_stop().

        SIGINT/SIGTERM request a graceful stop when running in the main thread.
        Failed/incomplete updates cannot be saved or continued without restoring.
        The writer stays open until close(), allowing validation and inspection.
        """
        if self._running:
            raise RuntimeError("PretrainingTrainer.train() is already running")
        self._assert_update_boundary()
        self._running = True
        self.train_model.train()
        iterator = None
        started = None
        window_step = self.global_step
        window_tokens = self.tokens_seen

        def log_throughput() -> None:
            nonlocal started, window_step, window_tokens
            if started is None:
                return
            # Drain queued optimizer work only at measurement boundaries.
            # Allocator statistics themselves do not require synchronization.
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - started
            self.writer.add_scalar("train/tokens_per_second",
                                   (self.tokens_seen - window_tokens) / elapsed, self.global_step)
            started = None
            window_step = self.global_step
            window_tokens = self.tokens_seen

        try:
            with self._stop_signals():
                if (self.device.type == "cuda" and self.global_step < self.total_steps
                        and not self._stop_requested):
                    # Exclude component setup or checkpoint restoration from timing.
                    torch.cuda.synchronize(self.device)
                while self.global_step < self.total_steps and not self._stop_requested:
                    if started is None:
                        started = time.perf_counter()
                    if iterator is None:
                        iterator = self._train_iterator()
                    metrics = self._update(iterator)
                    if self.next_batch == 0:
                        iterator = None
                    if self.device.type == "cuda":
                        metrics["runtime/cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device)
                    for name, value in metrics.items():
                        self.writer.add_scalar(name, value, self.global_step)
                    logger.info("step={}/{} train_loss={:.6f} gradient_norm={:.6f} lr={:.6g}",
                                self.global_step, self.total_steps, self.train_loss,
                                metrics["train/gradient_norm"], metrics["train/learning_rate"])
                    validate = (self.global_step % self.config.evaluation.interval_steps == 0
                                or self.global_step == self.total_steps)
                    interval = self.config.checkpoint.interval_steps
                    checkpoint = interval and self.global_step % interval == 0 and self.global_step < self.total_steps
                    if (self.global_step - window_step >= self.config.logging.throughput_interval_steps
                            or validate or checkpoint or self._stop_requested):
                        log_throughput()
                    if validate:
                        self.validate()
                    if checkpoint:
                        self.save_checkpoint()
                # A stop may arrive between updates rather than during one.
                log_throughput()
                if self.global_step == self.total_steps:
                    if self.validation_loss is None or self.last_validation_step != self.global_step:
                        self.validate()
                    path = self.save_checkpoint()
                    self._append_result(path)
                    logger.success("Completed '{}' after {} optimizer steps", self.config.name, self.global_step)
                else:
                    self.save_checkpoint()
                    logger.info("Stopped '{}' at step {}; the checkpoint can resume this run", self.config.name, self.global_step)
        except BaseException:
            logger.exception("Training '{}' failed at step {}; resume from a saved checkpoint", self.config.name, self.global_step)
            raise
        finally:
            self._running = False
            self._stop_requested = False
            self.writer.flush()

    def state_dict(self) -> dict[str, Any]:
        """Capture only state, never Python instances or a compiled model."""
        self._assert_update_boundary()
        return {
            "model": self.model.state_dict(),
            "model_training": self.model.training,
            "optimizers": {name: opt.state_dict() for name, opt in self.optimizer.optimizers.items()},
            "schedulers": {name: scheduler.state_dict() for name, scheduler in self.optimizer.schedulers.items()},
            "optimizer_parameter_names": optimizer_parameter_names(self.model, self.optimizer.optimizers),
            "training_signature": _training_signature(self.config),
            "dataset_identities": dict(self.dataset_identities),
            "execution": {"device_type": self.device.type, "torch_version": str(torch.__version__)},
            "progress": {
                "global_step": self.global_step, "epoch": self.epoch, "next_batch": self.next_batch,
                "tokens_seen": self.tokens_seen, "clipped_steps": self.clipped_steps,
                "train_loss": self.train_loss, "validation_loss": self.validation_loss,
                "best_validation_loss": self.best_validation_loss,
                "last_validation_step": self.last_validation_step,
                "total_steps": self.total_steps, "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
            },
            "rng": {
                "python": random.getstate(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
                "train_loader": self.train_loader.generator.get_state(),
                "validation_loader": self.validation_loader.generator.get_state(),
                "epoch_iterator": self._epoch_loader_rng,
            },
        }

    def _validate_state(self, state: dict[str, Any]) -> None:
        required = {
            "model", "model_training", "optimizers", "schedulers", "optimizer_parameter_names",
            "training_signature", "dataset_identities", "execution", "progress", "rng",
        }
        if not isinstance(state, dict) or not required <= state.keys():
            raise ValueError("checkpoint trainer state is incomplete")
        if not _matches_training_signature(state["training_signature"], self.config):
            raise ValueError("resume training settings differ from the checkpoint")
        if state["dataset_identities"] != self.dataset_identities:
            raise ValueError("resume dataset identities differ from the checkpoint")
        execution = {"device_type": self.device.type, "torch_version": str(torch.__version__)}
        if state["execution"] != execution:
            raise ValueError("resume requires the same device type and PyTorch version")
        expected_names = optimizer_parameter_names(self.model, self.optimizer.optimizers)
        if state["optimizer_parameter_names"] != expected_names:
            raise ValueError("resume optimizer parameter assignments differ from the checkpoint")
        if set(state["optimizers"]) != set(self.optimizer.optimizers) or set(state["schedulers"]) != set(self.optimizer.schedulers):
            raise ValueError("checkpoint optimizer/scheduler state is incomplete")
        for name, optimizer in self.optimizer.optimizers.items():
            saved = state["optimizers"][name]
            if not isinstance(saved, dict) or not isinstance(saved.get("state"), dict):
                raise ValueError(f"checkpoint optimizer state is invalid: {name}")
            groups = saved.get("param_groups", [])
            if len(groups) != len(optimizer.param_groups):
                raise ValueError(f"checkpoint optimizer groups differ: {name}")
            for previous, current in zip(groups, optimizer.param_groups):
                if len(previous.get("params", [])) != len(current["params"]):
                    raise ValueError(f"checkpoint optimizer groups differ: {name}")
                for index, parameter in zip(previous["params"], current["params"]):
                    for buffer_name in ("exp_avg", "exp_avg_sq", "momentum_buffer"):
                        buffer = saved["state"].get(index, {}).get(buffer_name)
                        if buffer is not None and (not isinstance(buffer, torch.Tensor)
                                or buffer.shape != parameter.shape or buffer.dtype != parameter.dtype):
                            raise ValueError(f"checkpoint optimizer buffer differs: {name}.{buffer_name}")
            scheduler = state["schedulers"][name]
            if not isinstance(scheduler, dict) or type(scheduler.get("last_epoch")) is not int:
                raise ValueError(f"checkpoint scheduler state is invalid: {name}")
        model_state = self.model.state_dict()
        if set(state["model"]) != set(model_state):
            raise ValueError("checkpoint model parameters differ")
        for name, value in state["model"].items():
            if not isinstance(value, torch.Tensor) or value.shape != model_state[name].shape or value.dtype != model_state[name].dtype:
                raise ValueError(f"checkpoint tensor shape/dtype differs for {name}")
        if type(state["model_training"]) is not bool:
            raise ValueError("checkpoint model training mode is invalid")

        progress = state["progress"]
        if not isinstance(progress, dict):
            raise ValueError("checkpoint progress is invalid")
        for name in ("global_step", "epoch", "next_batch", "tokens_seen", "clipped_steps"):
            if type(progress.get(name)) is not int or progress[name] < 0:
                raise ValueError(f"invalid checkpoint progress: {name}")
        if progress.get("total_steps") != self.total_steps or progress.get("optimizer_steps_per_epoch") != self.optimizer_steps_per_epoch:
            raise ValueError("checkpoint training duration differs")
        if progress["global_step"] > self.total_steps or progress["epoch"] > self.config.training.epochs:
            raise ValueError("checkpoint progress exceeds the training duration")
        if progress["next_batch"] > len(self.train_loader.dataset) // self.config.training.batch_size:
            raise ValueError("checkpoint data cursor exceeds the epoch")
        accumulation = self.config.training.grad_accum_steps
        if (progress["next_batch"] % accumulation
                or progress["next_batch"] >= self.optimizer_steps_per_epoch * accumulation
                or progress["global_step"] != progress["epoch"] * self.optimizer_steps_per_epoch + progress["next_batch"] // accumulation
                or progress["clipped_steps"] > progress["global_step"]
                or progress["tokens_seen"] != progress["global_step"] * self.config.training.batch_size * self.config.model.seq_len * accumulation):
            raise ValueError("checkpoint progress does not describe a completed update")
        if any(saved["last_epoch"] != progress["global_step"] for saved in state["schedulers"].values()):
            raise ValueError("checkpoint schedulers do not match the completed step")
        validation_step = progress.get("last_validation_step", 0)
        if type(validation_step) is not int or not 0 <= validation_step <= progress["global_step"]:
            raise ValueError("invalid checkpoint last validation step")
        for name in ("train_loss", "validation_loss", "best_validation_loss"):
            if name not in progress or (progress[name] is not None and
                    (type(progress[name]) not in (float, int) or not math.isfinite(progress[name]))):
                raise ValueError(f"invalid checkpoint metric: {name}")
        if progress["global_step"] and progress["train_loss"] is None:
            raise ValueError("checkpoint is missing the completed update's loss")
        rng = state["rng"]
        if not isinstance(rng, dict) or not {"python", "torch", "cuda", "train_loader", "validation_loader"} <= rng.keys():
            raise ValueError("checkpoint RNG state is incomplete")
        probe = random.Random(0)
        probe.setstate(rng["python"])
        for name in ("torch", "train_loader", "validation_loader"):
            torch.Generator().set_state(rng[name])
        if rng.get("epoch_iterator") is not None:
            torch.Generator().set_state(rng["epoch_iterator"])
            if progress["next_batch"] == 0:
                raise ValueError("checkpoint iterator state must belong to an in-progress epoch")
        expected_cuda = torch.cuda.device_count() if self.device.type == "cuda" else 0
        if len(rng["cuda"]) != expected_cuda:
            raise ValueError("checkpoint CUDA RNG device count differs")
        for value in rng["cuda"]:
            if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
                raise ValueError("checkpoint CUDA RNG state is invalid")

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a run's trajectory, after constructing its configured objects."""
        if self._running:
            raise RuntimeError("cannot load a checkpoint while training is running")
        self._validate_state(state)
        self.model.load_state_dict(state["model"], strict=True)
        for name, optimizer in self.optimizer.optimizers.items():
            optimizer.load_state_dict(state["optimizers"][name])
            self.optimizer.schedulers[name].load_state_dict(state["schedulers"][name])
        self.model.zero_grad(set_to_none=True)
        self.train_model.train(state["model_training"])
        for name in ("global_step", "epoch", "next_batch", "tokens_seen", "clipped_steps",
                     "train_loss", "validation_loss", "best_validation_loss"):
            setattr(self, name, state["progress"][name])
        self.last_validation_step = state["progress"].get("last_validation_step", 0)
        self._epoch_loader_rng = state["rng"].get("epoch_iterator")
        self._at_update_boundary = True
        self._stop_requested = False
        self.train_loader.set_epoch(self.epoch, self.next_batch)
        self.writer.close()
        self.writer = SummaryWriter(
            log_dir=self.run_dir / "tensorboard", purge_step=self.global_step + 1,
            flush_secs=self.config.logging.tensorboard_flush_secs,
        )
        # Restore RNG last: construction and logging must not perturb continuation.
        rng = state["rng"]
        random.setstate(rng["python"])
        torch.set_rng_state(rng["torch"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all(rng["cuda"])
        self.train_loader.generator.set_state(rng["train_loader"])
        self.validation_loader.generator.set_state(rng["validation_loader"])

    def save_checkpoint(self) -> Path:
        self.writer.flush()
        return self.checkpoints.save(self)

    def load_checkpoint(self, path: str | Path | None = None) -> None:
        self.checkpoints.load(self, path)

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, *, overrides: Mapping[str, Any] | None = None,
    ) -> Self:
        payload = CheckpointManager.read(path)
        if not payload["full_state"]:
            raise ValueError("resuming requires a full checkpoint saved with checkpoint.save_training_state=true")
        config = PretrainingConfig.from_toml(payload["config_toml"])
        if overrides:
            config = config.with_overrides(overrides)
        if not _matches_training_signature(payload["trainer_state"].get("training_signature"), config):
            raise ValueError("resume training settings differ from the checkpoint")
        trainer = cls(config)
        try:
            trainer.load_state_dict(payload["trainer_state"])
        except Exception:
            trainer.close()
            raise
        trainer.source_checkpoint = str(Path(path).expanduser().resolve())
        logger.info("Resumed PretrainingTrainer '{}' at step {}", trainer.config.name, trainer.global_step)
        return trainer
