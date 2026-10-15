"""Explicit-path token datasets and train/validation data-loader wrappers."""

from __future__ import annotations

import bisect
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset, Sampler

from config import PretrainingConfig


class EpochBatchSampler(Sampler[list[int]]):
    """Recreate shuffles with a consumed-batch cursor independent of prefetch."""

    def __init__(self, dataset_size: int, batch_size: int, seed: int):
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
            yield indices[start:start + self.batch_size]


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

        sample = self.metadata.get("fixed_evaluation_sample")
        self.evaluation_starts: list[int] | None = None
        if sample is not None:
            if sample["sequence_length"] != sequence_length:
                raise ValueError("model.seq_len must match the fixed validation sample sequence_length")
            starts = [block["start_token"] for block in sample["blocks"]]
            if not starts or len(starts) != sample["block_count"]:
                raise ValueError("invalid fixed validation sample block count")
            if len(set(starts)) != len(starts) or any(
                not isinstance(start, int) or start < 0
                or start + sequence_length >= total_tokens for start in starts
            ):
                raise ValueError("invalid or duplicate fixed validation sample offsets")
            self.evaluation_starts = starts
            self.block_count = len(starts)

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

        start = (self.evaluation_starts[index] if self.evaluation_starts is not None
                 else index * self.sequence_length)
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


def _loader_options(config: PretrainingConfig, generator: torch.Generator) -> dict[str, Any]:
    options: dict[str, Any] = {
        "num_workers": config.runtime.num_workers,
        "pin_memory": config.runtime.pin_memory,
        "generator": generator,
    }
    if config.runtime.num_workers:
        options.update(
            persistent_workers=config.runtime.persistent_workers,
            prefetch_factor=config.runtime.prefetch_factor,
        )
    return options


class TrainDataLoader(DataLoader[tuple[torch.Tensor, torch.Tensor]]):
    """Shuffled full batches with a reproducible, explicit consumed-batch cursor."""

    def __init__(self, dataset_path: str | Path, config: PretrainingConfig):
        dataset = TokenBlockDataset(Path(dataset_path).expanduser(), config.model.seq_len)
        if dataset.evaluation_starts is not None:
            raise ValueError("the fixed validation sample must not be used as training data")
        if len(dataset) < config.training.batch_size * config.training.grad_accum_steps:
            raise ValueError("training dataset is too small for one optimizer step")
        generator = torch.Generator().manual_seed(config.training.seed)
        sampler = EpochBatchSampler(len(dataset), config.training.batch_size, config.training.seed)
        super().__init__(
            dataset, batch_sampler=sampler, **_loader_options(config, generator),
        )
        logger.info("Loaded training artifact {} ({} tokens)", dataset_path, dataset.total_tokens)

    def set_epoch(self, epoch: int, start_batch: int = 0) -> None:
        """Skip already consumed batches without reading or counting prefetch."""
        self.batch_sampler.set_epoch(epoch, start_batch)


class ValidationDataLoader(DataLoader[tuple[torch.Tensor, torch.Tensor]]):
    """Ordered validation batches, including the final partial batch."""

    def __init__(self, dataset_path: str | Path, config: PretrainingConfig):
        dataset = TokenBlockDataset(Path(dataset_path).expanduser(), config.model.seq_len)
        if dataset.evaluation_starts is not None:
            required_batches = (len(dataset) + config.training.batch_size - 1) // config.training.batch_size
            if config.evaluation.steps < required_batches:
                raise ValueError(
                    f"evaluation.steps must be at least {required_batches} to evaluate "
                    "the complete fixed stratified sample"
                )
        generator = torch.Generator().manual_seed(config.training.seed)
        super().__init__(
            dataset, batch_size=config.training.batch_size, shuffle=False,
            drop_last=False, **_loader_options(config, generator),
        )
        logger.info("Loaded validation artifact {} ({} tokens)", dataset_path, dataset.total_tokens)

