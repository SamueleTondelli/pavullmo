from __future__ import annotations

import bisect
import json
import csv
import os
import random
import sys
from pathlib import Path
from typing import Any
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


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


BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
SEED = int(os.environ.get("SEED", 42))

# Data, evaluation, and runtime settings.
DATASET_DIR = Path(
    os.environ.get("DATASET_DIR", str(PROJECT_DIR / "dataset" / "ds"))
).expanduser()
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 4))
PIN_MEMORY = env_bool("PIN_MEMORY", True)


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


def config_string(hyperparameters: dict[str, object]) -> str:
    return ";".join(f"{name}={value}" for name, value in hyperparameters.items())


@torch.no_grad()
def validation_loss(
    model: torch.nn.Module,
    validation_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    model.eval()
    loss_sum = 0.0
    token_count = 0

    for input_ids, targets in validation_loader:
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
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


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[DecoderTransformer, dict[str, object]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")

    hyperparameters = checkpoint.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        raise ValueError("checkpoint does not contain a hyperparameters mapping")

    model = DecoderTransformer(
        vocab_size=int(hyperparameters["VOCAB_SIZE"]),
        n_blocks=int(hyperparameters["N_BLOCKS"]),
        embed_dim=int(hyperparameters["EMBED_DIM"]),
        attn_heads=int(hyperparameters["ATTN_HEADS"]),
        ffn_dim=int(hyperparameters["FFN_DIM"]),
        dropout=float(hyperparameters["DROPOUT"]),
        seq_len=int(hyperparameters["SEQ_LEN"]),
        rope_base=float(hyperparameters["ROPE_BASE"]),
        qk_norm=bool(hyperparameters["QK_NORM"]),
    )

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint does not contain a model_state_dict")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, hyperparameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate model on full tiny validation"
    )
    parser.add_argument(
        "--model", type=Path, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--csv", type=Path, required=True, help="Path to output csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model, hyperparameters = load_model(args.model, device)

    validation_dir = DATASET_DIR / f"validation_{hyperparameters['DATASET_PREFIX']}"
    validation_dataset = TokenBlockDataset(
        validation_dir, int(hyperparameters["SEQ_LEN"])
    )

    validation_loader = build_loader(validation_dataset, shuffle=False)

    val_loss = validation_loss(
        model,
        validation_loader,
        device,
    )

    print(f"Final validation loss: {val_loss}")

    train_dir = (
        DATASET_DIR
        / f"train_{hyperparameters['DATASET_PREFIX']}_{hyperparameters['DATASET_VARIANT']}"
    )
    metadata_path = train_dir / "metadata.json"
    train_size = 0
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
        train_size = metadata["token_count"]

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embedding_parameters = model.embeddings.weight.numel()
    scaling_parameters = total_trainable - embedding_parameters

    csv_path = args.csv

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["tokenizer", "parameters", "train_tokens", "validation_loss"],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "tokenizer": hyperparameters["DATASET_PREFIX"],
                "parameters": scaling_parameters,
                "train_tokens": train_size,
                "validation_loss": val_loss,
            }
        )


if __name__ == "__main__":
    main()
