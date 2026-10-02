"""Read and decode flat uint16 artifacts without importing a training entry point.

Example: python dataset/inspect_tokens.py artifacts/datasets/train_balanced_1b
    --start 800000000 --tokens 2048
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm


class TokenArtifact:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        storage = self.metadata["storage"]
        if (storage["dtype"], storage["endianness"], storage["layout"]) != (
            "uint16", "little", "flat_token_stream"
        ):
            raise ValueError("expected a flat little-endian uint16 artifact")
        self.maps = []
        self.ends = []
        total = 0
        for shard in self.metadata["shards"]:
            path = self.directory / shard["file"]
            if path.stat().st_size != shard["tokens"] * 2:
                raise ValueError(f"incorrect shard size: {path}")
            self.maps.append(np.memmap(path, mode="r", dtype="<u2"))
            total += shard["tokens"]
            self.ends.append(total)
        if total != self.metadata["token_count"]:
            raise ValueError("shard and metadata token counts differ")
        self.total_tokens = total
        self.ranges = self.metadata.get("source_ranges")
        if self.ranges is None:
            order = self.metadata.get("artifact_source_order") or self.metadata[
                "stream_shuffle"
            ]["artifact_source_order"]
            self.ranges = []
            start = 0
            for source in order:
                count = self.metadata["dataset"]["tokens_by_source"][source]
                self.ranges.append(dict(source=source, start_token=start, tokens=count))
                start += count

    def read(self, start: int, count: int) -> np.ndarray:
        if start < 0 or count < 0 or start + count > self.total_tokens:
            raise IndexError((start, count))
        pieces = []
        while count:
            i = bisect.bisect_right(self.ends, start)
            offset = start - (self.ends[i - 1] if i else 0)
            take = min(count, len(self.maps[i]) - offset)
            pieces.append(self.maps[i][offset:offset + take])
            start += take
            count -= take
        return np.concatenate(pieces) if len(pieces) != 1 else pieces[0]

    def source_at(self, start: int) -> str:
        for region in self.ranges:
            if region["start_token"] <= start < region["start_token"] + region["tokens"]:
                return region["source"]
        raise IndexError(start)

    def documents(self):
        """Yield complete BOS/EOS chunks, including chunks crossing shard boundaries.

        Quota-truncated final chunks are intentionally excluded. These chunks
        are not necessarily whole upstream documents.
        """
        bos = self.metadata["tokenizer"]["special_token_ids"]["bos"]
        eos = self.metadata["tokenizer"]["special_token_ids"]["eos"]
        pending = None
        base = 0
        for shard in self.maps:
            ends = np.flatnonzero(shard == eos)
            begins = np.flatnonzero(shard == bos)
            begin_indices = np.searchsorted(begins, ends, side="right") - 1
            left = 0
            for end, begin_index in zip(ends, begin_indices, strict=True):
                part = shard[left:int(end) + 1]
                start = base + left
                # Each source quota may end mid-chunk, followed immediately by
                # the next source's BOS. Keep the new complete chunk rather
                # than merging the two documents at this quota boundary.
                if begin_index >= 0 and begins[begin_index] >= left:
                    start = base + int(begins[begin_index])
                    part = shard[int(begins[begin_index]):int(end) + 1]
                    pending = None
                elif pending is not None:
                    start, prefix = pending
                    part = np.concatenate([prefix, part])
                    pending = None
                if len(part) and part[0] == bos:
                    yield start, part
                left = int(end) + 1
            if left < len(shard):
                tail = shard[left:]
                if len(begins) and begins[-1] >= left:
                    pending = base + int(begins[-1]), shard[int(begins[-1]):]
                elif pending is not None:
                    start, prefix = pending
                    pending = start, np.concatenate([prefix, tail])
                else:
                    pending = base + left, tail
            base += len(shard)


def load_tokenizer(path: str | Path, artifact: TokenArtifact):
    path = Path(path)
    expected = artifact.metadata["tokenizer"]["sha256"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("tokenizer hash does not match artifact")
    return spm.SentencePieceProcessor(model_file=str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--tokenizer", type=Path,
                        default=Path("artifacts/tokenizers/production_16k/tokenizer.model"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--output", type=Path, help="optional JSONL output")
    args = parser.parse_args()
    artifact = TokenArtifact(args.artifact)
    tokenizer = load_tokenizer(args.tokenizer, artifact)
    rng = np.random.default_rng(args.seed)
    regions = [r for r in artifact.ranges if not args.source or r["source"] == args.source]
    if not regions:
        parser.error("source not present in artifact")
    records = []
    for i in range(args.samples):
        start = args.start + i * args.tokens
        if args.random or args.source:
            weights = np.array([r["tokens"] for r in regions], dtype=float)
            region = regions[int(rng.choice(len(regions), p=weights / weights.sum()))]
            if args.tokens > region["tokens"]:
                parser.error("sample exceeds source region")
            start = int(rng.integers(region["start_token"],
                                     region["start_token"] + region["tokens"] - args.tokens + 1))
        ids = artifact.read(start, args.tokens).tolist()
        # Decode each document separately: SentencePiece otherwise hides BOS/EOS.
        parts, current = [], []
        special = {v: k for k, v in artifact.metadata["tokenizer"]["special_token_ids"].items()}
        for token in ids:
            if token in special:
                if current:
                    parts.append(tokenizer.decode(current))
                    current = []
                parts.append(f"\n<{special[token].upper()}>\n")
            else:
                current.append(token)
        if current:
            parts.append(tokenizer.decode(current))
        record = dict(start_token=start, source=artifact.source_at(start), tokens=len(ids),
                      text="".join(parts))
        records.append(record)
        print(json.dumps(record, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


if __name__ == "__main__":
    main()
