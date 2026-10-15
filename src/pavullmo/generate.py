from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import sentencepiece as spm
import torch
from loguru import logger

from components.checkpoint import CheckpointManager, model_state_from_checkpoint
from config import CheckpointConfig, ModelConfig
from config.serialization import parse_toml
from model import DecoderTransformer


# The context length stored in the checkpoint is always respected, so a prompt
# reduces the number of tokens that can be generated.
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.8
TOP_K = 50

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively generate text with a PavuLLMo checkpoint."
    )
    parser.add_argument("model_dir", type=Path, help="saved run or migrated model directory")
    parser.add_argument("tokenizer", type=Path, help="path to tokenizer")
    parser.add_argument(
        "--checkpoint", type=Path,
        help="specific checkpoint, absolute or relative to the model directory",
    )
    return parser.parse_args(argv)


def find_checkpoint(model_dir: Path, checkpoint: Path | None = None) -> Path:
    """Select the latest managed save, or an unambiguous legacy checkpoint."""
    if not model_dir.is_dir():
        raise NotADirectoryError(f"model directory not found: {model_dir}")
    if checkpoint is not None:
        path = checkpoint.expanduser()
        path = path if path.is_absolute() else model_dir / path
        if not path.is_file():
            raise FileNotFoundError(f"model checkpoint not found: {path}")
        return path

    directory = model_dir / "checkpoints"
    if not directory.is_dir():
        directory = model_dir
    manager = CheckpointManager(CheckpointConfig(), directory)
    try:
        return manager.latest_checkpoint
    except FileNotFoundError:
        paths = sorted(path for path in directory.glob("*.pt") if path.is_file())
        if not paths:
            raise FileNotFoundError(f"no checkpoints in {directory}") from None
        if len(paths) != 1:
            raise ValueError(
                f"multiple checkpoints in {directory}; select one with --checkpoint"
            ) from None
        return paths[0]


def load_model(
    model_dir: Path,
    device: torch.device,
    *,
    checkpoint: Path | None = None,
) -> tuple[DecoderTransformer, int, int]:
    model_dir = Path(model_dir).expanduser().resolve()
    checkpoint_path = find_checkpoint(model_dir, checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = model_state_from_checkpoint(payload)
    if "trainer_state" in payload:
        # The selected checkpoint's snapshot is authoritative even when a run
        # directory has since been reused with a different config.toml.
        text = payload.get("config_toml")
        if not isinstance(text, str):
            raise ValueError("checkpoint is missing its TOML configuration")
    else:
        config_path = model_dir / "config.toml"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"model config not found: {config_path}; run migrate_checkpoint_config.py first"
            )
        text = config_path.read_text(encoding="utf-8")
    data = parse_toml(text)
    if not isinstance(data.get("model"), dict):
        raise ValueError("saved configuration must contain a [model] section")
    config = ModelConfig.from_dict(data["model"])
    # Release optimizer/RNG payloads before allocating the model.
    del payload
    model = DecoderTransformer(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Loaded model checkpoint {} on {}", checkpoint_path, device)
    return model, config.vocab_size, config.seq_len


def sample_next_token(logits: torch.Tensor, blocked_ids: set[int]) -> int:
    logits = logits.float() / TEMPERATURE
    for token_id in blocked_ids:
        logits[token_id] = -torch.inf

    top_k = min(TOP_K, logits.numel())
    cutoff = torch.topk(logits, top_k).values[-1]
    logits[logits < cutoff] = -torch.inf
    probabilities = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1).item())


@torch.inference_mode()
def generate(
    model: DecoderTransformer,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
    device: torch.device,
    sequence_length: int,
) -> str:
    prompt_ids = tokenizer.encode(prompt, out_type=int)
    token_ids = [tokenizer.bos_id(), *prompt_ids]
    if len(token_ids) >= sequence_length:
        raise ValueError(
            f"prompt uses {len(token_ids)} tokens, but the model context length is "
            f"{sequence_length}; shorten the prompt"
        )

    blocked_ids = {
        token_id
        for token_id in (tokenizer.unk_id(), tokenizer.bos_id(), tokenizer.pad_id())
        if token_id >= 0
    }
    generated_ids: list[int] = []
    available_tokens = sequence_length - len(token_ids)

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else nullcontext()
    )
    with autocast_context:
        for _ in range(min(MAX_NEW_TOKENS, available_tokens)):
            input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)[
                None, :
            ]
            logits = model(input_ids)[0, -1]
            next_token = sample_next_token(logits, blocked_ids)

            if next_token == tokenizer.eos_id():
                break
            token_ids.append(next_token)
            generated_ids.append(next_token)

    return tokenizer.decode(generated_ids)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    model, vocab_size, sequence_length = load_model(
        args.model_dir, device, checkpoint=args.checkpoint,
    )
    if tokenizer.vocab_size() != vocab_size:
        raise ValueError(
            f"tokenizer vocabulary size {tokenizer.vocab_size()} does not match "
            f"checkpoint vocabulary size {vocab_size}"
        )
    if tokenizer.bos_id() < 0 or tokenizer.eos_id() < 0:
        raise ValueError("tokenizer must define both BOS and EOS tokens")

    print("Enter a prompt. Press Ctrl-D or Ctrl-C to exit.")

    while True:
        try:
            prompt = input("\nPrompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        try:
            completion = generate(
                model,
                tokenizer,
                prompt,
                device,
                sequence_length,
            )
        except ValueError as error:
            print(f"Error: {error}")
            continue

        print(f"Generation> {completion}")


if __name__ == "__main__":
    main()
