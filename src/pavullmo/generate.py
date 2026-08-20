from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import sentencepiece as spm
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent / "model"
sys.path.insert(0, str(MODEL_DIR))

from model import DecoderTransformer


# The context length stored in the checkpoint is always respected, so a prompt
# reduces the number of tokens that can be generated.
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.8
TOP_K = 50

ARCHITECTURE_KEYS = (
    "VOCAB_SIZE",
    "N_BLOCKS",
    "EMBED_DIM",
    "ATTN_HEADS",
    "FFN_DIM",
    "SEQ_LEN",
    "ROPE_BASE",
    "DROPOUT",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively generate text with a PavuLLMo checkpoint."
    )
    parser.add_argument("model", type=Path, help="path to the model checkpoint")
    parser.add_argument("tokenizer", type=Path, help="path to tokenizer")
    return parser.parse_args()


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[DecoderTransformer, int, int]:
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

    missing = [key for key in ARCHITECTURE_KEYS if key not in hyperparameters]
    if missing:
        raise ValueError(
            "checkpoint is missing architecture settings: " + ", ".join(missing)
        )

    model = DecoderTransformer(
        vocab_size=int(hyperparameters["VOCAB_SIZE"]),
        n_blocks=int(hyperparameters["N_BLOCKS"]),
        embed_dim=int(hyperparameters["EMBED_DIM"]),
        attn_heads=int(hyperparameters["ATTN_HEADS"]),
        ffn_dim=int(hyperparameters["FFN_DIM"]),
        dropout=float(hyperparameters["DROPOUT"]),
        seq_len=int(hyperparameters["SEQ_LEN"]),
        rope_base=float(hyperparameters["ROPE_BASE"]),
    )

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint does not contain a model_state_dict")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return (
        model,
        int(hyperparameters["VOCAB_SIZE"]),
        int(hyperparameters["SEQ_LEN"]),
    )


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
    model, vocab_size, sequence_length = load_model(args.model, device)
    if tokenizer.vocab_size() != vocab_size:
        raise ValueError(
            f"tokenizer vocabulary size {tokenizer.vocab_size()} does not match "
            f"checkpoint vocabulary size {vocab_size}"
        )
    if tokenizer.bos_id() < 0 or tokenizer.eos_id() < 0:
        raise ValueError("tokenizer must define both BOS and EOS tokens")

    print(f"Loaded {args.model} on {device}.")
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
