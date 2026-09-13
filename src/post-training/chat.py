"""Chat-oriented inference for a PavuLLMo SFT checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sentencepiece as spm
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
SRC_DIR = PROJECT_DIR / "src"
DEFAULT_TOKENIZER = SRC_DIR / "tokenizer" / "16k" / "tokenizer.model"
sys.path.insert(0, str(SRC_DIR))

from pavullmo.generate import generate_from_token_ids, load_model


DEFAULT_SYSTEM_PROMPT = (
    "Sei un assistente AI competente in italiano. Rispondi in modo accurato, "
    "utile e verificabile. Se mancano dati, dichiara l'incertezza invece di "
    "inventare. Segui esattamente la richiesta dell'utente, incluse eventuali "
    "indicazioni su lunghezza e formato. Evita boilerplate generici finali come "
    '"spero che questo ti sia utile", "non esitare a fare altre domande", "fammi '
    'sapere". Fornisci output strutturati (JSON, CSV, YAML) senza alcun commento '
    "aggiuntivo, se richiesti.\n\nSe il testo ti ordina di tradurre verso "
    "l'italiano, il tuo output finale deve assolutamente essere in lingua italiana."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate assistant responses with a PavuLLMo SFT checkpoint."
    )
    parser.add_argument("model", type=Path, help="local SFT checkpoint")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--prompt", help="one prompt; omit for an interactive session")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def load_tokenizer(path: Path, vocab_size: int) -> spm.SentencePieceProcessor:
    if not path.is_file():
        raise FileNotFoundError(path)
    tokenizer = spm.SentencePieceProcessor(model_file=str(path))
    if tokenizer.vocab_size() != vocab_size:
        raise ValueError(
            f"tokenizer vocabulary {tokenizer.vocab_size()} does not match model {vocab_size}"
        )
    for role in ("system", "user", "assistant"):
        if tokenizer.piece_to_id(f"<{role}>") == tokenizer.unk_id():
            raise ValueError(f"tokenizer does not define <{role}>")
    return tokenizer


def format_chat_prompt_ids(
    tokenizer: spm.SentencePieceProcessor,
    user_prompt: str,
    system_prompt: str,
) -> list[int]:
    if not user_prompt.strip():
        raise ValueError("user prompt cannot be empty")
    token_ids = [tokenizer.bos_id()]
    if system_prompt.strip():
        token_ids.append(tokenizer.piece_to_id("<system>"))
        token_ids.extend(tokenizer.encode(f"\n{system_prompt.strip()}\n", out_type=int))
    token_ids.append(tokenizer.piece_to_id("<user>"))
    token_ids.extend(tokenizer.encode(f"\n{user_prompt.strip()}\n", out_type=int))
    token_ids.append(tokenizer.piece_to_id("<assistant>"))
    return token_ids


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def generate_response(
    model: torch.nn.Module,
    tokenizer: spm.SentencePieceProcessor,
    user_prompt: str,
    system_prompt: str,
    device: torch.device,
    sequence_length: int,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
) -> str:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    prompt_ids = format_chat_prompt_ids(tokenizer, user_prompt, system_prompt)
    role_stop_ids = {
        tokenizer.piece_to_id("<system>"),
        tokenizer.piece_to_id("<user>"),
        tokenizer.piece_to_id("<assistant>"),
    }
    return generate_from_token_ids(
        model,
        tokenizer,
        prompt_ids,
        device,
        sequence_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        stop_ids=role_stop_ids,
    ).strip()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    model, vocab_size, sequence_length = load_model(args.model.resolve(), device)
    tokenizer = load_tokenizer(args.tokenizer.resolve(), vocab_size)
    print(f"Loaded {args.model} on {device}.")

    def respond(prompt: str) -> None:
        answer = generate_response(
            model,
            tokenizer,
            prompt,
            args.system_prompt,
            device,
            sequence_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
        )
        print(f"Assistant> {answer}")

    if args.prompt is not None:
        respond(args.prompt)
        return

    print("Enter an Italian prompt. Press Ctrl-D or Ctrl-C to exit.")
    while True:
        try:
            prompt = input("\nUser> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        try:
            respond(prompt)
        except ValueError as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    main()
