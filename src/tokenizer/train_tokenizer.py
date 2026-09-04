import argparse
from pathlib import Path

from datasets import load_dataset
import sentencepiece as spm
from tqdm import tqdm


DATASET = "gsarti/clean_mc4_it"
VARIANT = "tiny"
N_THREADS = 8

ds = iter(load_dataset(DATASET, VARIANT, split="train", streaming=True))


def text_iterator(byte_budget):
    used_bytes = 0
    with tqdm(
        total=byte_budget,
        desc="Reading corpus",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress:
        for doc in ds:
            if used_bytes >= byte_budget:
                break
            text = doc["text"].strip()

            if text:
                text_bytes = len(text.encode("utf-8"))
                used_bytes += text_bytes
                progress.update(text_bytes)
                yield text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, help="Size of the vocabulary")
    parser.add_argument(
        "--byte-budget",
        type=int,
        help="How many bytes of text are used to train the tokenizer",
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
        help="Directory where tokenizer.model and tokenizer.vocab are saved",
    )

    args = parser.parse_args()
    output_dir = Path(args.model_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_prefix = output_dir / "tokenizer"

    print(
        f"Training with size {args.vocab_size} on {args.byte_budget} bytes, "
        f"saving to {tokenizer_prefix}.*"
    )
    spm.SentencePieceTrainer.train(
        sentence_iterator=text_iterator(args.byte_budget),
        model_prefix=str(tokenizer_prefix),
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        byte_fallback=True,  # fallsback to raw byte as token if it wasnt trained on it
        # special tokens
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        user_defined_symbols=["<system>", "<user>", "<assistant>"],
        num_threads=N_THREADS,
    )


if __name__ == "__main__":
    main()
