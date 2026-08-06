import argparse
from itertools import islice
from pathlib import Path

import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm


DATASET = "gsarti/clean_mc4_it"
SAMPLE_VARIANT = "tiny"
DEFAULT_SAMPLE_DOCS = 1_000

# The dataset card reports approximate word counts for the training variants.
SPLITS = {
    "tiny": {"docs": 10_000_000, "words": 4_000_000_000},
    "small": {"docs": 20_000_000, "words": 8_000_000_000},
    "medium": {"docs": 50_000_000, "words": 20_000_000_000},
    "large": {"docs": 75_000_000, "words": 30_000_000_000},
    "full": {"docs": 103_000_000, "words": 41_000_000_000},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate token counts for clean_mc4_it from a tiny sample."
    )
    parser.add_argument(
        "--sample-docs",
        type=int,
        default=DEFAULT_SAMPLE_DOCS,
        help=f"number of tiny training documents to tokenize (default: {DEFAULT_SAMPLE_DOCS:,})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).with_name("tokenizer.model"),
        help="SentencePiece model to use",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_docs <= 0:
        raise ValueError("--sample-docs must be greater than zero")

    tokenizer = spm.SentencePieceProcessor(model_file=str(args.model))
    dataset = load_dataset(DATASET, SAMPLE_VARIANT, split="train", streaming=True)

    sampled_docs = 0
    sampled_chars = 0
    sampled_words = 0
    sampled_tokens = 0

    documents = islice(dataset, args.sample_docs)
    for document in tqdm(
        documents,
        total=args.sample_docs,
        desc=f"Tokenizing {SAMPLE_VARIANT}",
        unit="docs",
    ):
        text = document["text"]
        sampled_docs += 1
        sampled_chars += len(text)
        sampled_words += len(text.split())
        sampled_tokens += len(tokenizer.encode(text, out_type=int))

    if sampled_docs == 0 or sampled_words == 0 or sampled_tokens == 0:
        raise RuntimeError("The sample did not contain enough text to make an estimate")

    tokens_per_word = sampled_tokens / sampled_words

    print(f"\nSample ({SAMPLE_VARIANT} train)")
    print(f"  documents:  {sampled_docs:,}")
    print(f"  words:      {sampled_words:,}")
    print(f"  tokens:     {sampled_tokens:,}")
    print(f"  tokens/word: {tokens_per_word:.4f}")
    print(f"  chars/token: {sampled_chars / sampled_tokens:.4f}")

    print("\nEstimated training tokens")
    print(f"{'split':<8} {'docs':>13} {'words':>14} {'tokens':>18}")
    print("-" * 56)
    for split, size in SPLITS.items():
        estimated_tokens = round(size["words"] * tokens_per_word)
        print(
            f"{split:<8} {size['docs']:>13,} {size['words']:>14,} "
            f"{estimated_tokens:>18,}"
        )

    print(
        "\nEstimates exclude BOS/EOS and inherit the uncertainty of the "
        "dataset card's rounded word counts."
    )


if __name__ == "__main__":
    main()
