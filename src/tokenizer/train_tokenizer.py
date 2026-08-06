from datasets import load_dataset
import sentencepiece as spm
from tqdm import tqdm
from itertools import islice


DATASET = "gsarti/clean_mc4_it"
VARIANT = "tiny"
N_DOCS = 1000000
VOCAB_SIZE = 16000
N_THREADS = 8

ds = iter(load_dataset(DATASET, VARIANT, split="train", streaming=True))


def text_iterator():
    docs = islice(ds, N_DOCS)

    for doc in tqdm(
        docs,
        total=N_DOCS,
        desc="Reading corpus",
        unit="docs",
    ):
        text = doc["text"].strip()

        if text:
            yield text


spm.SentencePieceTrainer.train(
    sentence_iterator=text_iterator(),
    model_prefix="tokenizer",
    vocab_size=VOCAB_SIZE,
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
