# pavullmo

This project aims to pretrain a small (~50M parameter) decoder-only language
model for Italian. It is an early-stage `uv` project: dataset sizing and the
tokenizer have been explored, while the model and training pipeline have not
yet been implemented.

## Data and tokenizer

- The corpus is `gsarti/clean_mc4_it`, loaded from Hugging Face Datasets in
  streaming mode. It provides `tiny`, `small`, `medium`, `large`, and `full`
  variants; current tokenizer work uses `tiny`.
- `src/tokenizer/train_tokenizer.py` trains a 16,000-token SentencePiece BPE on
  up to 1,000,000 documents, with full character coverage and byte fallback.
- Token IDs are fixed as UNK=0, BOS=1, EOS=2, and PAD=3. The vocabulary also
  includes `<system>`, `<user>`, and `<assistant>`.
- `src/tokenizer/count_tokens.py` samples the streamed corpus and extrapolates
  token counts for each dataset variant from the dataset card's approximate
  word counts.

## Environment

- Use Python 3.12 and manage dependencies and the lockfile with `uv`.
- Training dependencies are PyTorch 2.9.1 with its CUDA 12.8 wheels,
  `datasets<4`, SentencePiece, tqdm, and TensorBoard 2.21.
- The optional `cloud` extra provides the Modal client. Full training is
  expected to run on a Modal L4 (24 GB); local smoke tests target an RTX 3070
  (8 GB).
- Use PyTorch's native scaled-dot-product attention. Its FlashAttention-2
  backend works on both target GPUs; do not add the third-party `flash-attn`
  package unless a concrete unsupported API is required.

Keep `pyproject.toml` and `uv.lock` synchronized, preserve the pinned GPU stack,
and run `pavullmo-check` after changing PyTorch, CUDA-related dependencies, or
GPU environment checks.
