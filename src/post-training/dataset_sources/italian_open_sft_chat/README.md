# Italian Open SFT Chat source adapter

This adapter downloads `SerFabio89/italian-open-sft-chat-dataset`, preserves
its official train/validation/test assignment, filters records, checks the
fully formatted length using PavuLLMo's SentencePiece tokenizer, and writes
normalized split JSONL files under the ignored
`data/italian_open_sft_chat/` directory.

`magpie` is the publisher's label for prompts created with Magpie-like
synthetic user simulation. `meta` means self-instruct/meta-prompt generation;
it does not mean that the examples came from Meta Platforms. These labels
describe generation provenance, not human verification, so examples still
need qualitative auditing.

The default selection is intentionally conservative for the 1,024-token model:

- `length_band=short`
- `source` in `magpie,meta`
- exactly one system/user/assistant exchange
- at most 1,024 tokens after role markers, BOS, and EOS are added

Run from the repository root:

```bash
uv run python src/post-training/dataset_sources/italian_open_sft_chat/prepare.py
uv run python src/post-training/build_post_dataset.py \
  --split-source-dir src/post-training/data/italian_open_sft_chat
```

Filters are repeatable. For example, add `--source safety_seed` to an explicit
list of sources or use `--conversation-mode any` when intentionally studying
multi-turn data. `--max-per-split N` builds a small deterministic development
subset without changing the publisher's split boundaries.

The preparation metadata records the exact Hugging Face repository commit used
for each build. The upstream dataset is licensed **CC BY-NC 4.0**, so outputs
derived from it require attribution and are restricted to non-commercial use.
