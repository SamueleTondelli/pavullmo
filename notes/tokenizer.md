# Tokenizer experiment history

Status: 16 September 2026. Vocabulary labels use K = 1,024 entries.

## Results, in chronological order

1. **Initial tokenizer-only comparison: 8K / 12K / 16K.** We built a frozen
   four-source Italian pool (web, Wikipedia, books, educational PDFs), targeting
   64 MiB training and 8 MiB held-out text with 40/20/20/20 byte quotas.
   SentencePiece BPE settings and training text were shared. Larger vocabularies
   reduced fragmentation: held-out bytes/token were **3.539 / 3.797 / 3.968**.
   This suggested diminishing compression returns, but did not establish LM quality.

2. **Expanded tokenizer-only sweep: 4K–32K.** Ten sizes
   (4, 6, 8, 10, 12, 14, 16, 20, 24, 32K) showed continuing, diminishing
   compression gains. Vocabulary entries observed fewer than 100 times rose
   sharply beyond 20K: **2,160 at 20K, 6,045 at 24K, 15,202 at 32K**.
   Frequency and fragmentation plots distinguished vocabulary-entry counts from
   actual text usage; these exposure thresholds were diagnostics, not rejection rules.

3. **Independent tokenizer-only test.** All ten frozen candidates were evaluated
   on new, parent-disjoint four-source text. Compression trends persisted:
   **3.639 / 3.910 / 4.085 bytes/token for 8K / 12K / 16K**, reaching 4.447 at
   32K. Pieces trained fewer than 100 times accounted for **0.401% / 2.399% /
   5.983%** of test tokens at 20K / 24K / 32K. This tested tokenization
   generalization, not language prediction; the book subset contained only 19 parents.

4. **First tokenizer + LM screen: 14K / 16K / 20K.** One seed, the same raw
   training pool, and approximately 34.2M parameters gave validation scores of
   **1.66014 / 1.65720 / 1.65076 BPB** (bits per normalized byte; lower is better).
   FFN widths changed to offset embedding size. The 20K result motivated testing
   larger vocabularies, with seed uncertainty still unresolved.

5. **Overnight screen and confirmation.** Fresh, broader validation/test sets
   and shared raw-text prefixes were prepared. At 30M reference tokens, the
   16K / 20K / 24K / 32K screen scored **1.555240 / 1.548757 / 1.546066 /
   1.542376 validation BPB**. The selected 24K and 32K candidates were then trained
   from scratch at 80M reference tokens for three matched seeds:

   | Vocabulary | Mean validation BPB | Mean test BPB |
   |---|---:|---:|
   | 24K | 1.436366 | 1.433377 |
   | 32K | 1.435677 | 1.432227 |

   The 32K model won two seeds out of three; its mean test advantage was only
   **0.001150 BPB (0.08%)**, below the predefined 0.005 practical-tie tolerance.
   All jobs completed in **4h 16m**, within six hours. Reference-token budgets
   describe shared text measured with 16K; actual token counts and compute differ.
   Absolute scores are not directly comparable with the earlier LM screen because
   the evaluation text changed.

## Summary

Compression improves with vocabulary size, with diminishing returns and more
low-exposure entries. At approximately 34.19M parameters and the tested training
budget, **24K and 32K are practically tied**. Longer, three-seed LM confirmation
for **8K / 12K / 16K remains proposed, not completed**. No final tokenizer or
Italian-specific scaling law has been established. These are general-prose
experiments; the 30 conversational probes are qualitative, not a dialogue benchmark.

## Limitations and cleanup

Learning rates were shared rather than tuned per vocabulary, and confirmation
used fewer than 100M actual tokens. Results therefore describe this short-run
recipe, not each configuration's tuned or long-training optimum.

The tokenizer-selection code, datasets, checkpoints and generated reports were
removed after this review. The numerical history above is retained; prior report
locations were `artifacts/tokenizer-selection/{report.md,sweep_report.md,
test/report.md,lm_screen/report.md,overnight-20260916/report.md}`.
