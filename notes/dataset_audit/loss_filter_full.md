# Full sanitized-corpus paired losses

This experiment compares `35m_balanced_1b` with `35m_newbalsanitized_1b` on every available next-token transition in the sanitized training, validation, and test artifacts. Both checkpoints use the production tokenizer and identical 1,024-token contexts, with FP32 parameters, BF16 autocast, and FP32 cross entropy.

The score is the signed difference

\[
\Delta(x)=L_{\mathrm{old}}(x)-L_{\mathrm{sanitized}}(x).
\]

Positive values mean the sanitized model predicts the sequence better. This measures corpus specialization; it does not establish text quality.

## Method and coverage

Training contains 976,563 ordered windows covering all 999,999,999 next-token transitions. Validation and test each contain 9,766 windows covering 9,999,999 transitions. Final partial windows are included. Adjacent windows share a boundary token without double-counting any transition; the first token has no predecessor and is not scored.

Windows contain packed document fragments, with attention continuing across document boundaries as in training. Cross-source windows contribute to whole-split results but are excluded from source-specific summaries. Reported means are weighted by target-token count; summary quantiles describe the distribution of window losses. Coverage, offsets, finite scores, tokenizer identity, and numerical agreement with the ordinary checkpoint forward pass were verified.

## Results

All losses and differences are in nats per target token.

| Full split | Old-model loss | Sanitized-model loss | Difference |
|---|---:|---:|---:|
| Training | 3.216015 | 2.994571 | 0.221444 |
| Validation | 3.215461 | 3.017478 | 0.197983 |
| Test | 3.215184 | 3.018645 | 0.196539 |

Validation and test give nearly identical differences. The sanitized model's larger training advantage includes in-sample learning and should not be interpreted as independent detector accuracy.

| Source | Training difference | Validation difference | Test difference |
|---|---:|---:|---:|
| Web | 0.356868 | 0.344676 | 0.344766 |
| Wikipedia | 0.133797 | 0.086532 | 0.080162 |
| Educational PDFs | 0.038245 | 0.016109 | 0.016508 |

With web representing approximately half the evaluation mixture, it accounts for roughly 88% of the aggregate test gap. This attributes improvement to the web source, not specifically to spam. The [screening pilot](loss_filter_findings.md) supplies the decoded examples and exploratory topic-based evidence that some large differences come from incoherent loan/vaping templates.

## Ranked sequences with text

Training windows were ranked by descending signed difference, with ascending token offset breaking ties.

| Export | Windows | Sources | Difference range |
|---|---:|---|---:|
| `top_50000_delta.parquet` | 50,000 | 49,998 web; 2 Wikipedia | 1.246183–2.530049 |
| `top_10000_delta.parquet` | 10,000 | All web | 1.766861–2.530049 |

The top-50,000 export is 53,847,809 bytes. It retains the score columns and adds `rank` and `text`. Rank 1 has the largest difference. Text decodes the input tokens beginning at `start_token`; control tokens such as BOS/EOS are made visible to distinguish neighboring document fragments. The final scored target at `start_token + target_tokens` is just outside this input text.

Ranking, tokenizer identity, nonempty text, and the Parquet round-trip were verified. These exports are candidates for review, not labeled spam datasets. Useful text can also have a large difference, and unwanted text familiar to both models may have a small difference.

## Artifacts

Results are Zstandard-compressed Parquet under `artifacts/dataset_audit/loss_filter_full/`:

- `train.parquet`, `validation.parquet`, and `test.parquet`: complete paired window losses.
- `summary.parquet`: token-weighted means, counts, and window-score quantiles by split and source.
- `top_50000_delta.parquet` and `top_10000_delta.parquet`: ranked training windows with decoded text.

The main analytical fields are:

| Column | Meaning |
|---|---|
| `split` | Training, validation, or test |
| `start_token` | Global offset of the first input token in that split |
| `target_tokens` | Number of scored transitions, normally 1,024 |
| `source`, `source_end` | Sources of the first and last target tokens |
| `crosses_source_boundary` | Whether the target window spans a source boundary |
| `old_loss`, `sanitized_loss` | Mean negative log-likelihood under each checkpoint |
| `delta` | Old loss minus sanitized loss |
| `rank`, `text` | Position and decoded input sequence in the ranked exports |

Parquet metadata records checkpoint hashes, artifact and tokenizer identities, scoring settings, and export conventions. The implementation is in `loss_filter_full.py`; ranked-text extraction is in `export_loss_sequences.py`. The [main dataset audit](findings.md) covers source-mixture changes, benchmarks, scheduling, norms, and leakage checks.
