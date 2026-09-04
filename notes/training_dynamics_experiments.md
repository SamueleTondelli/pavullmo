# Pavullmo pretraining dynamics: diagnosis and experiment plan

_Originally written 29 August 2026; results updated 4 September 2026. Scope: optimizer, schedule, stabilization, and small architecture changes. Dataset work is intentionally excluded._

## Current status

The original diagnosis was correct: initialization had to be fixed before interpreting the other dynamics. `gpt_scaled` initialization removed the pathological initial loss and was promoted. Subsequent sweeps selected a WSD 80/20 schedule, batch 16, token-scaled Adam `beta2=0.9025`, non-embedding weight decay `0.05`, and QK norm. The 16k tokenizer won the tokenizer comparison and is now the scaling baseline.

The current tuned 10M/16k reference configuration is:

| Setting | Value |
|---|---:|
| Initialization | `gpt_scaled`, std `0.02` |
| QK norm | on |
| Peak LR | `2.4e-3` |
| Warmup | 219 optimizer steps |
| Schedule | WSD, 80% stable / 20% decay to zero |
| Batch | 16, no accumulation |
| Adam betas | `(0.9, 0.9025)` |
| Weight decay | `0.05`, excluding embeddings and norms |
| Z-loss | off |
| Norm LR multiplier | `1.0` |

The peak LR was re-tuned by model size: the current 20M, 35M, and 50M 16k configurations use `2.3e-3`, `2.2e-3`, and `2.1e-3`, respectively, while retaining the rest of the recipe above. These are the baselines against which future optimizer and architecture experiments should be compared.

Two attempted long-horizon stabilizers have now been rejected:

- Z-loss at `1e-5` had a noise-sized short-horizon benefit but hurt the longer run, did not consistently control raw logits, and imposed a large memory cost.
- Halving the learning rate of RMSNorm parameters also had a noise-sized short-horizon benefit but hurt the longer run and increased, rather than calmed, late norm-gradient magnitude.

The next high-value branch is therefore a controlled optimizer comparison, starting with native PyTorch Muon versus this tuned AdamW baseline. The small architecture lane should start with attention-output gating after that; z-loss and norm-specific LR tuning should not remain in the main queue.

## What the original runs said

### The original late gradient increase was mostly not an instability

`train/gradient_norm` is the **pre-clipping** value returned by `clip_grad_norm_`; every value above 1.0 is subsequently clipped. In the 141,337-step `35m_tok8k_4b` run:

| Training region | Median norm | p99 norm | Steps clipped |
|---|---:|---:|---:|
| First decile | 0.657 | 3.700 | 20.12% |
| Second decile | 0.361 | 1.274 | 1.30% |
| Last decile | 0.535 | 0.892 | 0.52% |
| After first 10%, overall | — | — | 0.91% |

So the median does fall and then rise, but it remains well below the clipping threshold and late clipping is rare. I would treat that slow rise as telemetry to watch, not as exploding gradients.

The real issue is the isolated tail: pre-clip norms reach 3,064, 2,461, 1,364, and 1,244 at ordinary batch losses of roughly 3.25–3.87. Loss alone cannot localize these events. Add per-module gradient/update and attention-logit logging before choosing a remedy.

### The original final slowdown followed the LR curve almost exactly

The original full-run cosine falls below 10% of peak for approximately the final 20% and reaches zero. Validation improvement by decile changes as follows:

| Decile | LR at end | Validation improvement |
|---:|---:|---:|
| 2 | 1.131e-3 | 0.310 |
| 5 | 0.625e-3 | 0.071 |
| 8 | 0.119e-3 | 0.026 |
| 9 | 0.0306e-3 | 0.016 |
| 10 | 0 | 0.005 |

This does not mean “never decay.” A terminal cooldown is useful. It means that a cosine spanning the whole run spends too much of this particular token budget at a low LR. [Warmup-stable-decay (WSD)](https://arxiv.org/abs/2410.05192) keeps a high-LR branch that can continue training, then applies a shorter cooldown to produce a terminal checkpoint.

### The original batch-16 sweep was incomplete

At equal tokens, the matched 35m/8k/42m sweeps give best validation loss **5.229** for batch 8 and **5.600** for batch 16. Available wall times show only about a **6.6% throughput gain** for batch 16 (roughly 87.5k → 93.3k tokens/s), so “train a few more tokens in the same time” is unlikely to close that gap on the measured GPU.

However, batch 16 also:

- halves the number of parameter updates;
- reaches the top of the tested LR grid, suggesting its optimum may be higher than 1.25e-3;
- doubles Adam's moment horizon measured in tokens when beta2 is held at 0.95;
- suffers more from the bad initialization because it gets half as many corrective updates.

A [NeurIPS 2025 small-batch LM study](https://arxiv.org/abs/2507.07101) reports that small batches are robust and efficient per FLOP, recommends avoiding accumulation on a single replica, and proposes keeping Adam moment half-life fixed in tokens. For batch 8 → 16, that rule maps `beta2=0.95` to `0.95²=0.9025`. Treat this as an ablation, not a default.

Gradient accumulation offers no hidden advantage here. With dropout zero, `BATCH_SIZE=8, GRAD_ACCUM_STEPS=2` is approximately an effective batch of 16 with fewer optimizer updates. It is useful when memory forces it or as a correctness check; it does not preserve the dynamics that made batch 8 better.

### The original CSV could not answer dropout or weight decay

The registry contains 158 successful runs, but every row records:

- dropout 0;
- weight decay 0;
- gradient accumulation 1;
- Adam `(beta1, beta2) = (0.9, 0.95)` and epsilon `1e-8`.

Your earlier dropout/WD tests may have happened, but they are not recoverable from this CSV. More importantly, weight decay on the current std≈1 tied embedding is a qualitatively different experiment from weight decay after sane initialization.

This paragraph describes the state of the registry when the plan was first written. Later recorded sweeps did test grouped weight decay after corrected initialization and selected `WEIGHT_DECAY=0.05`; the tied embedding and RMSNorm parameters are excluded from decay.

## Results added after the original plan

### Initialization, schedule, AdamW, and tokenizer

The staged order avoided confounding the optimizer experiments with the original embedding scale:

| Original initialization probe | Embedding std | Logit std | Random-target CE |
|---|---:|---:|---:|
| PyTorch defaults | 1.0001 | 21.651 | 379.288 |
| Linear + Embedding std `0.02`, biases zero | 0.0200 | 0.424 | 9.007 |

For the original 8,192-token model, `ln(V)=9.011`, while the actual `35m_tok8k_4b` run started at loss 415.94. This localized much of the early gradient problem to the tied embedding/output scale and justified making initialization the first intervention.

1. `gpt_scaled` initialization fixed the extreme initial cross-entropy caused by the tied embedding/output matrix initialized near unit standard deviation.
2. The schedule sweep selected WSD with an 80/20 stable/decay split. The initial 4k result used peak LR `1.5e-3` and approximately 6M warmup tokens.
3. Re-tuning AdamW and batch size selected batch 16 with `beta2=0.9025`; the follow-up LR/WD sweep selected `LR=3.0e-3`, `WEIGHT_DECAY=0.05` for that 4k screening setup.
4. QK norm survived its factorial ablation and is enabled in the tuned baselines; z-loss did not.
5. The tokenizer/model-size experiments selected 16k. Per-size LR tuning then produced the current `2.4e-3`, `2.3e-3`, `2.2e-3`, and `2.1e-3` ladder for the 10M, 20M, 35M, and 50M configurations.

These values should not be mixed indiscriminately: `3.0e-3` was the best point in the earlier 4k AdamW/WD screen, while the lower LR ladder is the result of subsequent 16k and model-size tuning.

### Long-horizon z-loss result

Z-loss coefficient `1e-5` was compared with the tuned 10M/16k baseline on the `907m` and `1b9` dataset variants. The z-loss runs used microbatch 8 with two accumulation steps because the float32 log-normalizer graph did not fit at batch 16; effective batch size and the number of validation tokens were kept equal.

| Dataset variant | Approx. train tokens | Δ final train CE | Δ final validation CE | Decision |
|---|---:|---:|---:|---|
| `907m` | 213M | +0.01045 | -0.00333 | Too small to promote |
| `1b9` | 459M | +0.01363 | +0.00952 | Regression |

Negative validation delta is better. The apparent short-run gain reverses at the longer horizon, while train CE is worse at both endpoints. The z-loss contribution itself grows from roughly `0.0010` to `0.0018` in the long run.

This is not an effective global LR increase. In the last 10% of the `1b9` run, global gradient norm is 3.6% higher but the actual AdamW update norm is 0.7% lower. The update/parameter ratio is only 0.9% higher because parameter norm is 1.5% lower. Layer gradients are redistributed rather than uniformly scaled: attention-output and FFN-input/gate gradients rise while FFN-output gradients fall.

Z-loss targets `logsumexp(logits)`, not raw logit RMS. The raw-logit diagnostics are inconsistent across horizons: late median logit RMS is about 21% higher on `907m` but 15% lower on `1b9`, while the long-run absolute maximum is higher. Moreover, the diagnostic samples only the final microbatch, so the batch-8 accumulated runs are not perfectly matched to the batch-16 control for this metric. The combination of no durable CE benefit, inconsistent logit control, and substantial VRAM overhead is sufficient to reject z-loss for the main recipe.

### Norm learning-rate multiplier result

`NORM_LR_MULTIPLIER=0.5` was tested on the same two horizons with batch 16 and otherwise identical settings. It halves the scheduled LR only for RMSNorm parameters, which are already excluded from weight decay.

| Dataset variant | Approx. train tokens | Δ final train CE | Δ final validation CE | Decision |
|---|---:|---:|---:|---|
| `907m` | 213M | +0.00683 | -0.00301 | Too small to promote |
| `1b9` | 459M | +0.00935 | +0.00711 | Regression |

Again, the tiny short-run validation gain reverses at the longer horizon. The intervention also fails its mechanistic goal: over the last 10% of training, the RMSNorm gradient-group norm is 22.3% higher on `907m` and 23.3% higher on `1b9`. This does not mean the norm parameters received larger updates—their LR was explicitly halved—but it shows that slowing them caused the optimization trajectory to present larger residual gradients to those parameters rather than calming the signal.

The global AdamW update norm changes by only -0.4% on `907m` and -0.03% on `1b9` over the same late region. Late logit RMS is also about 3.7% and 4.9% higher, respectively. There is therefore no validation, global-update, or logit-scale evidence for keeping the multiplier. Retain `NORM_LR_MULTIPLIER=1.0`.

### Updated interpretation of the long-run telemetry

The continuously growing raw gradient and logit curves are worth monitoring, but neither rejected intervention improved the long-horizon objective. The actual update norm still follows the WSD learning-rate decay, and the growing gradients are concentrated differently across parameter groups rather than producing a global update explosion. Checkpoint probes also indicate that much of the raw-logit growth is common-mode; centered-logit scale is substantially more stable. Future diagnostics should log centered logit RMS and the log-normalizer directly, and should add per-group update/parameter ratios before another group-specific LR intervention is attempted.

## Prioritized experiment program

### Tier 0 — fix the baseline and instrumentation (completed)

Add two explicit initialization choices, keeping current defaults only as the control:

- `olmo_002`: every Linear/Embedding weight `N(0, 0.02²)`, biases zero, RMSNorm scales one.
- `gpt_scaled`: the same, but attention output projections and FFN down projections use `0.02 / sqrt(2 * N_BLOCKS)`.

Run default vs these two recipes on the same 195m-labelled 8k prefix and seed. Promote the winner only if its initial CE is near `ln(V)`, early clipping collapses, and final validation does not regress. Repeat the winner on a second seed.

Result: `gpt_scaled` was promoted and the requested instrumentation was added. The initializer is now part of every tuned configuration.

Also make RMSNorm epsilon explicit and recorded. Add:

- embedding/parameter RMS;
- Q/K and attention-logit RMS/max, plus attention entropy;
- output-logit RMS/max;
- global and per-module pre-clip norm;
- clip coefficient and fraction of steps clipped;
- optimizer update RMS and update/weight ratio;
- tokens/s, elapsed time, and peak memory.

For a compact spike metric, copy OLMo 2's definition: percentage of points at least seven standard deviations from a rolling 1,000-step mean. Report it before and after warmup separately. Small-scale experiments have shown that attention/output-logit instabilities and their remedies can be reproduced without billion-parameter models ([Wortsman et al.](https://arxiv.org/abs/2309.14322)).

### Tier 1 — tune the corrected AdamW schedule (completed for the baseline)

Use warmup tokens, not a fixed 10 steps across every horizon and batch. For batch 8 × 1024 tokens, screen:

| Warmup tokens | Optimizer steps |
|---:|---:|
| ~1M | 122 |
| ~4M | 488 |
| ~8M | 977 |

The original long run warmed for only 81,920 tokens. OLMo 2 explicitly used shortened warmup to reproduce spikes and uses 2,000 steps in its main recipe.

Correct initialization will shift the LR optimum, so re-sweep `0.75e-3, 1.0e-3, 1.25e-3, 1.5e-3`. Use successive halving rather than a full warmup × LR Cartesian product.

Then compare schedules on the 907m-labelled 8k prefix (~249M actual tokens):

| Schedule | Purpose |
|---|---|
| Cosine → 0 | Existing control |
| Cosine → 0.1 × peak | Continued-learning checkpoint |
| WSD 80% stable + 20% linear → 0 | Main candidate |
| WSD 90% stable + 10% linear → 0 | Short cooldown candidate |

For the final base model, prefer a cooldown-to-zero winner. Also retain the pre-cooldown WSD checkpoint if you expect to extend pretraining later.

Result: WSD 80/20 was promoted. Peak LR was subsequently re-tuned for the 16k tokenizer and each model size rather than copied from the initial 4k schedule sweep.

### Tier 2 — weight decay and batch-size recovery (completed for the baseline)

Re-test WD only after initialization is fixed:

- decay hidden 2D Linear weights;
- exclude the tied embedding/output matrix, RMSNorm weights, and biases;
- sweep `0, 0.01, 0.05, 0.1` cheaply, promoting at most two settings.

[AdamW](https://arxiv.org/abs/1711.05101) decouples weight decay from the loss gradient, but its shrink still occurs as `1 - lr * wd` each step. Therefore a separate WD schedule is unnecessary initially. Compare the integrated decay budget `wd * sum(lr_t)`: WSD keeps LR high longer than cosine, so identical numeric WD does not mean identical total shrink. [OLMo 2 found that decaying embeddings drove their norm too low and increased gradient growth and spikes](https://arxiv.org/html/2501.00656v3).

Keep dropout at zero unless a meaningful and growing train-validation gap appears. It adds noise and does not target the observed mechanisms.

After this, re-test batch size:

| Variant | Batch | LR | beta2 | Warmup |
|---|---:|---:|---:|---|
| Control | 8 | tuned | 0.95 | fixed tokens |
| Larger batch | 16 | sweep 1.4–2.0× control | 0.95 | same tokens |
| Token-scaled moments | 16 | same LR sweep | 0.9025 | same tokens |
| Middle point | 12 | tune locally | 0.9259 | same tokens |

Measure throughput on the L4. Select the loss-at-fixed-wall-time Pareto point, not the highest raw tokens/s.

Result: batch 16, token-scaled `beta2=0.9025`, and grouped weight decay `0.05` were promoted. The tuned implementation excludes the tied embedding/output matrix, RMSNorm weights, and biases from decay.

### Tier 3 — QK norm and z-loss (completed)

Run one clean factorial:

| Run | QK norm | z-loss coefficient |
|---|---|---:|
| A | off | 0 |
| B | on | 0 |
| C | off | 1e-5 or 1e-4 |
| D | on | same |

[QK norm](https://arxiv.org/abs/2010.04245) prevents arbitrary attention-softmax saturation. In this model, use per-head RMSNorm on Q and K before RoPE/SDPA, retaining native SDPA. Z-loss adds `c * mean(logsumexp(logits)^2)` in float32 and discourages final logits from growing. OLMo 2's detailed section uses `1e-4`, while its summary table lists `1e-5`, so bracket both rather than assuming universality.

Promote only if final validation improves, or if spikes fall materially with no validation regression and the model tolerates a higher LR.

Result: QK norm was promoted. Z-loss was rejected after the initial factorial and the longer `907m`/`1b9` follow-up described above. A follow-up `NORM_LR_MULTIPLIER=0.5` experiment was also rejected; it should not be treated as a remaining Tier 3 candidate.

If clipping remains frequent after initialization and warmup are fixed, compare global clipping with StableAdamW-style **update** clipping. The original work found loss spikes 1–8 steps after Adam's second moment underestimated squared gradients, and its AdamW–Adafactor hybrid outperformed global gradient clipping in that setting ([Wortsman et al.](https://arxiv.org/abs/2304.13013)).

### Tier 4 — optimizer alternatives

Try **Muon first**. PyTorch 2.9.1 in this environment already exposes `torch.optim.Muon`, so no new dependency is required. Use Muon only for hidden 2D Linear weights and AdamW for the tied embedding, norms, and biases, as the [PyTorch Muon documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html) requires. `adjust_lr_fn="match_rms_adamw"` is a sensible controlled variant, but still sweep 0.5×/1×/2× around the tuned Adam-equivalent LR.

The [Moonshot Muon paper](https://arxiv.org/abs/2502.16982) reports roughly 2× compute efficiency at much larger scale. That is encouraging, not a promise for a 30–50M model on one L4. Compare final validation, wall-clock to fixed loss, tokens/s, and peak memory.

If you later want a second-order lane, prefer [SOAP](https://arxiv.org/abs/2409.11321) over raw Shampoo. SOAP reports >35% wall-clock improvement over AdamW for 360M/660M large-batch LM training, but it adds a third-party implementation and preconditioner overhead in a regime unlike yours. It ranks below native Muon.

### Tier 5 — small architecture variants

Initialization and QK norm are now completed prerequisites. The remaining architecture order is:

1. Head-specific sigmoid gate after SDPA — a [large 2025 study](https://arxiv.org/abs/2505.06708) found this placement improved quality, stability, and LR tolerance.
2. [Exclusive Self Attention](https://arxiv.org/abs/2603.09078) — promising results up to 2.7B, but new, not primarily a spike remedy, and reportedly more valuable at longer contexts than your 1024.
3. Residual/ReZero gates — [useful evidence exists](https://arxiv.org/abs/2003.04887), especially for very deep networks, but the current 8–16-block range makes this lower priority.

Do not combine these in one “modern architecture” run. Each change should survive a medium horizon and at least two seeds independently.

## Run horizons and promotion rules

| Stage | 16k artifact | Actual tokens | Use |
|---|---|---:|---|
| Screen | `90m` | 21.2M | reject bad ideas cheaply |
| Promote | `421m` | 98.9M | check that the early ranking persists |
| Long confirm | `907m` / `1b9` | 213M / 459M | expose horizon-dependent reversals |
| Final | `4b` | 989M | top one or two recipes |

Use validation CE at fixed tokens as the primary metric. For finalists, increase from 20 to at least 100 fixed validation batches. Secondary metrics are post-warmup clip fraction, p99 norm, spike score, attention/output-logit scale, update RMS, tokens/s, memory, and wall-clock to target validation losses.

Promote a change when it produces roughly ≥0.01 validation-loss improvement at the confirm horizon or a clear wall-clock Pareto improvement, and reproduces on another seed. A stability-only change must materially reduce spikes/clipping without hurting validation.

## Experiment queue and status

| Item | Status | Result or next action |
|---|---|---|
| Telemetry and initialization | Completed | `gpt_scaled` promoted |
| LR, warmup, and schedule | Completed | WSD 80/20; per-size LR tuning |
| Grouped WD and batch size | Completed | WD `0.05`, batch 16, `beta2=0.9025` |
| QK norm × z-loss | Completed | QK norm on; z-loss off |
| Tokenizer selection | Completed | 16k selected |
| Long z-loss confirmation | Rejected | Short gain reversed at `1b9`; high VRAM cost |
| Norm LR multiplier `0.5` | Rejected | Short gain reversed at `1b9`; norm gradients increased |
| Tuned AdamW vs native Muon | Next | Match token budget, wall time, and validation protocol |
| Attention-output gate | Pending | Test only after optimizer comparison |
| XSA, residual gates, or SOAP | Deferred | Require independent medium-horizon ablations |

## Important limitations

- The final validation numbers above use the fixed short validation window configured during training: seven batches at batch 16, or fourteen batches at batch 8. The token count is matched, but finalists should still be evaluated over the full validation artifact.
- The z-loss comparison changes physical microbatching from 16×1 to 8×2. The effective gradient batch is matched, but kernels, roundoff, performance, and the last-microbatch logit diagnostic are not identical.
- Gradient-group norms are logged, but group-specific parameter norms and actual update norms are not. Consequently, the norm-LR experiment proves that validation did not improve; it does not directly measure the isolated RMSNorm update/weight ratio.
- The long z-loss and norm-LR comparisons are single-seed ablations. Their short-horizon deltas are below the plan's promotion threshold and reverse at the longer horizon, so another seed is not a good use of compute unless full-validation evaluation changes the ranking.
- Dataset variant names such as `907m` and `1b9` are artifact labels, not the number of 16k-tokenizer training tokens. The corresponding runs processed approximately 213M and 459M tokens.
- The batch throughput numbers are from the available local Fedora logs, not a controlled L4 benchmark.
- WSD, Muon, gated attention, and XSA evidence mostly comes from larger models and different corpora. They justify tests, not automatic adoption.
- OLMo 2 found QK norm and reordered norm strongest together, while other work finds QK norm useful by itself. That disagreement justified the completed explicit QK ablation instead of copying the whole OLMo 2 block.
