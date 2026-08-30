# Pavullmo pretraining dynamics: diagnosis and experiment plan

_29 August 2026 · Scope: optimizer, schedule, stabilization, and small architecture changes. Dataset work is intentionally excluded._

## The short answer

The next experiment should not be Muon, Shampoo, a residual gate, or XSA. First fix and ablate initialization.

The current model creates an `nn.Embedding` and uses the same matrix as the output head, but never applies a custom initializer. [PyTorch initializes `nn.Embedding.weight` from `N(0,1)`](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.sparse.Embedding.html). In a local check of the 35m/8k architecture, that gives:

| Initializer | Embedding std | Logit std | Random-target CE |
|---|---:|---:|---:|
| Current PyTorch defaults | 1.0001 | 21.651 | 379.288 |
| Linear + Embedding std 0.02, biases zero | 0.0200 | 0.424 | 9.007 |

For 8,192 tokens, `ln(V) = 9.011`. The actual `35m_tok8k_4b` run starts at loss **415.94**, so a large part of the early gradient problem is self-inflicted by the tied embedding/output scale. [OLMo 2's stability work](https://arxiv.org/html/2501.00656v3) independently found that initializing all parameters at std 0.02 preserved activation/gradient scale better and reduced its gradient spike score from 0.40 to 0.03.

After that correction, the highest-value sequence is:

1. Tune warmup and replace full-run cosine with a warmup-stable-decay experiment.
2. Re-test weight decay using parameter groups that do **not** decay the tied embedding.
3. Re-test batch size with LR and Adam moments scaled for the batch—not just the same optimizer settings.
4. Test QK norm × z-loss as a 2×2 stability ablation.
5. Compare a tuned AdamW baseline with native PyTorch Muon.
6. Only then try gated attention, XSA, residual gates, or SOAP/Shampoo.

## What the existing runs actually say

### The late gradient increase is mostly not an instability

`train/gradient_norm` is the **pre-clipping** value returned by `clip_grad_norm_`; every value above 1.0 is subsequently clipped. In the 141,337-step `35m_tok8k_4b` run:

| Training region | Median norm | p99 norm | Steps clipped |
|---|---:|---:|---:|
| First decile | 0.657 | 3.700 | 20.12% |
| Second decile | 0.361 | 1.274 | 1.30% |
| Last decile | 0.535 | 0.892 | 0.52% |
| After first 10%, overall | — | — | 0.91% |

So the median does fall and then rise, but it remains well below the clipping threshold and late clipping is rare. I would treat that slow rise as telemetry to watch, not as exploding gradients.

The real issue is the isolated tail: pre-clip norms reach 3,064, 2,461, 1,364, and 1,244 at ordinary batch losses of roughly 3.25–3.87. Loss alone cannot localize these events. Add per-module gradient/update and attention-logit logging before choosing a remedy.

### The final slowdown follows the LR curve almost exactly

The current cosine falls below 10% of peak for approximately the final 20% and reaches zero. Validation improvement by decile changes as follows:

| Decile | LR at end | Validation improvement |
|---:|---:|---:|
| 2 | 1.131e-3 | 0.310 |
| 5 | 0.625e-3 | 0.071 |
| 8 | 0.119e-3 | 0.026 |
| 9 | 0.0306e-3 | 0.016 |
| 10 | 0 | 0.005 |

This does not mean “never decay.” A terminal cooldown is useful. It means that a cosine spanning the whole run spends too much of this particular token budget at a low LR. [Warmup-stable-decay (WSD)](https://arxiv.org/abs/2410.05192) keeps a high-LR branch that can continue training, then applies a shorter cooldown to produce a terminal checkpoint.

### Batch 16 is worse, but the old sweep is incomplete

At equal tokens, the matched 35m/8k/42m sweeps give best validation loss **5.229** for batch 8 and **5.600** for batch 16. Available wall times show only about a **6.6% throughput gain** for batch 16 (roughly 87.5k → 93.3k tokens/s), so “train a few more tokens in the same time” is unlikely to close that gap on the measured GPU.

However, batch 16 also:

- halves the number of parameter updates;
- reaches the top of the tested LR grid, suggesting its optimum may be higher than 1.25e-3;
- doubles Adam's moment horizon measured in tokens when beta2 is held at 0.95;
- suffers more from the bad initialization because it gets half as many corrective updates.

A [NeurIPS 2025 small-batch LM study](https://arxiv.org/abs/2507.07101) reports that small batches are robust and efficient per FLOP, recommends avoiding accumulation on a single replica, and proposes keeping Adam moment half-life fixed in tokens. For batch 8 → 16, that rule maps `beta2=0.95` to `0.95²=0.9025`. Treat this as an ablation, not a default.

Gradient accumulation offers no hidden advantage here. With dropout zero, `BATCH_SIZE=8, GRAD_ACCUM_STEPS=2` is approximately an effective batch of 16 with fewer optimizer updates. It is useful when memory forces it or as a correctness check; it does not preserve the dynamics that made batch 8 better.

### The CSV cannot answer dropout or weight decay yet

The registry contains 158 successful runs, but every row records:

- dropout 0;
- weight decay 0;
- gradient accumulation 1;
- Adam `(beta1, beta2) = (0.9, 0.95)` and epsilon `1e-8`.

Your earlier dropout/WD tests may have happened, but they are not recoverable from this CSV. More importantly, weight decay on the current std≈1 tied embedding is a qualitatively different experiment from weight decay after sane initialization.

## Prioritized experiment program

### Tier 0 — fix the baseline and instrumentation

Add two explicit initialization choices, keeping current defaults only as the control:

- `olmo_002`: every Linear/Embedding weight `N(0, 0.02²)`, biases zero, RMSNorm scales one.
- `gpt_scaled`: the same, but attention output projections and FFN down projections use `0.02 / sqrt(2 * N_BLOCKS)`.

Run default vs these two recipes on the same 195m-labelled 8k prefix and seed. Promote the winner only if its initial CE is near `ln(V)`, early clipping collapses, and final validation does not regress. Repeat the winner on a second seed.

Also make RMSNorm epsilon explicit and recorded. Add:

- embedding/parameter RMS;
- Q/K and attention-logit RMS/max, plus attention entropy;
- output-logit RMS/max;
- global and per-module pre-clip norm;
- clip coefficient and fraction of steps clipped;
- optimizer update RMS and update/weight ratio;
- tokens/s, elapsed time, and peak memory.

For a compact spike metric, copy OLMo 2's definition: percentage of points at least seven standard deviations from a rolling 1,000-step mean. Report it before and after warmup separately. Small-scale experiments have shown that attention/output-logit instabilities and their remedies can be reproduced without billion-parameter models ([Wortsman et al.](https://arxiv.org/abs/2309.14322)).

### Tier 1 — tune the corrected AdamW schedule

Use warmup tokens, not a fixed 10 steps across every horizon and batch. For batch 8 × 1024 tokens, screen:

| Warmup tokens | Optimizer steps |
|---:|---:|
| ~1M | 122 |
| ~4M | 488 |
| ~8M | 977 |

The current long run warms for only 81,920 tokens. OLMo 2 explicitly used shortened warmup to reproduce spikes and uses 2,000 steps in its main recipe.

Correct initialization will shift the LR optimum, so re-sweep `0.75e-3, 1.0e-3, 1.25e-3, 1.5e-3`. Use successive halving rather than a full warmup × LR Cartesian product.

Then compare schedules on the 907m-labelled 8k prefix (~249M actual tokens):

| Schedule | Purpose |
|---|---|
| Cosine → 0 | Existing control |
| Cosine → 0.1 × peak | Continued-learning checkpoint |
| WSD 80% stable + 20% linear → 0 | Main candidate |
| WSD 90% stable + 10% linear → 0 | Short cooldown candidate |

For the final base model, prefer a cooldown-to-zero winner. Also retain the pre-cooldown WSD checkpoint if you expect to extend pretraining later.

### Tier 2 — weight decay and batch-size recovery

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

### Tier 3 — QK norm and z-loss

Run one clean factorial:

| Run | QK norm | z-loss coefficient |
|---|---|---:|
| A | off | 0 |
| B | on | 0 |
| C | off | 1e-5 or 1e-4 |
| D | on | same |

[QK norm](https://arxiv.org/abs/2010.04245) prevents arbitrary attention-softmax saturation. In this model, use per-head RMSNorm on Q and K before RoPE/SDPA, retaining native SDPA. Z-loss adds `c * mean(logsumexp(logits)^2)` in float32 and discourages final logits from growing. OLMo 2's detailed section uses `1e-4`, while its summary table lists `1e-5`, so bracket both rather than assuming universality.

Promote only if final validation improves, or if spikes fall materially with no validation regression and the model tolerates a higher LR.

If clipping remains frequent after initialization and warmup are fixed, compare global clipping with StableAdamW-style **update** clipping. The original work found loss spikes 1–8 steps after Adam's second moment underestimated squared gradients, and its AdamW–Adafactor hybrid outperformed global gradient clipping in that setting ([Wortsman et al.](https://arxiv.org/abs/2304.13013)).

### Tier 4 — optimizer alternatives

Try **Muon first**. PyTorch 2.9.1 in this environment already exposes `torch.optim.Muon`, so no new dependency is required. Use Muon only for hidden 2D Linear weights and AdamW for the tied embedding, norms, and biases, as the [PyTorch Muon documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html) requires. `adjust_lr_fn="match_rms_adamw"` is a sensible controlled variant, but still sweep 0.5×/1×/2× around the tuned Adam-equivalent LR.

The [Moonshot Muon paper](https://arxiv.org/abs/2502.16982) reports roughly 2× compute efficiency at much larger scale. That is encouraging, not a promise for a 30–50M model on one L4. Compare final validation, wall-clock to fixed loss, tokens/s, and peak memory.

If you later want a second-order lane, prefer [SOAP](https://arxiv.org/abs/2409.11321) over raw Shampoo. SOAP reports >35% wall-clock improvement over AdamW for 360M/660M large-batch LM training, but it adds a third-party implementation and preconditioner overhead in a regime unlike yours. It ranks below native Muon.

### Tier 5 — small architecture variants

My order would be:

1. Explicit initialization — mandatory baseline repair.
2. QK norm — directly targets attention-logit instability.
3. Head-specific sigmoid gate after SDPA — a [large 2025 study](https://arxiv.org/abs/2505.06708) found this placement improved quality, stability, and LR tolerance.
4. [Exclusive Self Attention](https://arxiv.org/abs/2603.09078) — promising results up to 2.7B, but new, not primarily a spike remedy, and reportedly more valuable at longer contexts than your 1024.
5. Residual/ReZero gates — [useful evidence exists](https://arxiv.org/abs/2003.04887), especially for very deep networks, but 11–16 blocks and a broken initializer make this low priority.

Do not combine these in one “modern architecture” run. Each change should survive a medium horizon and at least two seeds independently.

## Run horizons and promotion rules

| Stage | 8k artifact | Actual tokens | Use |
|---|---|---:|---|
| Screen | 195m-labelled | ~53.6M | reject bad ideas cheaply |
| Confirm | 907m-labelled | ~249.3M | two-seed comparison |
| Final | 4b-labelled | 1.158B | top one or two recipes |

Use validation CE at fixed tokens as the primary metric. For finalists, increase from 20 to at least 100 fixed validation batches. Secondary metrics are post-warmup clip fraction, p99 norm, spike score, attention/output-logit scale, update RMS, tokens/s, memory, and wall-clock to target validation losses.

Promote a change when it produces roughly ≥0.01 validation-loss improvement at the confirm horizon or a clear wall-clock Pareto improvement, and reproduces on another seed. A stability-only change must materially reduce spikes/clipping without hurting validation.

## The first concrete queue

1. Add telemetry and `pytorch_default` / `olmo_002` / `gpt_scaled` initializers.
2. Compare those three on 195m/8k, seed 42; repeat the winner on seed 43.
3. Tune LR and token-based warmup on the winning initializer.
4. Compare cosine0, cosine0.1, WSD80/20, and WSD90/10 at ~249M tokens.
5. Sweep grouped WD and re-test batches 8/12/16.
6. Run QK norm × z-loss.
7. Compare tuned AdamW with native Muon.
8. Only then test gated attention, XSA, or SOAP.

## Important limitations

- The 35m long run is not in the successful-run CSV; its complete TensorBoard stream is the evidence source.
- No per-layer gradients, attention logits, update norms, or batch IDs were logged, so isolated post-warmup spikes cannot yet be assigned to data, attention saturation, or Adam state.
- The batch throughput numbers are from the available local Fedora logs, not a controlled L4 benchmark.
- WSD, Muon, gated attention, and XSA evidence mostly comes from larger models and different corpora. They justify tests, not automatic adoption.
- OLMo 2 found QK norm and reordered norm strongest together, while other work finds QK norm useful by itself. That disagreement is why the plan calls for an explicit QK ablation instead of copying the whole OLMo 2 block.
