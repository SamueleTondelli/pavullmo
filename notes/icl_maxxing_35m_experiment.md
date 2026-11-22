# 35M ICL-Maxxing Experiment (`35m_icl_maxxing_1b`)

This note records the code changes, controlled ablations, multi-seed optimizer
sweep, schedule/EMA comparison, and final 1B-token local training run carried
out on the `model/icl-maxxing` branch after rebasing onto `9a95ab3`
(`origin/pretrain-evaluation`).

All experiments used the existing 16,384-token SentencePiece BPE tokenizer
(`production_16k`, SHA-256 `88575f7c0f8891e1caf885a4fbfbc453604170422f6571ea75e0a5d807272890`;
tokenizer retraining was deferred) and the local `epfml/FineWeb2-HQ` Italian
filtered dataset (`artifacts/datasets/fineweb2-hq-filter`: `train_full` with
1.674B tokens, `validation` with 10.00M tokens, untouched `test` with 10.00M
tokens). Local training ran on an RTX 4060 Laptop GPU (7.7 GiB VRAM) using
`batch_size = 8, grad_accum_steps = 2` (identical 16,384-token effective batch
size to `batch_size = 16, grad_accum_steps = 1` with `dropout = 0.0`).

---

## 1. Code and Configuration Changes

1. **Branch rebase onto `9a95ab3` (`33a2d17`)**
   - Rebased the previous `model/icl-maxxing` commit (`97e4ffa`) onto the
     refactored TOML/Pydantic `PretrainingTrainer` codebase (`9a95ab3`).
   - Updated `ModelConfig` defaults in `src/config/schema.py` to
     `n_blocks = 13`, `num_kv_heads = 2`, added `num_kv_heads = 1` to
     `src/pavullmo/configs/35m_config.toml`, restored `KV_HEADS` alias and
     weight-based `num_kv_heads` inference in
     `src/pavullmo/migrate_checkpoint_config.py`, and ported GQA/CUDA tests to
     `tests/test_gqa_config.py`.
2. **Trainer, optimizer, schedule, and EMA extensions (`e40d02a`)**
   - **Embedding weight decay** (`optimizer.adamw.embedding_weight_decay: bool = False`):
     Previously, under `optimizer.kind = "muon_adamw"` with
     `canon_layers = false`, AdamW's decayed parameter group was empty (all 2D
     hidden matrices went to Muon, while embeddings, biases, and norms received
     zero decay). Setting `embedding_weight_decay = true` routes
     `embeddings.weight` into AdamW's decayed group so `optimizer.adamw.weight_decay`
     acts as an explicit experimental axis while norms and 1D biases remain
     undecayed.
   - **Constant LR schedule** (`optimizer.schedule.kind = "constant"`): Linear
     warmup for `warmup_steps`, then constant `1.0` multiplier.
   - **Protected pre-decay checkpointing** (`checkpoint.save_pre_decay: bool = False`):
     Automatically validates and saves a checkpoint at `decay_start_step` (the
     step immediately before WSD cooldown begins) and protects it from
     `checkpoint.keep_last` pruning.
   - **FP32 weight EMA** (`training.ema_decay`, `training.ema_warmup`):
     Maintains FP32 shadow weights updated after every optimizer step with
     optional warmup `min(ema_decay, (1 + step) / (10 + step))`. Evaluates both
     raw and EMA weights during validation (`validation/loss` and
     `validation/ema_loss`), saves both `model` and `ema_model` inside
     checkpoints without overwriting raw weights, restores EMA state on resume,
     and supports `--ema` loading in `src/pavullmo/generate.py`,
     `src/evaluation/evaluate_loss.py`, and
     `src/evaluation/score_synthetic_benchmark.py`.
   - Unit tests added in `tests/test_training_extensions.py`.
3. **Winning 35M preset (`8a80b11`)**
   - Added `src/pavullmo/configs/35m_icl_maxxing_config.toml` capturing the final
     architecture, optimizer, schedule, and EMA recipe selected by the study
     below.

---

## 2. Ablation 1 — Head Geometry and MHA vs. GQA/MQA

Fixed settings: `n_blocks = 12` (+1 layer over the 11-layer `35m_config.toml`),
`embed_dim = 448`, `ffn_dim = 1216`, `seq_len = 1024`, `rope_base = 10,000`,
`canon_layers = false`, `qk_norm = true`, `split_qkv_projections = true`,
`lr = 2.4e-3`, `muon.lr_multiplier = 1.0`, `muon.weight_decay = 0.05`,
`adamw.weight_decay = 0.05` (`embedding_weight_decay = true`), WSD 40% decay,
1,500 steps (24.58M tokens), seed `42`. Evaluated on the full 10M-token
validation split and all 30 Italian synthetic benchmarks (6 categories).

| Run | Heads × `head_dim` | `num_kv_heads` (`kv_dim`) | Parameters | Full 10M Val Loss | Bench Bidir | `icl` Bidir | Steady tok/s |
|---|---|---|---:|---:|---:|---:|---:|
| `abl1_h4x112_mha` | 4 × 112 | 4 (448) | 36,633,920 | **3.79177** | 0.3088 | 0.350 | 51,015 |
| **`abl1_h4x112_gqa2`** | **4 × 112** | **2 (224)** | **34,225,472** | **3.79340** | **0.3096** | **0.358** | **54,581** |
| `abl1_h7x64_mha` | 7 × 64 | 7 (448) | 36,632,768 | 3.80560 | 0.3106 | 0.332 | 52,464 |
| `abl1_h8x56_mha` | 8 × 56 | 8 (448) | 36,632,576 | 3.80884 | **0.3129** | 0.338 | 50,652 |
| `abl1_h4x112_mqa1` | 4 × 112 | 1 (112) | 33,021,248 | 3.81439 | 0.3059 | 0.332 | 55,821 |
| `abl1_h8x56_gqa2` | 8 × 56 | 2 (112) | 33,019,904 | 3.84044 | 0.3046 | 0.330 | 55,657 |
| `abl1_h7x64_mqa1` | 7 × 64 | 1 (64) | 32,504,000 | 3.85721 | 0.3009 | 0.309 | 57,285 |

**Decision:** Selected **`4×112` GQA-2 (`attn_heads = 4, num_kv_heads = 2`)**.
- `4×112` outperformed both `7×64` (-0.0138 nats under MHA) and `8×56` (-0.0171
  nats under MHA) on full validation loss and `icl` accuracy.
- `8×56` MHA compiled slower than `7×64` MHA (50.7k vs. 52.5k tok/s) because
  `head_dim = 56` pads internally to 64 in FlashAttention-2.
- `4×112` GQA-2 matched `4×112` MHA within 0.0016 nats while using 2.41M fewer
  parameters (34.23M total, matching the original 11-block MHA 35M parameter
  budget at 12 layers), achieving the highest `icl` score (0.358), halving KV
  cache size, and increasing throughput by +7.0% (54.6k tok/s). Aggressive
  4:1/7:1 KV compression (`8×56` GQA-2, `7×64` MQA-1) degraded loss sharply.

---

## 3. Ablation 2 — RoPE Base

Evaluated on the selected `4×112` GQA-2 12-block architecture at 1,500 steps
(24.58M tokens), seed `42`:

| `rope_base` | Full 10M Val Loss | Overall Bench Bidir | `icl` Bidir | `reasoning` Bidir |
|---:|---:|---:|---:|---:|
| **50,000** | **3.79004** | 0.3103 | 0.3490 | 0.1912 |
| 25,000 | 3.79196 | **0.3134** | **0.3637** | **0.1960** |
| 10,000 | 3.79340 | 0.3096 | 0.3585 | 0.1947 |

**Decision:** Selected **`rope_base = 50,000.0`**, which improved full
validation loss monotonically over 10k and 25k (-0.0034 nats vs. 10k) and
prepares the model for future context-length extension.

---

## 4. Staged Multi-Seed Optimizer Sweep

Fixed architecture (`12 blocks`, `4×112` GQA-2, `rope_base = 50,000`,
`canon_layers = false`) and frozen secondary optimizer hyperparameters
(`muon.momentum = 0.95`, `muon.nesterov = true`, `muon.ns_iters = 5`,
`adamw.betas = (0.9, 0.99)`, `adamw.eps = 1e-8`,
`adamw.embedding_weight_decay = true`), WSD 40% decay, 1,200 steps (19.66M
tokens) per run.

Staged search order: Peak LR (`1.8e-3, 2.4e-3, 3.0e-3`) → Muon LR multiplier
(`0.5, 0.75, 1.0`) → Muon weight decay (`0.02, 0.05, 0.10`) → Embedding/AdamW
weight decay (`0.02, 0.05, 0.10`), initially with 2 seeds (`42, 43`), then
promoting the top 3 candidates to 3 seeds (`42, 43, 44`).

### 2-Seed Stage Summary

| Peak LR | `muon.lr_mult` | `muon.wd` | `adamw.wd` (emb) | Full 10M Val Loss (mean ± SE) | Bench Bidir |
|---:|---:|---:|---:|---:|---:|
| **0.0030** | **0.50** | **0.10** | **0.02** | **3.86326 ± 0.00097** | 0.3105 |
| 0.0030 | 0.50 | 0.10 | 0.05 | 3.86386 ± 0.00124 | 0.3114 |
| 0.0030 | 0.50 | 0.10 | 0.10 | 3.86461 ± 0.00102 | 0.3120 |
| 0.0030 | 0.50 | 0.05 | 0.05 | 3.86487 ± 0.00067 | **0.3137** |
| 0.0030 | 0.50 | 0.02 | 0.05 | 3.86495 ± 0.00109 | 0.3059 |
| 0.0030 | 0.75 | 0.05 | 0.05 | 3.86813 ± 0.00277 | 0.3124 |
| 0.0030 | 1.00 | 0.05 | 0.05 | 3.87315 ± 0.00409 | 0.3042 |
| 0.0024 | 1.00 | 0.05 | 0.05 | 3.87399 ± 0.00182 | 0.3033 |
| 0.0018 | 1.00 | 0.05 | 0.05 | 3.88622 ± 0.00193 | 0.3105 |

### Top-3 Promoted (3 Seeds: `42, 43, 44`)

| Config (`lr / mlr / mwd / awd`) | Full 10M Val Loss | Overall Bench | `icl` | `know` | `lang` | `math` | `math-l` | `reas` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`0.003 / 0.5 / 0.10 / 0.02`** | **3.86348 ± 0.00060** | 0.3096 ± 0.0015 | 0.361 | 0.309 | **0.414** | 0.318 | 0.254 | 0.202 |
| `0.003 / 0.5 / 0.10 / 0.05` | 3.86406 ± 0.00074 | **0.3101 ± 0.0019** | 0.366 | 0.310 | 0.391 | **0.328** | **0.264** | 0.202 |
| `0.003 / 0.5 / 0.10 / 0.10` | 3.86497 ± 0.00069 | 0.3095 ± 0.0046 | **0.371** | **0.317** | 0.401 | 0.318 | 0.247 | 0.202 |

**Decision:** Selected **`lr = 0.003`, `muon.lr_multiplier = 0.5` (effective Muon
LR `1.5e-3`), `muon.weight_decay = 0.10`, `adamw.weight_decay = 0.02`
(`embedding_weight_decay = true`)**. Raising AdamW/embedding LR while halving
Muon's relative LR accounted for the largest gain (-0.010 nats over the initial
`2.4e-3 / 1.0` baseline).

---

## 5. Schedule and Weight EMA Comparison (100M Tokens)

Evaluated at 6,104 steps (100.0M tokens, `warmup_steps = 219`,
`ema_decay = 0.9995`, `ema_warmup = true`, `save_pre_decay = true`, seed `42`):

| Schedule | Pre-Decay Raw Val (Bench) | Pre-Decay EMA Val (Bench) | Final Raw Val (Bench) | Final EMA Val (Bench) |
|---|---|---|---|---|
| **WSD 40% decay** | 3.67339 (0.3282) | **3.52946** (0.3302) | **3.38766** (0.3419) | 3.39128 (0.3404) |
| **WSD 15% decay** | 3.59916 (0.3343) | **3.45318** (0.3385) | 3.41770 (**0.3449**) | **3.41123** (0.3414) |
| **Constant LR + EMA** | — | — | 3.56780 (0.3338) | **3.42078** (0.3387) |

**Findings:**
- EMA (`0.9995`) yielded **-0.144 to -0.147 nats** on un-annealed weights
  (Constant LR and pre-decay WSD), recovering 81.6% of WSD-40%'s annealing gain
  with zero LR decay.
- **WSD 40% decay** achieved the lowest final validation loss (`3.38766`,
  beating WSD-15% EMA by -0.0236 nats and Constant+EMA by -0.0331 nats).
- At 100M tokens (6,104 steps), a fixed `0.9995` EMA (~2,000-step window)
  slightly lagged the fully annealed WSD-40% raw weights (+0.0036 nats), while
  improving final `icl` (0.350 vs. 0.338) and `reasoning` (0.200 vs. 0.197).

---

## 6. Final 1B-Token Run (`35m_icl_maxxing_1b`)

Trained using `src/pavullmo/configs/35m_icl_maxxing_config.toml` for 61,035
steps (999,997,440 tokens, `batch_size = 8`, `grad_accum_steps = 2`,
`seed = 42`, `checkpoint.interval_steps = 1000`, `checkpoint.save_pre_decay = true`,
`ema_decay = 0.9995`). An unexpected local machine shutdown interrupted the run
at step 25,212; training resumed cleanly from `step_00025000.pt` with exact
optimizer, scheduler, EMA, data-cursor, and RNG restoration and completed at
step 61,035.

| Checkpoint / Weights | Full 10M Val Loss (PPL) | Untouched 10M Test Loss (PPL) | Overall Bench | `icl` | `know` | `lang` | `math` | `math-l` | `reas` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pre-decay Raw (`step_00036708`, 601M tok) | 3.35685 (28.70) | — | 34.79% | 34.3% | 39.1% | 56.0% | 35.9% | 26.7% | 16.7% |
| Pre-decay EMA (`step_00036708`, 601M tok) | **3.17814** (24.00) | — | **37.84%** | **35.8%** | **48.6%** | 59.2% | 39.8% | 26.3% | 17.3% |
| Final Raw (`step_00061035`, 1.00B tok) | 3.02633 (20.62) | 3.03298 (20.76) | 39.69% | 34.2% | 48.2% | 69.2% | 40.6% | 25.7% | 20.3% |
| **Final EMA** (`step_00061035`, 1.00B tok) | **3.02563** (**20.61**) | **3.03219** (**20.74**) | **39.96%** | 33.8% | 47.7% | **69.2%** | **41.1%** | **27.4%** | **20.6%** |

At 1B scale (where `ema_decay = 0.9995` spans only ~3.3% of the run), EMA
improved the pre-decay checkpoint by **-0.1787 nats** (+3.05% benchmark
accuracy) and **outperformed raw weights on the final annealed checkpoint across
every aggregate metric** (validation loss, untouched test loss, and overall
30-benchmark bidirectional accuracy).

### Reproduction and Inference

```bash
# Train 1B run locally (8 GB GPU)
uv run python src/pavullmo/pretrain.py --config 35m_icl_maxxing_config.toml \
  --override 'name=35m_icl_maxxing_1b;runtime.dataset_dir=artifacts/datasets/fineweb2-hq-filter;train_dataset.path=train_full;validation_dataset.path=validation;training.batch_size=8;training.grad_accum_steps=2;training.max_steps=61035;evaluation.interval_steps=1000;evaluation.steps=14;checkpoint.interval_steps=1000'

# Generate from EMA weights (omit --ema for raw weights)
uv run python src/pavullmo/generate.py artifacts/runs/35m_icl_maxxing_1b \
  src/tokenizer/production_16k/tokenizer.model --ema
```

Full evaluation outputs are saved locally under
`artifacts/results/35m_icl_maxxing_1b_eval.json` and
`artifacts/results/icl_maxxing_study_results.json`; the completed run is logged
in both `artifacts/results/pretrain_runs.csv` and
`src/pavullmo/pretrain_runs.csv`.
