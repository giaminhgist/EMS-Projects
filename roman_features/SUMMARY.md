# SUMMARY — roman_features/ isolated feature ablation (10 seeds)

Generated 2026-09-23 from `roman_features/results/` (locked 10 seeds,
official 4-fold). Companion to `README.md` (definitions, policies, commands)
and the auto-generated `results/summary.md` / `results/summary.csv` /
`results/per_seed_metrics.csv` / `results/deltas.csv`.

## Protocol (brief)

- Official subject-level 4-fold validation (Set_0..Set_3; 120 train / 40 val
  subjects per fold), same folds as the main suite.
- Model: **EXP-PROP-001 configuration** — learned stimulus-conditioned
  normative deviation, mlp comparator, deepset (mean‖max) pooling.
  Identical hyperparameters across all runs: epochs 150, patience 30,
  AdamW lr 1e-3, wd 1e-4, batch 16, dropout 0.3; fixed threshold 0.5, SZ = 1.
- **10 locked seeds** `[1234, 100, 0, 2024, 42, 3, 123, 13, 64, 2026]` — the
  same seeds as the original suite.
- Best checkpoint = highest outer-validation AUC (earliest epoch wins ties);
  metrics reported at that checkpoint. Feature normalization fitted on
  train-fold subjects only; the HC latent bank uses train-fold HC subjects
  only — exactly as in the original method.
- Reporting convention (identical to the original suite): per seed = mean of
  the 4 fold metrics; reported as mean ± sample SD (ddof=1) over the 10 seed
  means. Deltas are **paired per seed** (all runs share the same seeds and
  folds), reported as mean ± SD over the per-seed deltas.
- CPU-only (torch 2.11.0+cu128), `OMP/MKL/OPENBLAS_NUM_THREADS=1` to
  reproduce the original suite environment.
- Development comparison with selection optimism — not an unbiased
  generalization estimate.

## Question and hypotheses

**Question.** Do (A) per-stimulus relative frequencies of inferred-saccade
directions and (B) the ambient/focal K-index add predictive value for SZ/HC
recognition over the existing 45 hand-crafted features?

- **H-A**: direction frequencies carry SZ/HC signal beyond base45.
- **H-B**: the subject-standardized ambient/focal trade-off (`att_k_mean`)
  carries signal beyond base45.
- **H-A4**: bin resolution matters if direction matters (10° vs 20°).

## Feature sets

| name | contents | dim |
|---|---|---|
| `base45` | existing 45 features (verbatim from `processed_dataset/`) | 45 |
| `base45_freq20` | + 18 direction-frequency features, 20° bins | 63 |
| `base45_k` | + `att_k_mean` (ambient/focal K-index) | 46 |
| `base45_freq20_k` | + 18 direction features + `att_k_mean` | 64 |
| `base45_freq10` | + 36 direction-frequency features, 10° bins (sensitivity) | 81 |

Saccades are inferred from consecutive cleaned fixation centers ordered by
`FIX_INDEX`; zero-amplitude (≤ 1e-6 px) transitions are excluded; bins hold
**relative** frequencies `count_b / n_valid`. K-index:
`K_i = z(d_i) - z(a_{i+1})` with z standardized by **subject-level** mean/std
over all valid pairs of the full session (std clamped to 1e-6); `att_k_mean`
is the per-stimulus mean of K_i. Full formulas and degenerate-sequence
policies: `README.md`.

## Ablation matrix

| ID | feature set | dim | tests |
|---|---|---|---|
| A0 | base45 | 45 | reproduced 45-feature reference |
| A1 | base45_freq20 | 63 | independent contribution of direction distribution |
| A2 | base45_k | 46 | independent contribution of K-index |
| A3 | base45_freq20_k | 64 | combined contribution and possible interaction |
| A4 | base45_freq10 | 81 | sensitivity to direction-bin resolution (not part of the primary 2×2) |

## Results (mean ± SD over 10 seed means)

| run | dim | AUC | ACC | balACC | SEN | SPEC | PRE | F1 |
|---|---|---|---|---|---|---|---|---|
| A0_base45 | 45 | **0.9403 ± 0.0156** | **0.8494** | **0.8473** | **0.8229** | **0.8717** | **0.8711** | **0.8380** |
| A1_freq20 | 63 | 0.9270 ± 0.0167 | 0.8206 | 0.8254 | 0.8153 | 0.8355 | 0.8461 | 0.8166 |
| A2_k | 46 | 0.9305 ± 0.0098 | 0.8206 | 0.8212 | 0.7656 | 0.8769 | 0.8666 | 0.7933 |
| A3_freq20_k | 64 | 0.9187 ± 0.0151 | 0.7987 | 0.7939 | 0.7799 | 0.8080 | 0.8152 | 0.7837 |
| A4_freq10 | 81 | 0.9171 ± 0.0106 | 0.8187 | 0.8168 | 0.7897 | 0.8439 | 0.8426 | 0.8061 |

The A0 row reproduces the published main-proposal number exactly
(AUC 0.9403 ± 0.0156, EXPERIMENTS_RESULTS.md).

### Pairwise deltas (paired per seed, mean ± SD over 10 seeds)

| comparison | ΔAUC | ΔACC | ΔbalACC | ΔSEN | ΔSPEC | ΔPRE | ΔF1 |
|---|---|---|---|---|---|---|---|
| A1_freq20 - A0_base45 | -0.0133 ± 0.0242 | -0.0288 | -0.0219 | -0.0076 | -0.0362 | -0.0249 | -0.0214 |
| A2_k - A0_base45 | -0.0099 ± 0.0165 | -0.0288 | -0.0261 | -0.0574 | +0.0052 | -0.0044 | -0.0447 |
| A3_freq20_k - A1_freq20 | -0.0083 ± 0.0252 | -0.0219 | -0.0315 | -0.0355 | -0.0275 | -0.0309 | -0.0329 |
| A3_freq20_k - A2_k | -0.0117 ± 0.0228 | -0.0219 | -0.0273 | +0.0143 | -0.0689 | -0.0515 | -0.0097 |
| A3_freq20_k - A0_base45 | -0.0216 ± 0.0200 | -0.0506 | -0.0534 | -0.0431 | -0.0637 | -0.0559 | -0.0544 |
| A4_freq10 - A1_freq20 | -0.0100 ± 0.0164 | -0.0019 | -0.0086 | -0.0257 | +0.0084 | -0.0036 | -0.0105 |

### Per-seed fold-mean AUC (paired unit for the delta statistics)

| run | 1234 | 100 | 0 | 2024 | 42 | 3 | 123 | 13 | 64 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|
| A0_base45 | 0.9665 | 0.9617 | 0.9515 | 0.9481 | 0.9277 | 0.9385 | 0.9311 | 0.9277 | 0.9263 | 0.9240 |
| A1_freq20 | 0.9381 | 0.9253 | 0.9315 | 0.8982 | 0.9492 | 0.9073 | 0.9259 | 0.9492 | 0.9305 | 0.9151 |
| A2_k | 0.9273 | 0.9438 | 0.9293 | 0.9309 | 0.9358 | 0.9200 | 0.9491 | 0.9264 | 0.9212 | 0.9208 |
| A3_freq20_k | 0.9264 | 0.8957 | 0.9392 | 0.9233 | 0.9060 | 0.9434 | 0.9042 | 0.9213 | 0.9133 | 0.9146 |
| A4_freq10 | 0.9131 | 0.9101 | 0.9170 | 0.9189 | 0.9172 | 0.9202 | 0.9042 | 0.9388 | 0.9277 | 0.9034 |

A1 > A0 on 3/10 seeds, A2 > A0 on 2/10, A3 > A0 on 1/10, A4 > A0 on 2/10.

## A0 reproduction check

**EXACT for all 10/10 seeds.** Fold metrics, best epochs and epochs trained
are identical to the original `outputs/experiments/EXP-PROP-001/full45/`
runs for every seed. For seed 2026, all non-timing fields of `metrics.jsonl`
were verified bit-identical across all epochs and folds; validation
predictions identical. Only the wall-clock `elapsed_s` field differs.

## Missingness and valid feature values

| run | dim | rows | masked rows | cells | NaN cells | NaN % |
|---|---|---|---|---|---|---|
| A0_base45 | 45 | 16000 | 312 | 720000 | 14666 | 2.04 |
| A1_freq20 | 63 | 16000 | 312 | 1008000 | 20282 | 2.01 |
| A2_k | 46 | 16000 | 312 | 736000 | 14978 | 2.04 |
| A3_freq20_k | 64 | 16000 | 312 | 1024000 | 20594 | 2.01 |
| A4_freq10 | 81 | 16000 | 312 | 1296000 | 25898 | 2.00 |

The 312 masked rows are the original pipeline's missing stimuli; the
extended cache preserves that mask exactly (asserted at build time and in
tests). No stimulus with a valid base45 row has zero valid transitions in
this dataset (mean 12.7 valid transitions per stimulus, min 1).

## Interpretation

**Negative result, retained. Neither feature group adds predictive value
under this model and protocol.** Paired per-seed AUC deltas (10 seeds):

- **A1 - A0** (direction frequencies, 20°): -1.33 ± 2.42 pp, paired
  t(9) = -1.74, p = 0.117 — no evidence of a contribution; H-A not supported.
- **A2 - A0** (K-index): -0.99 ± 1.65 pp, t(9) = -1.89, p = 0.091 — no
  evidence of a contribution; H-B not supported. Descriptive secondary
  pattern: at the fixed 0.5 threshold the K-index shifts the operating point
  toward specificity (ΔSEN -5.74 pp with ΔSPEC +0.52 pp); not interpreted as
  a finding (secondary metric, 6 comparisons, development data).
- **A3 - A0** (both): -2.16 ± 2.00 pp, t(9) = -3.42, p = 0.0076 — small but
  consistent degradation (9/10 seeds negative); nominally significant and
  borderline at the Bonferroni-adjusted level for the 6 pre-specified
  comparisons (0.05/6 ≈ 0.0083). No positive interaction: A3 - A1 = -0.83 pp
  and A3 - A2 = -1.17 pp.
- **A4 - A1** (10° vs 20° sensitivity): -1.00 ± 1.64 pp, t(9) = -1.93,
  p = 0.086 — no evidence that finer bins help; H-A4 not supported.
  (A4 - A0 = -2.33 ± 2.07 pp, p = 0.006, but A4's pre-specified reference is
  A1, not A0.)

Effect sizes: the A3/A4 degradations (-2.2 pp) are ≈ 1.4× the between-seed
SD of A0 (1.56 pp) — small in absolute terms on a 0.94 AUC scale. The
supported conclusion: **no added predictive value; the combined set (and the
10° variant) appear mildly harmful rather than helpful**, with all caveats
of a development comparison below.

Untested explanations (hypotheses only): the encoder/learned-bank pipeline
already absorbs directional and duration-amplitude structure from the 45
base features; the new features mostly add high-variance (sparse histogram)
or redundant (K) inputs; the added dimensions increase optimization noise
within the fixed 150-epoch budget.

## Limitations

- Development comparison: same participants, same 4 folds, 10 seeds — not
  an external generalization estimate; fold scores are not independent
  replicates; the paired t-statistics treat the 10 per-seed fold-mean AUCs
  as the unit, as in the original suite's reporting.
- Multiplicity: 6 pre-specified comparisons; the p = 0.008 for A3 - A0 is
  borderline after Bonferroni adjustment and is not claimed as a standalone
  finding.
- Sparse direction histograms for stimuli with few fixations (min 1 valid
  transition).
- Inferred saccades (no saccade events in the EMS release); consecutive-
  fixation transitions approximate saccades.
- K-index reference statistics couple stimuli within a subject (by design).
- SVM-RBF sanity-check not implemented (optional in the task spec; the
  primary `mlp_deepset` ablation was not blocked).

## Artifacts and reproduction

```bash
.venv/bin/python roman_features/tests/run_tests.py          # 16/16 tests
.venv/bin/python roman_features/code/build_cache.py         # extended cache
.venv/bin/python roman_features/code/run_seed_sweep.py --workers 7   # 10-seed matrix
.venv/bin/python roman_features/code/summarize.py           # results/summary.*
```

Every run directory (`results/<id>/seed<seed>/`) contains
`resolved_config.json`, `feature_names.txt`, `run.log`, `folds_summary.json`
and per-fold `metrics.jsonl`, `val_probs.jsonl`, `best.pt`/`last.pt`,
`predictions.csv`, `summary.json`, `run_info.json`. Deleting
`roman_features/` leaves the original pipeline unchanged (nothing outside the
folder was modified; no commit/push).
