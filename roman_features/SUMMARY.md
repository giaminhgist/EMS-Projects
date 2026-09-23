# SUMMARY — roman_features/ feature ablation (10 seeds)

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
- CPU-only (torch 2.11.0+cu128), `OMP/MKL/OPENBLAS_NUM_THREADS=1` to
  reproduce the original suite environment.

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

### Pairwise deltas (paired per seed, mean ± SD over 10 seeds)

| comparison | ΔAUC | ΔACC | ΔbalACC | ΔSEN | ΔSPEC | ΔPRE | ΔF1 |
|---|---|---|---|---|---|---|---|
| A1_freq20 - A0_base45 | -0.0133 ± 0.0242 | -0.0288 | -0.0219 | -0.0076 | -0.0362 | -0.0249 | -0.0214 |
| A2_k - A0_base45 | -0.0099 ± 0.0165 | -0.0288 | -0.0261 | -0.0574 | +0.0052 | -0.0044 | -0.0447 |
| A3_freq20_k - A1_freq20 | -0.0083 ± 0.0252 | -0.0219 | -0.0315 | -0.0355 | -0.0275 | -0.0309 | -0.0329 |
| A3_freq20_k - A2_k | -0.0117 ± 0.0228 | -0.0219 | -0.0273 | +0.0143 | -0.0689 | -0.0515 | -0.0097 |
| A3_freq20_k - A0_base45 | -0.0216 ± 0.0200 | -0.0506 | -0.0534 | -0.0431 | -0.0637 | -0.0559 | -0.0544 |
| A4_freq10 - A1_freq20 | -0.0100 ± 0.0164 | -0.0019 | -0.0086 | -0.0257 | +0.0084 | -0.0036 | -0.0105 |
