# EMS-Projects — Documentation

Methodology and results for every experiment of the current 10-seed suite
plus the full reference of the 45 hand-crafted features.

**Authoritative sources.** Numbers in this folder are copied from
`EXPERIMENTS_RESULTS.md` (root) and `outputs/experiments/reports/*.md`; the
per-run artifacts live under `outputs/experiments/`. The executable code in
`src/` is the ground truth whenever a formula here looks ambiguous.

## Layout

| file | content |
|---|---|
| [`features45.md`](features45.md) | the 45 hand-crafted eye-movement features: definition, formula, units, missingness rules, behavioral/clinical interpretation |
| [`experiments/baselines.md`](experiments/baselines.md) | the 10 classical-ML / FNN baselines |
| [`experiments/EXP-PROP-001.md`](experiments/EXP-PROP-001.md) | main proposal — learned normative deviation + deepset pooling |
| [`experiments/EXP-PROP-002.md`](experiments/EXP-PROP-002.md) | pooling ablation: mean |
| [`experiments/EXP-PROP-003.md`](experiments/EXP-PROP-003.md) | pooling ablation: attention |
| [`experiments/EXP-PROP-004.md`](experiments/EXP-PROP-004.md) | deviation ablation: fixed hard z-deviation |
| [`experiments/EXP-PROP-005.md`](experiments/EXP-PROP-005.md) | pooling ablation: masked max |

## Dataset and protocol (shared by all experiments)

- **EMS** (Song et al., IEEE TNNLS 2024): public free-viewing eye-tracking
  dataset, 160 labelled subjects (80 HC with ids < 200, 80 SZ with ids ≥ 200),
  100 natural/synthetic/manipulated images per subject, 48 official test
  subjects with withheld labels.
- **Cleaning** (`src/preprocess.py::clean_fixations`): off-screen fixations
  dropped, duration in (40, 2000] ms, per-stimulus pupil outliers (±4 SD of
  the stimulus median) dropped; stimuli with < 2 clean fixations are
  missing.
- **Features**: 45 hand-crafted features per (subject, stimulus), see
  [`features45.md`](features45.md).
- **Protocol P1 (official 4-fold)**: `Train_Valid.xlsx` assigns the 160
  subjects to Set_0..Set_3 (40 each); per fold, train on the other 120 and
  validate on the fold. Same folds for every experiment. The official test
  labels are withheld, so all reported metrics are validation metrics.
- **Seeds**: 10 locked seeds `[1234, 100, 0, 2024, 42, 3, 123, 13, 64, 2026]`
  for every experiment; the trainer reseeds at the start of each fold.
- **Metric convention**: SZ = positive class; AUC threshold-free, ACC/SEN/
  SPEC/F1 at fixed threshold 0.5. Best checkpoint = highest outer-validation
  AUC (earliest epoch wins ties). Per seed = mean of the 4 fold metrics;
  reported as mean ± sample SD (ddof=1) over the 10 seed means. These are
  development numbers — not an unbiased generalization estimate.
- **Environment**: CPU-only, torch 2.11.0+cu128, sklearn 1.9.0,
  OMP/MKL/OPENBLAS threads = 1.

## Experiment map (10 seeds, official 4-fold)

| Experiment | What it tests | AUC (mean ± SD) |
|---|---|---|
| EXP-PROP-001 (mlp_deepset) | **main proposal**: learned latent deviation + mlp comparator + deepset pooling | 0.9403 ± 0.0156 |
| EXP-PROP-002 (mlp_mean) | pooling ablation: mean vs deepset | 0.9383 ± 0.0064 |
| EXP-PROP-003 (mlp_attn) | pooling ablation: attention vs deepset | 0.9299 ± 0.0077 |
| EXP-PROP-004 (z_mean) | deviation ablation: fixed hard z-deviation (no encoder/comparator/bank) | 0.9073 ± 0.0040 |
| EXP-PROP-005 (mlp_max) | pooling ablation: masked max vs deepset | 0.9304 ± 0.0156 |
| fnn | best baseline: FNN on agg representation | 0.9176 ± 0.0082 |
| svm_rbf | best classical baseline | 0.8793 ± 0.0000 |
| fnn_cat | FNN on per-category representation | 0.9049 ± 0.0059 |
| lr | logistic regression | 0.8711 ± 0.0000 |
| rf | random forest | 0.8581 ± 0.0038 |
| qda | quadratic discriminant analysis | 0.8229 ± 0.0000 |
| svm_linear | linear SVM | 0.8189 ± 0.0001 |
| knn | k-nearest neighbours | 0.8120 ± 0.0000 |
| gnb | Gaussian naive Bayes | 0.7913 ± 0.0000 |
| lr_l1 | L1 logistic regression on concat representation | 0.7360 ± 0.0035 |

See [`EXPERIMENTS_RESULTS.md`](../EXPERIMENTS_RESULTS.md) for the full
comparison tables (main vs baselines, ablations, MSNet reference).

## Related isolated work

`roman_features/` is a self-contained ablation system that adds inferred-
saccade direction-frequency features and an ambient/focal K-index to base45
(see `roman_features/SUMMARY.md`). It is isolated by design: it does not
modify this documentation, `src/`, `outputs/` or `processed_dataset/`.
