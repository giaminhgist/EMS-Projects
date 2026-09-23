# roman_features/ — isolated ablation: saccade direction frequencies & ambient/focal K-index

Isolated experiment system for two new hand-crafted eye-movement feature
groups on the EMS dataset, evaluated as a factorial ablation over the
existing EXP-PROP-001 model (learned stimulus-conditioned normative modeling,
45 base features), the official 4-fold protocol, and the locked 10-seed
suite (same seeds as the original experiments).

**Isolation contract.** Everything this folder produces (code, caches,
configs, logs, checkpoints, predictions, results) lives inside
`roman_features/`. The folder only READS `src/`, `processed_dataset/` and
`original_dataset/`. No file outside `roman_features/` is modified or
written; deleting this entire folder restores the original pipeline
unchanged. Do not commit or push changes from this folder.

## 1. Scientific motivation

**Direction frequencies (Feature A).** Schizophrenia eye-tracking research
reports altered scanpath structure (e.g. restricted exploration, altered
saccade metrics). The 45 base features summarize direction only globally
(`geo_angle_var`, `geo_dx_mean`, `geo_dy_mean`) and contain no stimulus-level
distribution of saccade directions. The relative histogram of inferred
saccade directions per stimulus tests whether *where the eyes move next*
(directional distribution) adds predictive signal beyond *where fixations
land* (already captured by the spatial grid features).

**Ambient/focal K-index (Feature B).** Velichkovsky et al. (2005) and
Krejtz et al. (2016) describe viewing as a mix of ambient (exploratory:
short fixations, long saccades) and focal (inspection: long fixations, short
saccades) processing. The K coefficient standardizes this trade-off. The 45
base features contain duration statistics (`tem_*`) and amplitude statistics
(`geo_sacc_amp_*`) but never their *paired* trade-off within a fixation-
saccade transition. A single per-stimulus scalar `att_k_mean` tests whether
this trade-off carries SZ/HC signal.

## 2. Saccade inference and conventions

The EMS release contains fixations only (no explicit saccade events). Every
transition between consecutive cleaned fixation centers — ordered by
`FIX_INDEX`, after the original cleaning rules of
`src/preprocess.py::clean_fixations` (off-screen, duration (40, 2000] ms,
per-stimulus pupil ±4 SD) — is treated as an **inferred saccade**:

```
dx = np.diff(x);  dy = np.diff(y);  amplitude = np.hypot(dx, dy)
```

Transitions with amplitude ≤ 1e-6 are excluded: their direction is
undefined (position jitter at the same point).

**Angle convention** (Cartesian screen coordinates, y grows downward):

```
angle = (degrees(arctan2(-dy, dx)) + 360) % 360
0 = right, 90 = up, 180 = left, 270 = down; angles in [0, 360).
```

Bin edges are `np.arange(0, 361, bin_width)`. `np.histogram`'s half-open
convention applies: an angle exactly on a bin edge falls into the HIGHER
bin (e.g. 180° → bin [180, 200)).

**Counts vs relative frequency.** Histogram bins hold *relative frequencies*
`count_b / n_valid`, not raw counts: total fixation count is already a model
input (`spa_fix_count`), and raw counts would scale with viewing length.
Frequencies sum to 1 exactly when at least one valid transition exists.

## 3. K-index definition and reference statistics

For each valid transition pair i (fixation i with duration d_i and the
subsequent inferred saccade i→i+1 with amplitude a_{i+1}; i = 0..n−2):

```
K_i = z(d_i) − z(a_{i+1}),   z(x) = (x − mean) / max(std, 1e-6)
```

K_i > 0: relatively long fixation followed by a relatively short saccade
(**focal-like**). K_i < 0: relatively short fixation followed by a
relatively long saccade (**ambient-like**). The amplitude is the saccade
term (prose definition of the task; the explicit formula writes it as a_i
for brevity). The last fixation of a stimulus never contributes a pair
(no following saccade).

**Reference-statistics scope (locked).** `mean`/`std` of duration and of
amplitude are computed **per subject** over ALL valid fixation→saccade pairs
of the subject's full viewing session (all stimuli), using no diagnosis
labels. Stimuli are NOT standardized independently — that would force every
stimulus-level mean K toward zero and destroy the ambient/focal contrast.
Zero standard deviation is clamped to EPS = 1e-6 (a constant series gives
z = 0 exactly, so no division blow-up can reach the model).

**Model feature.** `att_k_mean` = arithmetic mean of K_i within the stimulus
— a single scalar, so the ablation cleanly tests the supervisor's suggested
K-index. Diagnostics (`att_k_std`, `att_k_frac_pos`, `att_k_slope`) are
computed and cached but are NOT part of any primary model input.

## 4. Degenerate-sequence and missingness policy

| Situation | Direction frequencies | att_k_mean | Effect in pipeline |
|---|---|---|---|
| (subject, stimulus) absent or < 2 clean fixations (missing stimulus) | NaN | NaN | Row is all-NaN → masked exactly as in the original pipeline (invariant asserted at cache build) |
| ≥ 2 fixations, 0 valid transitions | all-zero vector (finite, documented) | NaN | Zero vector = "no directional information"; att_k_mean filled with train-fitted per-feature mean → zero deviation |
| ≥ 1 valid pair | relative frequencies (sum = 1) | mean of K_i | Model inputs, finite |

No NaN/inf from the frequency features can reach the model; `att_k_mean`
NaN cells follow the existing missing-feature policy of
`src/data/tabular.py` (fill with the train-fitted per-feature mean → zero
deviation). A subject with no valid pair in the whole session has undefined
reference stats → `att_k_mean` NaN on all their stimuli.

## 5. Feature sets

| name | contents | dim |
|---|---|---|
| `base45` | existing 45 features, copied verbatim from `processed_dataset/` | 45 |
| `base45_freq20` | base45 + 18 direction-frequency features (20° bins) | 63 |
| `base45_k` | base45 + `att_k_mean` | 46 |
| `base45_freq20_k` | base45 + 18 direction features + `att_k_mean` | 64 |
| `base45_freq10` | base45 + 36 direction-frequency features (10° bins) — sensitivity | 81 |
| `base45_freq10_k` | base45 + 36 direction features + `att_k_mean` — optional, supported | 82 |

Direction feature names: `geo_dir_freq_{lo:03d}_{hi:03d}`, e.g.
`geo_dir_freq_000_020` … `geo_dir_freq_340_360` (20°) and
`geo_dir_freq_000_010` … `geo_dir_freq_350_360` (10°). The exact ordered
feature-name list is saved with every run (`feature_names.txt`).

## 6. Ablation matrix and hypotheses

| ID | feature set | dim | tests | hypothesis |
|---|---|---|---|---|
| A0 | base45 | 45 | reproduced 45-feature reference | none — must reproduce `outputs/experiments/EXP-PROP-001/full45/` (verified EXACT for all 10 seeds) |
| A1 | base45_freq20 | 63 | independent contribution of the direction distribution | direction frequencies carry SZ/HC signal beyond base45 → AUC ↑ vs A0 |
| A2 | base45_k | 46 | independent contribution of the K-index | the ambient/focal trade-off carries signal beyond base45 → AUC ↑ vs A0 |
| A3 | base45_freq20_k | 64 | combined contribution and possible interaction | complementary: A3−A0 > max(A1−A0, A2−A0) |
| A4 | base45_freq10 | 81 | sensitivity to direction-bin resolution | if direction matters, resolution matters; A4−A1 ≈ 0 supports 20° sufficiency |

Required comparisons: A1−A0, A2−A0, A3−A1, A3−A2, A3−A0 (primary factorial),
A4−A1 (sensitivity). All runs share identical hyperparameters (epochs 150,
patience 30, AdamW lr 1e-3, wd 1e-4, batch 16, dropout 0.3), the same folds,
the same early-stopping rule, the same fixed threshold 0.5, and the same
locked 10 seeds (1234, 100, 0, 2024, 42, 3, 123, 13, 64, 2026).
The 20° bins are the pre-specified primary configuration; 10° is sensitivity
only (never selected by looking at validation folds). Feature normalization
is fitted on train-fold subjects only; the HC latent bank uses train-fold HC
subjects only — exactly as in the original method.

## 7. Reproduction commands (exact)

```bash
# from the repo root, with the project venv
# 1. unit + isolation tests (dependency-free runner, leaves the venv untouched)
.venv/bin/python roman_features/tests/run_tests.py

# 2. build the extended feature cache (reads raw EMS + processed_dataset)
.venv/bin/python roman_features/code/build_cache.py

# 3. full 10-seed matrix (locked src/registry.SEEDS, same seeds as the
#    original suite; resume-safe, skips seeds whose folds_summary.json exists)
.venv/bin/python roman_features/code/run_seed_sweep.py --workers 7

# single (ablation, seed) run, e.g. seed 2026:
.venv/bin/python roman_features/code/run_ablation.py \
    --ablation-id A0_base45 --feature-set base45 --seed 2026 --fold all

# optional: 10° + K (supported, not part of the required matrix)
.venv/bin/python roman_features/code/run_ablation.py \
    --ablation-id A5_freq10_k --feature-set base45_freq10_k --seed 2026 --fold all

# 4. consolidated summary (all 10 seeds by default; --seeds for a subset)
.venv/bin/python roman_features/code/summarize.py
```

Smoke runs (2 epochs, one fold, results under `results/_smoke/`, clearly
non-final):

```bash
.venv/bin/python roman_features/code/run_ablation.py --ablation-id A0_base45 \
    --feature-set base45 --seed 2026 --fold Set_0 --smoke --epochs 2 --patience 5
```

The runner sets `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`,
reproducing the environment of the original experiment runner (needed for
direct comparability with the existing seed-2026 run).

## 8. Output structure

```
roman_features/
├── SUMMARY.md                  # hand-written experiment summary (results,
│                               # deltas, reproduction status, limitations)
├── results/
│   ├── summary.csv             # one row per run: mean ± SD metrics, deltas vs A0, missingness
│   ├── deltas.csv              # all required pairwise deltas
│   ├── summary.md              # consolidated report (primary vs sensitivity separated)
├── A0_base45/seed2026/
│   ├── resolved_config.json   # full resolved configuration + hypothesis
│   ├── feature_names.txt      # exact ordered feature-name list
│   ├── run.log
│   ├── folds_summary.json     # per-fold metrics + mean ± SD
│   └── foldSet_0/             # per fold: config.json, run_info.json,
│       ...                    # metrics.jsonl, val_probs.jsonl, best.pt,
│                              # last.pt, predictions.csv, summary.json
└── _smoke/                    # smoke runs (clearly non-final)
```

## 9. Deleting runs or the whole folder

```bash
rm -r roman_features/results/A4_freq10        # delete one run
rm -r roman_features                          # delete everything (original
                                              # pipeline unaffected by design)
```

After deleting `results/` subfolders the original pipeline keeps working:
nothing in `src/`, `outputs/` or `processed_dataset/` was modified. The
cache can be rebuilt anytime with `build_cache.py`.

## 10. Known limitations

- **Sparse direction histograms.** Stimuli with few fixations give
  high-variance direction histograms (mean 12.7 valid transitions per
  stimulus, min 1); 20° bins with few transitions are noisy inputs.
- **Inferred saccades.** EMS has no saccade events; consecutive-fixation
  transitions approximate saccades and skip glissades/corrective saccades
  within a fixation cluster; zero-amplitude consecutive fixations are
  excluded by design.
- **K-index reference coupling.** Subject-level reference statistics couple
  stimuli within a subject (by design) — a subject's K value on one stimulus
  depends on their whole session.
- **Development comparison.** Ten seeds, same 4 folds as the main study:
  fold-level means with population SD are development numbers, not an
  independent generalization estimate.
- **Naming mismatch with the task text.** The task text references an
  "EXP 001 model configuration" / "mlp_norm01"; neither exists in the repo.
  The authoritative main experiment is EXP-PROP-001 (`mlp_deepset`),
  which is what all runs here use (recorded in `tasks/plan.md`).
