# EXP-PROP-004 — deviation ablation (z_mean)

**Fixed hard z-deviation, no learned encoder/comparator/bank.** The
deviation mechanism of the main proposal is replaced by plain
per-stimulus z-scores against HC normative statistics computed directly in
raw feature space; the rest (mean pooling + head) is shared with the
proposal.

Code: `src/proposal/model.py` (`deviation="z"`, `pool="mean"`, no
encoder/comparator/bank — `refresh_bank` is a no-op), dataset path in
`src/data/tabular.py::hc_normative_stats` / `apply_deviation`. Config in
`src/registry.py::PROPOSAL_EXPERIMENTS`. Artifacts:
`outputs/experiments/EXP-PROP-004/full45/seed<seed>/`.

## Architecture delta vs 001

- **Deviation (hard, fixed)**: per stimulus s and feature f,
  `z = (x_sf − μ_sf) / σ_sf` where μ, σ are the per-stimulus mean/std over
  the train-fold **HC subjects only** (stimuli without HC coverage get
  μ = 0, σ = EPS = 1e-6 and are masked anyway). Missing feature cells are
  filled with the HC stimulus norm μ, so they contribute zero deviation.
- **No trainable deviation machinery**: no encoder, no comparator, no
  latent bank — the model has no parameters before the head.
- Pooling: mean over valid stimuli → 45-dim subject embedding; head
  45 → 64 → BN → ReLU → Dropout(0.3) → 1.
- HP identical to 001 (epochs 150, patience 30, AdamW 1e-3, wd 1e-4,
  batch 16, dropout 0.3, threshold 0.5).

## What it tests

How much of the main proposal's value comes from the *learned* deviation
(encoder + comparator + latent bank) versus the classic fixed normative
z-deviation in raw feature space. Both share the same pooling and head, so
the comparison isolates the deviation mechanism.

## Results (10 seeds, official 4-fold)

- AUC 0.9073 ± 0.0040 | ACC 0.8094 | SEN 0.8165 | SPEC 0.8058 |
  F1 0.8048 | pooled OOF AUC 0.8966.
- Per-seed fold-mean AUCs: s0=0.907, s3=0.906, s13=0.907, s42=0.898,
  s64=0.908, s100=0.906, s123=0.913, s1234=0.908, s2024=0.912, s2026=0.908.

## Reading

- ΔAUC vs main (001): -3.30 pp on the mean-of-means (paired per-seed
  -0.0330) — the largest gap in the proposal suite.
- The learned pipeline (latent deviation + mlp comparator + bank) adds
  substantial value over raw feature-space z-scores; the hard z-deviation
  alone is below fnn (0.9176) and barely above svm_rbf (0.8793).
- The gap is consistent across seeds (small SD, 0.0040), which is expected:
  the deviation path is fixed and only the head is trained.

## Limitations

Development numbers over the same 10 seeds/folds; the HC normative
statistics use only the training-fold HC subjects, exactly as in the main
method.
