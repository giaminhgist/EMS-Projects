# EXP-PROP-002 — pooling ablation (mlp_mean)

**Mean pooling instead of deepset.** Identical to EXP-PROP-001 (learned
deviation, mlp comparator, same encoder/bank/head hyperparameters) except
the stimulus-level pooling: mean of the per-stimulus deviations over valid
stimuli only.

Code: same `src/proposal/model.py` with `pool="mean"`; config in
`src/registry.py::PROPOSAL_EXPERIMENTS`. Artifacts:
`outputs/experiments/EXP-PROP-002/full45/seed<seed>/`.

## Architecture delta vs 001

- Pooling: `d_mean = Σ(mask·d) / count(valid stimuli)` → 64-dim subject
  embedding (no max path).
- Head input shrinks from 128 to 64 accordingly.
- Everything else identical: encoder 45→128→128, per-epoch HC latent bank,
  comparator mlp on `[z, μ, z−μ, z·μ]` → 64, HP epochs 150 / patience 30 /
  AdamW 1e-3 / wd 1e-4 / batch 16 / dropout 0.3 / threshold 0.5.

## What it tests

Whether the max path of the deepset (mean‖max) pooling carries signal the
mean alone misses: deepset adds the largest per-stimulus deviation per
dimension, which captures extreme/atypical stimuli rather than only the
average atypicality.

## Results (10 seeds, official 4-fold)

- AUC 0.9383 ± 0.0064 | ACC 0.8375 | SEN 0.8224 | SPEC 0.8480 |
  F1 0.8271 | pooled OOF AUC 0.8957.
- Per-seed fold-mean AUCs: s0=0.944, s3=0.936, s13=0.932, s42=0.945,
  s64=0.942, s100=0.931, s123=0.934, s1234=0.949, s2024=0.939, s2026=0.932.

## Reading

- ΔAUC vs main (001): -0.20 pp on the mean-of-means (paired per-seed
  -0.0020) — mean pooling is close to deepset, the smallest gap among the
  pooling ablations.
- Mean pooling keeps most of the normative signal: the average deviation
  across stimuli already summarizes the subject well.

## Limitations

Development numbers over the same 10 seeds/folds; fold scores are not
independent replicates.
