# EXP-PROP-005 — pooling ablation (mlp_max)

**Masked max pooling instead of deepset.** Identical to EXP-PROP-001
except the pooling: only the per-dimension maximum of the masked
per-stimulus deviations forms the subject embedding (no mean path).

Code: same `src/proposal/model.py` with `pool="max"`; config in
`src/registry.py::PROPOSAL_EXPERIMENTS`. Artifacts:
`outputs/experiments/EXP-PROP-005/full45/seed<seed>/`.

## Architecture delta vs 001

- Pooling: `d_max = max over valid stimuli of (mask·d + (1−mask)·(−1e9))`
  → 64-dim subject embedding; missing stimuli are excluded by the −1e9
  offset (they can never be the max).
- Head input 64 (no mean path).
- Everything else identical: encoder 45→128→128, per-epoch HC latent bank,
  comparator mlp on `[z, μ, z−μ, z·μ]` → 64, HP epochs 150 / patience 30 /
  AdamW 1e-3 / wd 1e-4 / batch 16 / dropout 0.3 / threshold 0.5.

## What it tests

The max-only half of the deepset: whether the most extreme per-stimulus
deviation per dimension alone captures the subject-level signal (the
"worst stimulus" view of normative atypicality), and how it compares with
the mean path (002) and the combined mean‖max (001).

## Results (10 seeds, official 4-fold)

- AUC 0.9304 ± 0.0156 | ACC 0.8319 | SEN 0.8103 | SPEC 0.8534 |
  F1 0.8185 | pooled OOF AUC 0.9044.
- Per-seed fold-mean AUCs: s0=0.931, s3=0.916, s13=0.928, s42=0.928,
  s64=0.932, s100=0.948, s123=0.899, s1234=0.950, s2024=0.947, s2026=0.926.

## Reading

- ΔAUC vs main (001): -0.99 pp on the mean-of-means (paired per-seed
  -0.0099); below deepset but above attention pooling (003).
- Ranking of pooling variants on mean AUC: deepset (0.9403) > mean
  (0.9383) > max (0.9304) > attention (0.9299) — the mean‖max combination
  carries the strongest signal; max alone beats attention but loses to
  mean.

## Limitations

Development numbers over the same 10 seeds/folds; fold scores are not
independent replicates.
