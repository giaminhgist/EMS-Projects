# EXP-PROP-003 — pooling ablation (mlp_attn)

**Attention pooling instead of deepset.** Identical to EXP-PROP-001 except
the pooling: a learned attention scorer weights the per-stimulus deviations
before summing them into the subject embedding.

Code: same `src/proposal/model.py` with `pool="attention"`; config in
`src/registry.py::PROPOSAL_EXPERIMENTS`. Artifacts:
`outputs/experiments/EXP-PROP-003/full45/seed<seed>/`.

## Architecture delta vs 001

- Pooling: per-stimulus attention scores
  `a_s = softmax(scorer(d_s) + (mask_s − 1)·1e9)` with scorer
  64 → 32 → ReLU → 1; subject embedding = `Σ a_s·mask_s·d_s` (64 dims).
  Masked (missing) stimuli get effectively −∞ logits, so they never
  receive weight.
- Head input 64 (no max path).
- Everything else identical: encoder 45→128→128, per-epoch HC latent bank,
  comparator mlp on `[z, μ, z−μ, z·μ]` → 64, HP epochs 150 / patience 30 /
  AdamW 1e-3 / wd 1e-4 / batch 16 / dropout 0.3 / threshold 0.5.

## What it tests

Whether a learned, data-dependent weighting of stimuli (letting the model
emphasize the most informative stimuli per subject) beats the fixed
mean‖max combination of the deepset.

## Results (10 seeds, official 4-fold)

- AUC 0.9299 ± 0.0077 | ACC 0.8219 | SEN 0.8028 | SPEC 0.8399 |
  F1 0.8080 | pooled OOF AUC 0.8790.
- Per-seed fold-mean AUCs: s0=0.922, s3=0.931, s13=0.921, s42=0.932,
  s64=0.934, s100=0.922, s123=0.930, s1234=0.946, s2024=0.936, s2026=0.926.

## Reading

- ΔAUC vs main (001): -1.04 pp on the mean-of-means (paired per-seed
  -0.0104) — attention pooling is clearly below deepset.
- Learned stimulus weighting does not pay off here: with only ~120 train
  subjects the scorer adds parameters that deepset's fixed mean‖max
  combination already covers, and the gap is larger than for mean pooling.

## Limitations

Development numbers over the same 10 seeds/folds; fold scores are not
independent replicates.
