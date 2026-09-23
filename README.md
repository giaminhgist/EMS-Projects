# EMS — Learned Normative Gaze Modeling for Schizophrenia Recognition

Research codebase for **stimulus-conditioned learned normative deviations**
of eye-tracking behavior for schizophrenia recognition, on the public **EMS**
dataset (Song et al., *IEEE TNNLS* 2024 —
[paper](https://ieeexplore.ieee.org/document/10645682)).

- **Results**: [`EXPERIMENTS_RESULTS.md`](EXPERIMENTS_RESULTS.md) — main
  proposal vs baselines, pooling/hard-deviation ablations, and the comparison
  with the published MSNet reference.

## Layout

```
src/
├── common.py, features.py, preprocess.py  # data loading, 45-feature computation, cleaning
├── data/                                  # folds, subject matrices, leakage-safe stats, hashes
├── trainer/                               # generic seeded trainer: epoch logs, checkpoints, bank hook
├── proposal/                              # learned normative model + hard-deviation ablations
├── baseline/                              # classical ML + FNN baselines
├── metrics.py, seeding.py                 # shared metric implementation + single seeding source
├── registry.py                            # locked SEEDS + EXP-PROP-001..005 + baselines
├── run_experiments.py                     # experiment matrix runner (resume-safe)
├── summarize_experiments.py               # aggregation + reports + MSNet comparison
└── checks/                                # seed/correctness verification suite
```

## Quick start

```bash
uv sync                     # Python ≥ 3.12 (see pyproject.toml)

# data preparation (raw EMS must be in original_dataset/EMS/)
python src/preprocess.py

# one proposal run (4 official folds, one seed)
python src/proposal/train.py --experiment EXP-PROP-001 --ablation mlp_deepset \
    --comparator mlp --pool deepset --fold all --seed 42

# one baseline
python src/baseline/run_experiment.py --method svm_rbf --seed 42

# full 10-seed matrix (5 proposal + 10 baselines, official 4-fold)
python src/run_experiments.py --workers 8

# aggregation + reports (outputs/experiments/*.csv, EXPERIMENTS_RESULTS.md)
python src/summarize_experiments.py

# seed/correctness checks
python src/checks/check_seed.py
python src/checks/check_correctness.py
```

## Outputs

Every run writes to `outputs/experiments/{experiment}/{feature_set}/seed{seed}/fold{fold}/`
(config, run info with code/data hashes, per-epoch `metrics.jsonl`, per-epoch
val probabilities, `best.pt`/`last.pt` including normative-bank buffers,
predictions, summary).

## Data access & licensing

The EMS dataset is **not** included in this repository (usage agreement:
non-commercial research only). Download it from the
[official repo](https://github.com/YingjieSong1/EMS) into
`original_dataset/EMS/`, then regenerate `processed_dataset/` with
`python src/preprocess.py`.
