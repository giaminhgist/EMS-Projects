"""Post-hoc Otsu-threshold metrics for the Phase-3 matrix (no retraining).

Reads the stored best-checkpoint per-subject validation probabilities
(outputs/phase3/{exp}/.../predictions.csv for neural runs, val_preds.csv for
classical baselines) and recomputes ACC/SEN/SPEC/F1 at an Otsu threshold fit
per validation fold — the threshold protocol of the published MSNet reference
(Song et al., TNNLS 2024) — alongside the threshold-free AUC.

Writes:
  outputs/phase3/otsu_metrics.csv       per (experiment, seed, fold)
  outputs/phase3/otsu_seed_metrics.csv  per (experiment, seed) = mean of the 4 folds

AUC here equals fold_metrics.csv (threshold-free, same best checkpoint);
acc/sen/spec/f1 here are at the Otsu threshold, NOT at the fixed 0.5 of
fold_metrics.csv — do not mix the two tables. Otsu fits the threshold on the
same validation set it scores (threshold optimization on eval data): the
resulting ACC/SEN/SPEC/F1 are optimistic and only mirror the MSNet comparison
protocol; the fixed-0.5 main table remains the primary report.

Usage: python src/otsu_metrics.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.common import OUTPUTS  # noqa: E402
from registry import (SEEDS, PROPOSAL_EXPERIMENTS, BASELINE_METHODS,  # noqa: E402
                      FEATURE_SET, FOLD_NAMES)  # noqa: E402

BASE = OUTPUTS / "phase3"
PROPOSAL_IDS = [e["id"] for e in PROPOSAL_EXPERIMENTS]
EXPERIMENTS = PROPOSAL_IDS + list(BASELINE_METHODS)


def otsu_threshold(scores):
    """Otsu threshold on 1-D scores (maximizes between-class variance)."""
    vals = np.unique(np.asarray(scores, float))
    if len(vals) == 1:
        return float(vals[0])
    best_t, best_bc = 0.5, -1.0
    for t in vals:
        lo = scores <= t
        w0 = lo.mean()
        w1 = 1.0 - w0
        if w0 == 0 or w1 == 0:
            continue
        m0 = scores[lo].mean()
        m1 = scores[~lo].mean()
        bc = w0 * w1 * (m0 - m1) ** 2
        if bc > best_bc:
            best_bc, best_t = bc, t
    return float(best_t)


def thresholded_metrics(y, prob, t):
    """ACC/SEN/SPEC/F1 for predicted = (prob > t). NaN where undefined."""
    p = (np.asarray(prob) > t).astype(int)
    tp = int(((p == 1) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    sen = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    acc = (tp + tn) / len(y)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    return acc, sen, spec, f1


def fold_preds(exp_id):
    """Yield (kind, seed, fold, df) for every stored fold of exp_id."""
    if exp_id in PROPOSAL_IDS:
        kind, rep, file = "proposal", FEATURE_SET, "predictions.csv"
    else:
        kind, rep = "baseline", BASELINE_METHODS[exp_id]
        file = "predictions.csv" if exp_id in ("fnn", "fnn_cat") \
            else "val_preds.csv"
    for seed in SEEDS:
        for fold in FOLD_NAMES:
            p = BASE / exp_id / rep / f"seed{seed}" / f"fold{fold}" / file
            df = pd.read_csv(p)
            assert len(df) == 40, (exp_id, seed, fold, len(df))
            yield kind, seed, fold, df


def collect():
    rows = []
    for exp_id in EXPERIMENTS:
        for kind, seed, fold, df in fold_preds(exp_id):
            y = df["label"].astype(int).to_numpy()
            prob = df["prob"].astype(float).to_numpy()
            t = otsu_threshold(prob)
            acc, sen, spec, f1 = thresholded_metrics(y, prob, t)
            rows.append({"experiment": exp_id, "kind": kind, "seed": seed,
                         "fold": fold, "auc": roc_auc_score(y, prob),
                         "t_otsu": t, "acc": acc, "sen": sen, "spec": spec,
                         "f1": f1})
    return pd.DataFrame(rows)


def main():
    fold_df = collect()
    assert len(fold_df) == len(EXPERIMENTS) * len(SEEDS) * len(FOLD_NAMES)
    fold_df.to_csv(BASE / "otsu_metrics.csv", index=False)

    seed_df = (fold_df.groupby(["experiment", "kind", "seed"])
               [["auc", "t_otsu", "acc", "sen", "spec", "f1"]].mean().reset_index())
    seed_df = seed_df.rename(columns={"t_otsu": "t_otsu_mean"})
    seed_df.to_csv(BASE / "otsu_seed_metrics.csv", index=False)

    print(f"wrote {BASE / 'otsu_metrics.csv'} ({len(fold_df)} fold rows, "
          f"{fold_df.experiment.nunique()} experiments)")
    print(f"wrote {BASE / 'otsu_seed_metrics.csv'} ({len(seed_df)} seed rows)")


if __name__ == "__main__":
    main()
