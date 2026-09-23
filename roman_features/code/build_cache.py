"""Build the extended feature cache for the roman_features/ isolated ablation.

Reads (never writes):
  - processed_dataset/stimulus_features_train.pkl   (base45, authoritative)
  - original_dataset/EMS raw fixation xlsx          (via src.common + cleaning)

Writes (only under roman_features/):
  - cache/features_extended.pkl      base45 + direction freqs (20/10 deg) + K
  - cache/feature_names_extended.txt ordered column list of the cache
  - cache/transition_counts.csv      per-(subject, image) transition audit
  - cache/build_report.txt           build log + missingness audit

The 45 base columns are copied VERBATIM from processed_dataset and asserted
equal (pd.testing.assert_frame_equal); the new columns are aligned to the
same (subject_id, image) MultiIndex grid (all stimuli, missing pairs all-NaN),
exactly like src/preprocess.py does.

Feature-set column definitions (dims):
  base45          = base45                            (45)
  base45_freq20   = base45 + 18 geo_dir_freq_* 20deg  (63)
  base45_k        = base45 + att_k_mean               (46)
  base45_freq20_k = base45 + 18 geo_dir_freq_* + att_k_mean (64)
  base45_freq10   = base45 + 36 geo_dir_freq_* 10deg  (81)
  base45_freq10_k = base45 + 36 geo_dir_freq_* + att_k_mean (82, optional)

Usage (from repo root):
    .venv/bin/python roman_features/code/build_cache.py            # all subjects
    .venv/bin/python roman_features/code/build_cache.py --subjects 0 1 2 3 200
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROMAN = _HERE.parent
_CACHE = _ROMAN / "cache"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "src"))       # EMS-Projects/src
sys.path.insert(0, str(_HERE.parents[1]))               # EMS-Projects (repo root)

from common import load_all                                    # noqa: E402
from preprocess import clean_fixations                         # noqa: E402
from feature_builder import (direction_freq_names,             # noqa: E402
                             direction_freq_features,
                             transition_pairs, subject_k_stats,
                             k_features)

FREQ20_W, FREQ10_W = 20.0, 10.0
K_COLS = ["att_k_mean", "att_k_std", "att_k_frac_pos", "att_k_slope"]

FEATURE_SETS = {
    "base45": None,                    # resolved against the base table below
    "base45_freq20": None,
    "base45_k": None,
    "base45_freq20_k": None,
    "base45_freq10": None,
    "base45_freq10_k": None,
}


def stimulus_sequences(clean_df):
    """Yield (image, x, y, dur) for the cleaned fixations of one subject,
    ordered by FIX_INDEX (the original pipeline's ordering convention)."""
    for image, grp in clean_df.groupby("IMAGE"):
        grp = grp.sort_values("FIX_INDEX")
        yield image, (grp.FIX_X.values.astype(np.float64),
                      grp.FIX_Y.values.astype(np.float64),
                      grp.FIX_DURATION.values.astype(np.float64))


def build_extended_table(subject_ids):
    """Extended (subject_id, image) table for the given train subjects."""
    base = pd.read_pickle(Path(_HERE.parents[1]) /
                          "processed_dataset" / "stimulus_features_train.pkl")
    base_cols = list(base.columns)
    if subject_ids is not None:
        base = base[base.index.get_level_values(0).isin(subject_ids)]

    images = sorted(base.index.get_level_values(1).unique())
    grid = pd.MultiIndex.from_product(
        [sorted(base.index.get_level_values(0).unique()), images],
        names=["subject_id", "image"])

    raw = load_all("train")
    if subject_ids is not None:
        raw = raw[raw.subject_id.isin(subject_ids)]
    raw = raw.groupby("subject_id")

    freq20_cols = direction_freq_names(FREQ20_W)
    freq10_cols = direction_freq_names(FREQ10_W)
    frames = []
    audit_rows = []
    for sid, subj_df in raw:
        clean, _drop = clean_fixations(subj_df)
        seqs = {image: (x, y, dur) for image, (x, y, dur) in stimulus_sequences(clean)}
        rows20, rows10, rows_k = {}, {}, {}
        for image, (x, y, dur) in seqs.items():
            f20, n_valid = direction_freq_features(x, y, FREQ20_W)
            f10, _ = direction_freq_features(x, y, FREQ10_W)
            rows20[image] = f20
            rows10[image] = f10
            audit_rows.append({"subject_id": sid, "image": image,
                               "n_fix_clean": len(x), "n_valid_trans": n_valid})
        pairs = [transition_pairs(*seqs[im]) for im in seqs]
        mean_d, std_d, mean_a, std_a = subject_k_stats(pairs)
        for image, (x, y, dur) in seqs.items():
            rows_k[image] = k_features(x, y, dur, mean_d, std_d, mean_a, std_a)

        f20df = pd.DataFrame(rows20).T.reindex(columns=freq20_cols)
        f10df = pd.DataFrame(rows10).T.reindex(columns=freq10_cols)
        kdf = pd.DataFrame(rows_k).T.reindex(columns=K_COLS)
        subj = pd.concat([f20df, f10df, kdf], axis=1)
        subj["subject_id"] = sid
        subj["image"] = subj.index
        frames.append(subj.reset_index(drop=True))

    new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=freq20_cols + freq10_cols + K_COLS)
    new = new.set_index(["subject_id", "image"]).reindex(grid).sort_index()

    audit = pd.DataFrame(audit_rows).set_index(["subject_id", "image"]) \
        .reindex(grid).sort_index()

    base_grid = base.reindex(grid)
    # INVARIANT (mask preservation): a (subject, stimulus) row that is all-NaN
    # in base45 is a *missing stimulus* in the original pipeline. New columns
    # must be NaN on exactly those rows, otherwise a zero frequency vector
    # would resurrect the stimulus into the mask with imputed base45 values.
    # The zero-vector direction policy therefore applies only to rows VALID
    # in base45 that have zero valid transitions (n >= 2 fixations).
    missing = base_grid[base_cols].isna().all(axis=1)
    new.loc[missing] = np.nan
    extended = pd.concat([base_grid[base_cols], new], axis=1)
    # the 45 base columns must be copied verbatim (never recomputed)
    pd.testing.assert_frame_equal(extended[base_cols], base_grid[base_cols])
    # the extended table must mask exactly the same stimuli as the original
    assert (extended.isna().all(axis=1) == missing).all()
    return extended, audit, base_cols, freq20_cols, freq10_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="*", default=None,
                    help="restrict to these train subject ids (smoke builds)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    extended, audit, base_cols, freq20_cols, freq10_cols = \
        build_extended_table(args.subjects)
    n_rows = len(extended)
    lines = [
        "roman_features/ cache build report", "=" * 40,
        f"rows {n_rows} x cols {extended.shape[1]} "
        f"(subjects={extended.index.get_level_values(0).nunique()}, "
        f"images={extended.index.get_level_values(1).nunique()})",
        f"columns: {len(base_cols)} base45 + {len(freq20_cols)} freq20 "
        f"+ {len(freq10_cols)} freq10 + {len(K_COLS)} k = {extended.shape[1]}",
    ]

    # missingness audit
    for name, cols in [("base45", base_cols), ("freq20", freq20_cols),
                       ("freq10", freq10_cols), ("k", K_COLS)]:
        sub = extended[cols]
        lines.append(f"[{name}] NaN cells: {int(sub.isna().sum().sum())}/"
                     f"{sub.shape[0] * sub.shape[1]} "
                     f"({sub.isna().mean().mean() * 100:.3f}% of cells)")
    all_nan = extended.isna().all(axis=1)
    lines.append(f"all-NaN (missing stimulus) rows: {int(all_nan.sum())}/{n_rows}")
    # rows valid in base45 but with zero valid transitions (direction policy)
    base_valid = ~extended[base_cols].isna().all(axis=1)
    zero_trans = base_valid & (audit["n_valid_trans"] == 0)
    lines.append(f"rows valid in base45 with 0 valid transitions "
                 f"(freq=0-vector, k=NaN policy): {int(zero_trans.sum())}")
    n_k_nan = int(extended["att_k_mean"].isna().sum())
    lines.append(f"att_k_mean NaN cells: {n_k_nan}/{n_rows} "
                 f"({n_k_nan / n_rows * 100:.3f}%)")
    lines.append("transition audit (per valid row): "
                 f"n_valid_trans min={audit.n_valid_trans.min()} "
                 f"mean={audit.n_valid_trans.mean():.1f} "
                 f"max={audit.n_valid_trans.max()}")
    lines.append("att_k_mean over valid cells: "
                 f"min={extended.att_k_mean.min():.3f} "
                 f"mean={extended.att_k_mean.mean():.3f} "
                 f"max={extended.att_k_mean.max():.3f}")

    _CACHE.mkdir(parents=True, exist_ok=True)
    extended.to_pickle(_CACHE / "features_extended.pkl")
    audit.to_csv(_CACHE / "transition_counts.csv")
    col_order = base_cols + freq20_cols + freq10_cols + K_COLS
    (_CACHE / "feature_names_extended.txt").write_text(
        "\n".join(col_order) + "\n")
    report = "\n".join(lines) + "\n"
    (_CACHE / "build_report.txt").write_text(report)
    print(report)
    print(f"saved cache/features_extended.pkl (shape {extended.shape})")


if __name__ == "__main__":
    main()
