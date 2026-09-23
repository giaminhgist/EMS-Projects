"""Dataset adapter for the roman_features/ isolated ablation.

Strategy: import and reuse the existing model/dataset/trainer unchanged, and
re-implement ONLY the three data-LOADING functions of src/data/tabular.py so
they read the extended feature cache (roman_features/cache) instead of
processed_dataset/. The pure functions (fill_missing_cells, apply_deviation)
and the dataset contract (collate, hc_indices) are inherited from
proposal.model.NormativeDataset — only its __init__ loading part is
overridden (deviation='learned', the EXP-PROP-001 mode).

No source file outside roman_features/ is modified. Consistency with the
original pipeline is asserted by tests/test_isolation.py: on base45 columns
this adapter must produce bit-identical inputs to the original
NormativeDataset.

Missingness rules are IDENTICAL to src/data/tabular.py (documented there):
  - all-NaN row = missing stimulus -> masked everywhere (the cache preserves
    the original missing-row set exactly, asserted at build time);
  - single NaN cell in an otherwise valid row = missing feature -> filled
    with the train-fitted per-feature mean -> zero deviation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROMAN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROMAN.parent / "src"))            # EMS-Projects/src

from data.common import NUM_STIMULI, image_list, labels_of  # noqa: E402
from data.tabular import EPS, fill_missing_cells            # noqa: E402 (pure)
from proposal.model import NormativeDataset                 # noqa: E402

CACHE_PKL = _ROMAN / "cache" / "features_extended.pkl"
CACHE_NAMES = _ROMAN / "cache" / "feature_names_extended.txt"


def load_extended_features():
    """Extended (subject_id, image) table: 45 base + freq20 + freq10 + k."""
    return pd.read_pickle(CACHE_PKL)


def extended_feature_names():
    """Ordered column list of the extended cache (103 names)."""
    return [ln.strip() for ln in CACHE_NAMES.read_text().splitlines()
            if ln.strip()]


def subject_matrices(subject_ids, feature_cols):
    """{subject_id: (X (S, D) float32, mask (S,) bool)} from the extended cache.

    Mirror of src/data/tabular.subject_matrices (train partition only — the
    official 4-fold protocol uses the 160 Train_Valid subjects). X rows are
    NaN for missing stimuli or missing feature cells; mask marks rows that
    are not entirely NaN.
    """
    feature_cols = list(feature_cols)
    feat = load_extended_features()
    images = image_list()
    feat_ids = set(feat.index.get_level_values(0))
    out = {}
    for sid in subject_ids:
        key = int(sid)
        sub = feat.loc[key] if key in feat_ids else None
        if sub is None or len(sub) == 0:
            X = np.full((NUM_STIMULI, len(feature_cols)), np.nan, dtype=np.float32)
            mask = np.zeros(NUM_STIMULI, dtype=bool)
        else:
            X = sub.reindex(images)[feature_cols].to_numpy(dtype=np.float32)
            mask = ~np.isnan(X).all(axis=1)
        out[int(sid)] = (X, mask)
    return out


def feature_norm_stats(subject_ids, feature_cols):
    """Per-feature mean/std over the given subjects (unsupervised scaling).

    Mirror of src/data/tabular.feature_norm_stats on the extended cache.
    Fits on the passed subject ids only — call with the training fold to
    avoid leakage. Zero-variance features: std clamped to EPS.
    """
    mats = subject_matrices(subject_ids, feature_cols)
    X = np.concatenate([m[0][m[1]] for m in mats.values()], axis=0)
    mean = np.nanmean(X, axis=0).astype(np.float32)
    std = np.nanstd(X, axis=0).astype(np.float32)
    std = np.maximum(std, EPS)
    return mean, std


class ExtendedNormativeDataset(NormativeDataset):
    """NormativeDataset for the extended feature table (learned mode only).

    Overrides ONLY the loading part of the parent __init__ (subject matrices +
    train-fitted per-feature normalization stats read from the extended
    cache); the filling, standardization, masking, collate and hc_indices
    behavior is inherited unchanged. deviation is fixed to 'learned' — the
    EXP-PROP-001 configuration, the only mode exercised by this ablation.
    """

    def __init__(self, subject_ids, train_ids_for_stats, feature_cols=None):
        self.subjects = list(subject_ids)
        self.labels = labels_of(self.subjects)
        self.deviation = "learned"
        self.stim_subset = None
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        mats = subject_matrices(self.subjects, feature_cols=self.feature_cols)
        mu, sigma = feature_norm_stats(train_ids_for_stats,
                                       feature_cols=self.feature_cols)
        fill = mu  # per-feature (D,) means from train subjects
        self.X, self.mask = [], []
        for s in self.subjects:
            X, mask = mats[s]
            X = fill_missing_cells(X, fill)
            D = (X - mu) / sigma
            D = D * mask[:, None].astype(np.float32)
            self.X.append(torch.from_numpy(D.astype(np.float32)))
            self.mask.append(torch.from_numpy(mask.astype(np.float32)))
