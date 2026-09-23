"""Isolation + integration tests for the roman_features/ ablation.

Covers the remaining spec requirements:
  9.  feature dimensions are exactly 45 / 63 / 46 / 64 / 81
  10. base45 feature order matches the original pipeline exactly
  11. no output is written to outputs/ or processed_dataset/
plus the adapter-consistency check: on base45 columns the extended dataset
must produce bit-identical inputs to the original NormativeDataset.

Runs via tests/run_tests.py (dependency-free). Tests that need the built
cache report SKIP when it is absent.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
_ROMAN = _REPO / "roman_features"
sys.path.insert(0, str(_ROMAN / "code"))
sys.path.insert(0, str(_REPO / "src"))

import dataset as ext_ds        # noqa: E402
import run_ablation as rab      # noqa: E402


def _cache_available():
    return (ext_ds.CACHE_PKL.exists() and ext_ds.CACHE_NAMES.exists())


# --------------------------------------------------------------------------- #
# 9. feature dimensions are exactly 45 / 63 / 46 / 64 / 81
# --------------------------------------------------------------------------- #
def test_feature_set_dimensions():
    if not _cache_available():
        print("    SKIP (cache not built)")
        return
    expected = {"base45": 45, "base45_freq20": 63, "base45_k": 46,
                "base45_freq20_k": 64, "base45_freq10": 81,
                "base45_freq10_k": 82}
    for name, dim in expected.items():
        cols = rab.feature_cols(name)
        assert len(cols) == dim, f"{name}: {len(cols)} != {dim}"
        assert len(set(cols)) == len(cols), f"{name}: duplicate feature names"
    # the 4 required primary dims
    for dim in (45, 63, 46, 64, 81):
        assert dim in [len(rab.feature_cols(n)) for n in expected], dim


# --------------------------------------------------------------------------- #
# 10. base45 feature order matches the original pipeline exactly
# --------------------------------------------------------------------------- #
def test_base45_order_matches_original():
    orig_names = [ln.strip() for ln in
                  (_REPO / "processed_dataset" / "feature_names.txt")
                  .read_text().splitlines() if ln.strip()]
    assert len(orig_names) == 45
    assert rab._base45_cols() == orig_names
    if _cache_available():
        ext = ext_ds.extended_feature_names()
        assert ext[:45] == orig_names, "cache base45 order differs from processed schema"


# --------------------------------------------------------------------------- #
# 11. no output is written to outputs/ or processed_dataset/
# --------------------------------------------------------------------------- #
def _snapshot(root: Path):
    return {p: (p.stat().st_mtime_ns, p.stat().st_size)
            for p in root.rglob("*") if p.is_file()}


def test_no_writes_outside_roman_features():
    outputs = _REPO / "outputs"
    processed = _REPO / "processed_dataset"
    before_o, before_p = _snapshot(outputs), _snapshot(processed)

    # exercise the ablation machinery end-to-end up to dataset construction
    if _cache_available():
        cols = rab.feature_cols("base45_freq20_k")
        train_ids = [s for s in sorted(
            ext_ds.load_extended_features().index.get_level_values(0).unique())[:10]]
        ds = ext_ds.ExtendedNormativeDataset(train_ids, train_ids,
                                             feature_cols=cols)
        x, m = ds[0][0]
        assert x.shape == (100, len(cols)) and m.shape == (100,)

    # run-dir redirection: paths must live under roman_features/, never outputs/
    cfg = rab.AblationRunConfig(experiment="A0_base45", feature_set="base45",
                                seed=2026, fold="Set_0")
    run_dir = cfg.run_dir(create=False)
    rel = run_dir.relative_to(_ROMAN)
    assert rel.parts[0] == "results", run_dir
    assert "outputs" not in rel.parts and "processed_dataset" not in rel.parts
    smoke = rab.AblationRunConfig(experiment="A0_base45", feature_set="base45",
                                  seed=2026, fold="Set_0", smoke=True)
    assert smoke.run_dir(create=False).relative_to(_ROMAN).parts[0:2] == \
        ("results", "_smoke")

    after_o, after_p = _snapshot(outputs), _snapshot(processed)
    assert before_o == after_o, "files under outputs/ changed!"
    assert before_p == after_p, "files under processed_dataset/ changed!"


# --------------------------------------------------------------------------- #
# adapter consistency: extended dataset == original NormativeDataset on base45
# --------------------------------------------------------------------------- #
def test_extended_dataset_matches_original_on_base45():
    if not _cache_available():
        print("    SKIP (cache not built)")
        return
    from data.common import train_subject_ids
    from proposal.model import NormativeDataset

    torch.manual_seed(0)
    ids = sorted(train_subject_ids())[:12] + sorted(train_subject_ids())[-12:]
    base_cols = rab._base45_cols()

    orig = NormativeDataset(ids, ids, "learned", feature_cols=base_cols)
    mine = ext_ds.ExtendedNormativeDataset(ids, ids, feature_cols=base_cols)

    assert orig.subjects == mine.subjects
    for i in range(len(ids)):
        (xo, mo), yo, so = orig[i]
        (xm, mm), ym, sm = mine[i]
        assert yo == ym and so == sm
        assert torch.equal(mo, mm), f"mask mismatch for subject {ids[i]}"
        assert torch.equal(xo, xm), f"X mismatch for subject {ids[i]}"
        assert torch.allclose(xo, xm, atol=0.0, rtol=0.0)
    print("    original vs extended datasets bit-identical on base45")


# --------------------------------------------------------------------------- #
# feature-value sanity on the cache: freq sums, finite K, mask preservation
# --------------------------------------------------------------------------- #
def test_cache_mask_preserved_and_values_sane():
    if not _cache_available():
        print("    SKIP (cache not built)")
        return
    import pandas as pd
    orig = pd.read_pickle(_REPO / "processed_dataset" / "stimulus_features_train.pkl")
    ext = ext_ds.load_extended_features()
    assert (orig.isna().all(axis=1) == ext.isna().all(axis=1)).all(), \
        "extended table changed the missing-stimulus mask"
    freq20 = rab.feature_cols("base45_freq20")[45:]
    valid_rows = ~ext[freq20].isna().all(axis=1)
    sums = ext.loc[valid_rows, freq20].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-9), "freq20 rows do not sum to 1"
    assert np.isfinite(ext[freq20].to_numpy()[~ext[freq20].isna()]).all()
    k = ext["att_k_mean"].dropna()
    assert np.isfinite(k).all()
