"""Focused unit tests for roman_features/code/feature_builder.py.

Covers the feature-math requirements of the ablation spec (tests 1-8);
dimension and isolation tests live in tests/test_isolation.py because they
need the built cache / processed dataset.

Dependency-free on purpose (the shared venv has no pytest and is left
untouched): every function here is `test_*` + plain asserts and is executed
by tests/run_tests.py.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import feature_builder as fb  # noqa: E402


def _close(a, b, tol=1e-9):
    assert abs(a - b) <= tol, f"{a!r} != {b!r} (tol={tol})"


# --------------------------------------------------------------------------- #
# 1. np.diff produces n-1 inferred transitions from n fixations
# --------------------------------------------------------------------------- #
def test_n_minus_one_transitions():
    for n in [1, 2, 5, 100]:
        x = np.linspace(0, 10, n)
        y = np.zeros(n)
        dx, dy, amp, valid = fb.valid_transitions(x, y)
        assert len(dx) == max(n - 1, 0)
        assert len(dy) == max(n - 1, 0)
        assert len(amp) == max(n - 1, 0)
        assert len(valid) == max(n - 1, 0)
    # degenerate: fewer than 2 fixations -> empty
    dx, dy, amp, valid = fb.valid_transitions([3.0], [4.0])
    assert len(dx) == 0 and len(valid) == 0


# --------------------------------------------------------------------------- #
# 2. known right/up/left/down transitions map to the correct angle bins
# --------------------------------------------------------------------------- #
def test_cardinal_directions():
    x = np.array([0.0, 10.0])      # right
    y = np.array([0.0, 0.0])
    _close(fb.transition_angles(x, y)[0], 0.0)

    x = np.array([0.0, 0.0])       # up (screen y grows downward -> dy = -10)
    y = np.array([0.0, -10.0])
    _close(fb.transition_angles(x, y)[0], 90.0)

    x = np.array([0.0, -10.0])     # left
    y = np.array([0.0, 0.0])
    _close(fb.transition_angles(x, y)[0], 180.0)

    x = np.array([0.0, 0.0])       # down
    y = np.array([0.0, 10.0])
    _close(fb.transition_angles(x, y)[0], 270.0)


def test_cardinal_bins():
    """20-degree bins, np.histogram half-open convention (exact edge angles
    fall into the HIGHER bin): right->[000,020), up->[080,100),
    left 180deg->[180,200), down->[260,280)."""
    cases = {
        "right": (np.array([0.0, 10.0]), np.array([0.0, 0.0]), 0),
        "up":    (np.array([0.0, 0.0]), np.array([0.0, -10.0]), 4),
        "left":  (np.array([0.0, -10.0]), np.array([0.0, 0.0]), 9),
        "down":  (np.array([0.0, 0.0]), np.array([0.0, 10.0]), 13),
    }
    for name, (x, y, expected_bin) in cases.items():
        freqs, n_valid = fb.direction_frequencies(x, y, bin_width=20.0)
        assert n_valid == 1, name
        _close(freqs[expected_bin], 1.0)
        _close(freqs.sum(), 1.0)


# --------------------------------------------------------------------------- #
# 3. angles close to 0/360 are assigned correctly
# --------------------------------------------------------------------------- #
def test_angle_wraparound():
    # 359.94 deg (slightly below 360, i.e. slightly downward from right)
    x = np.array([0.0, 1000.0])
    y = np.array([0.0, 1.0])
    a = fb.transition_angles(x, y)[0]
    assert 359.0 < a < 360.0
    freqs, _ = fb.direction_frequencies(x, y, bin_width=20.0)
    _close(freqs[-1], 1.0)                          # bin [340, 360)

    # 0.057 deg (slightly upward from right)
    x = np.array([0.0, 1000.0])
    y = np.array([0.0, -1.0])
    a = fb.transition_angles(x, y)[0]
    assert 0.0 < a < 1.0
    freqs, _ = fb.direction_frequencies(x, y, bin_width=20.0)
    _close(freqs[0], 1.0)                           # bin [0, 20)


# --------------------------------------------------------------------------- #
# 4. zero-amplitude transitions are excluded
# --------------------------------------------------------------------------- #
def test_zero_amplitude_excluded():
    x = np.array([0.0, 10.0, 10.0, 0.0])      # right, zero, left
    y = np.array([0.0, 0.0, 0.0, 0.0])
    angles = fb.transition_angles(x, y)
    assert not np.isnan(angles[0]) and np.isnan(angles[1]) and not np.isnan(angles[2])
    freqs, n_valid = fb.direction_frequencies(x, y, bin_width=20.0)
    assert n_valid == 2
    _close(freqs[0], 0.5)
    _close(freqs[9], 0.5)       # 180 deg lands in bin [180, 200) (half-open)


def test_zero_amplitude_all_excluded():
    x = np.array([5.0, 5.0, 5.0])
    y = np.array([5.0, 5.0, 5.0])
    freqs, n_valid = fb.direction_frequencies(x, y, bin_width=20.0)
    assert n_valid == 0
    assert np.all(freqs == 0.0)                     # documented zero-transition policy


# --------------------------------------------------------------------------- #
# 5. frequency vectors sum to 1 when valid transitions exist
# --------------------------------------------------------------------------- #
def test_freq_sums_to_one():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1024, 200)
    y = rng.uniform(0, 768, 200)
    for width in (20.0, 10.0):
        freqs, n_valid = fb.direction_frequencies(x, y, bin_width=width)
        assert n_valid > 0
        _close(freqs.sum(), 1.0)
        assert np.all((freqs >= 0.0) & (freqs <= 1.0))
        n_bins = len(fb.direction_freq_names(width))
        assert n_bins == (18 if width == 20.0 else 36)
        assert len(freqs) == n_bins
        # names reflect their actual intervals
        edges = fb.direction_bin_edges(width)
        for i, name in enumerate(fb.direction_freq_names(width)):
            assert name == f"geo_dir_freq_{int(edges[i]):03d}_{int(edges[i + 1]):03d}"


# --------------------------------------------------------------------------- #
# 6. degenerate sequences do not produce NaN/infinity in model inputs
# --------------------------------------------------------------------------- #
def test_degenerate_sequences_finite():
    degenerate = [
        ([], []),                      # empty
        ([512.0], [384.0]),            # single fixation
        ([512.0] * 10, [384.0] * 10),  # all zero-amplitude
        ([0.0, 0.0], [0.0, 0.0]),
    ]
    for x, y in degenerate:
        freqs, _ = fb.direction_frequencies(x, y, bin_width=20.0)
        assert np.isfinite(freqs).all()
        feats, _ = fb.direction_freq_features(x, y, bin_width=20.0)
        assert all(np.isfinite(v) for v in feats.values())

    # degenerate K: no valid pair -> NaN diagnostics (policy), never inf
    k = fb.k_features([512.0] * 10, [384.0] * 10, [200.0] * 10,
                      200.0, 1.0, 500.0, 1.0)
    assert np.isnan(k["att_k_mean"])
    assert not any(np.isinf(v) for v in k.values() if not np.isnan(v))

    # constant durations with a valid transition: clamped std -> finite z, K=0
    mean_d, std_d, mean_a, std_a = fb.subject_k_stats(
        [fb.transition_pairs([0.0, 100.0], [0.0, 0.0], [200.0, 200.0])])
    _close(std_d, fb.K_EPS)                        # zero std clamped to EPS
    k = fb.k_features([0.0, 100.0], [0.0, 0.0], [200.0, 200.0],
                      mean_d, std_d, mean_a, std_a)
    _close(k["att_k_mean"], 0.0)                   # constant series -> z = 0
    assert np.isfinite(k["att_k_mean"])


# --------------------------------------------------------------------------- #
# 7. focal-like (long fixation, short saccade) has HIGHER K than
#    ambient-like (short fixation, long saccade)
# --------------------------------------------------------------------------- #
def test_k_ordering():
    # subject session: two stimuli providing the reference statistics.
    # focal stimulus: 400 ms fixation -> 50 px saccade
    # ambient stimulus: 80 ms fixation -> 800 px saccade
    pairs = [
        fb.transition_pairs(np.array([0.0, 50.0]), np.array([0.0, 0.0]),
                            np.array([400.0, 400.0])),
        fb.transition_pairs(np.array([0.0, 800.0]), np.array([0.0, 0.0]),
                            np.array([80.0, 80.0])),
    ]
    mean_d, std_d, mean_a, std_a = fb.subject_k_stats(pairs)

    k_focal = fb.k_features(np.array([0.0, 50.0]), np.array([0.0, 0.0]),
                            np.array([400.0, 400.0]),
                            mean_d, std_d, mean_a, std_a)["att_k_mean"]
    k_ambient = fb.k_features(np.array([0.0, 800.0]), np.array([0.0, 0.0]),
                              np.array([80.0, 80.0]),
                              mean_d, std_d, mean_a, std_a)["att_k_mean"]
    assert k_focal > 0.0
    assert k_ambient < 0.0
    assert k_focal > k_ambient


# --------------------------------------------------------------------------- #
# 8. K statistics are computed across the subject session, not per stimulus
# --------------------------------------------------------------------------- #
def test_k_stats_are_subject_level():
    """Reference stats must not depend on a single stimulus: two sessions with
    identical total pairs but different per-stimulus split must give identical
    stats, and stats must differ from per-stimulus standardization."""
    # session 1: one stimulus with both pair types
    s1 = [
        fb.transition_pairs(np.array([0.0, 50.0, 850.0]), np.array([0.0, 0.0, 0.0]),
                            np.array([400.0, 80.0, 80.0])),   # 2 pairs
    ]
    # session 2: same pairs split across two stimuli
    s2 = [
        fb.transition_pairs(np.array([0.0, 50.0]), np.array([0.0, 0.0]),
                            np.array([400.0, 400.0])),        # 1 pair
        fb.transition_pairs(np.array([0.0, 800.0]), np.array([0.0, 0.0]),
                            np.array([80.0, 80.0])),          # 1 pair
    ]
    stats1 = fb.subject_k_stats(s1)
    stats2 = fb.subject_k_stats(s2)
    for a, b in zip(stats1, stats2):
        _close(a, b)

    # and the subject-level stats differ from per-stimulus standardization of
    # session 2's FIRST stimulus (1 pair: d=400, a=50 -> mean 400/50, while
    # the subject-level means over both pairs are 240/425)
    stim1 = fb.transition_pairs(np.array([0.0, 50.0]), np.array([0.0, 0.0]),
                                np.array([400.0, 400.0]))
    d1, a1, v1 = stim1
    assert not np.isclose(stats2[0], d1[v1].mean())
    assert not np.isclose(stats2[2], a1[v1].mean())


def test_last_fixation_has_no_pair():
    """The last fixation never contributes a K pair (no following saccade)."""
    d, a, valid = fb.transition_pairs(np.array([0.0, 10.0, 20.0]),
                                      np.array([0.0, 0.0, 0.0]),
                                      np.array([100.0, 200.0, 300.0]))
    assert len(d) == 2
    assert np.allclose(d, [100.0, 200.0])
    assert np.allclose(a, [10.0, 10.0])
    assert valid.all()
