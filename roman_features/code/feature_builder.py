"""Hand-crafted feature construction for the roman_features/ isolated ablation.

Feature A — inferred-saccade direction frequencies
Feature B — ambient/focal K-index

Both operate on *cleaned* fixations of one (subject, stimulus), ordered by
FIX_INDEX (cleaning + ordering re-use src/preprocess.py::clean_fixations and
the groupby("IMAGE").sort_values("FIX_INDEX") convention of the original
pipeline). Saccades are NOT recorded in the EMS release; every transition is
an inferred saccade from consecutive cleaned fixation centers.

Feature A (direction frequencies)
---------------------------------
dx = np.diff(x), dy = np.diff(y), amplitude = hypot(dx, dy).
Transitions with amplitude <= AMP_EPS (1e-6) are excluded: their direction is
undefined and their angle would be noise (position jitter at the same point).
Angle convention (Cartesian screen coordinates, y grows DOWNWARD):

    angle = (degrees(arctan2(-dy, dx)) + 360) % 360

so 0 = right, 90 = up, 180 = left, 270 = down; angles lie in [0, 360).

Bins: primary 20-degree bins, edges np.arange(0, 361, 20) -> 18 bins,
names geo_dir_freq_{lo:03d}_{hi:03d}; sensitivity variant 10-degree bins
-> 36 bins. Bin assignment uses np.histogram's half-open convention: an angle
that lands exactly on a bin edge falls into the HIGHER bin (e.g. 180 deg ->
bin [180, 200)). Values are RELATIVE frequencies count_b / n_valid (not raw
counts: total fixation count is already represented by spa_fix_count, and a
histogram of raw counts would scale with viewing length).

Zero-valid-transition policy (deterministic, documented): when a NON-MISSING
stimulus (>= 2 clean fixations, i.e. a valid base45 row) has no valid
inferred transition, ALL frequency features are 0.0 — a finite, in-range
value meaning "no directional information". A MISSING stimulus (all-NaN
base45 row, masked by the existing pipeline) keeps NaN in every new column,
so the stimulus mask of the original pipeline is preserved exactly (build_cache
enforces this invariant). Frequency vectors sum to 1 exactly when at least
one valid transition exists.

Feature B (ambient/focal K-index)
---------------------------------
For transition pair i (fixation i with duration d_i, followed by the inferred
saccade i->i+1 with amplitude a_{i+1}; i = 0..n-2):

    K_i = z(d_i) - z(a_{i+1}),   z(x) = (x - mean) / max(std, EPS)

K_i > 0: relatively long fixation followed by a relatively short saccade
(focal-like); K_i < 0: relatively short fixation followed by a relatively
long saccade (ambient-like). Following Krejtz et al. (2016) the amplitude is
used as the saccade-length term (prose definition in the task spec; the
formula there writes the amplitude term as a_i for brevity).

Reference-statistics scope (locked): mean/std of duration and amplitude are
computed PER SUBJECT over ALL valid fixation->saccade pairs of that subject's
full viewing session (all stimuli). No diagnosis labels are involved.
Stimuli are NOT standardized independently: that would force every
stimulus-level mean K toward zero and destroy the ambient/focal contrast.

Zero-standard-deviation policy (documented): std is clamped to EPS = 1e-6.
A constant series then yields z = 0 exactly (x == mean whenever std == 0),
so no division blow-up can reach the model.

Model feature: att_k_mean = arithmetic mean of K_i within the stimulus
(one scalar). Diagnostics only (NOT model inputs unless a future config
explicitly adds them): att_k_std, att_k_frac_pos, att_k_slope.

No-valid-transition policy: att_k_mean (and diagnostics) are NaN, which the
existing pipeline handles as a missing *feature* cell (filled with the
train-fitted per-feature mean -> zero deviation). A stimulus with < 2 clean
fixations is already an all-NaN row (missing stimulus) in the base table,
and every new column stays NaN there. n >= 2 clean fixations always yields
>= 1 pair, so for valid rows att_k_mean is NaN only when every transition of
the stimulus is zero-amplitude. A subject with no valid pair in the whole
session has undefined reference stats -> att_k_mean NaN on all their stimuli.
"""
import numpy as np

AMP_EPS = 1e-6   # inferred saccades with amplitude <= AMP_EPS have no direction
K_EPS = 1e-6     # std clamp for the K-index z-scores


# --------------------------------------------------------------------------- #
# Feature A — direction frequencies
# --------------------------------------------------------------------------- #

def direction_bin_edges(bin_width=20.0):
    """np.arange(0, 361, bin_width): `n_bins` bins covering [0, 360)."""
    return np.arange(0.0, 361.0, bin_width)


def direction_freq_names(bin_width=20.0):
    """Ordered feature names geo_dir_freq_{lo:03d}_{hi:03d} for the bins."""
    edges = direction_bin_edges(bin_width)
    return [f"geo_dir_freq_{int(edges[i]):03d}_{int(edges[i + 1]):03d}"
            for i in range(len(edges) - 1)]


def valid_transitions(x, y):
    """(dx, dy, amp, valid) for consecutive fixation centers of one stimulus.

    valid marks amp > AMP_EPS. Returns empty arrays for n < 2 fixations.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return (np.array([]), np.array([]), np.array([]), np.array([], dtype=bool))
    dx = np.diff(x)
    dy = np.diff(y)
    amp = np.hypot(dx, dy)
    valid = amp > AMP_EPS
    return dx, dy, amp, valid


def transition_angles(x, y):
    """Angles in [0, 360) for each transition; NaN where the transition is
    zero-amplitude (direction undefined). 0=right, 90=up, 180=left, 270=down.
    """
    dx, dy, amp, valid = valid_transitions(x, y)
    angles = np.full(len(dx), np.nan, dtype=np.float64)
    if valid.any():
        angles[valid] = (np.degrees(np.arctan2(-dy[valid], dx[valid])) + 360.0) % 360.0
    return angles


def direction_frequencies(x, y, bin_width=20.0):
    """Relative direction-frequency vector for one stimulus.

    Returns (freqs, n_valid): freqs sums to 1.0 when n_valid > 0; the all-zero
    vector otherwise (documented zero-transition policy, see module docstring).
    Never NaN/inf.
    """
    angles = transition_angles(x, y)
    valid = ~np.isnan(angles)
    n_valid = int(valid.sum())
    edges = direction_bin_edges(bin_width)
    freqs = np.zeros(len(edges) - 1, dtype=np.float64)
    if n_valid:
        counts, _ = np.histogram(angles[valid], bins=edges)
        freqs = counts.astype(np.float64) / n_valid
    return freqs, n_valid


def direction_freq_features(x, y, bin_width=20.0):
    """Dict feature_name -> float for one stimulus (18 or 36 features)."""
    freqs, n_valid = direction_frequencies(x, y, bin_width)
    return {name: float(v) for name, v in zip(direction_freq_names(bin_width), freqs)}, n_valid


# --------------------------------------------------------------------------- #
# Feature B — ambient/focal K-index
# --------------------------------------------------------------------------- #

def transition_pairs(x, y, dur):
    """(d, a, valid) arrays for the fixation->saccade pairs of one stimulus.

    Pair i = (duration d_i, amplitude a_{i+1} of the inferred saccade from
    fixation i to fixation i+1), i = 0..n-2. The LAST fixation has no
    following saccade and never contributes a pair. valid = amp > AMP_EPS.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dur = np.asarray(dur, dtype=np.float64)
    if len(x) < 2:
        return (np.array([]), np.array([]), np.array([], dtype=bool))
    _, _, amp, valid = valid_transitions(x, y)
    d = dur[:-1]
    a = amp
    return d, a, valid


def subject_k_stats(pairs):
    """(mean_d, std_d, mean_a, std_a) over one subject's session pairs.

    `pairs` is an iterable of (d, a, valid) per stimulus; only VALID pairs
    (amp > AMP_EPS) enter the statistics. Standard deviations are clamped to
    K_EPS (documented zero-std policy). Returns NaNs when the subject has no
    valid pair at all.
    """
    d_all = np.concatenate([d[v] for d, a, v in pairs if v.any()]) \
        if any(v.any() for _, _, v in pairs) else np.array([])
    a_all = np.concatenate([a[v] for d, a, v in pairs if v.any()]) \
        if any(v.any() for _, _, v in pairs) else np.array([])
    if len(d_all) == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean_d = float(d_all.mean())
    std_d = float(np.maximum(d_all.std(), K_EPS))
    mean_a = float(a_all.mean())
    std_a = float(np.maximum(a_all.std(), K_EPS))
    return mean_d, std_d, mean_a, std_a


def k_values(x, y, dur, mean_d, std_d, mean_a, std_a):
    """Per-pair K_i for one stimulus given the SUBJECT-level reference stats.

    K_i = z(d_i) - z(a_{i+1}); NaN for invalid pairs or when the subject-level
    stats are undefined (no valid pair in the whole session).
    """
    d, a, valid = transition_pairs(x, y, dur)
    if len(d) == 0:
        return np.array([]), valid
    if np.isnan(mean_d):
        return np.full(len(d), np.nan), valid
    z_d = (d - mean_d) / std_d
    z_a = (a - mean_a) / std_a
    k = z_d - z_a
    k = np.where(valid, k, np.nan)
    return k, valid


def k_features(x, y, dur, mean_d, std_d, mean_a, std_a):
    """Per-stimulus K-index summary: model feature att_k_mean + diagnostics.

    att_k_mean = arithmetic mean of K_i over the valid pairs of the stimulus
    (the single scalar model input for the K ablation). Diagnostics
    (att_k_std, att_k_frac_pos, att_k_slope) are computed but are NOT part of
    the primary model inputs. All values NaN when the stimulus has no valid
    pair (existing missing-feature policy fills them to zero deviation).
    """
    k, valid = k_values(x, y, dur, mean_d, std_d, mean_a, std_a)
    k_valid = k[valid]
    out = {"att_k_mean": np.nan, "att_k_std": np.nan,
           "att_k_frac_pos": np.nan, "att_k_slope": np.nan}
    if len(k_valid):
        out["att_k_mean"] = float(k_valid.mean())
        if len(k_valid) > 1:
            out["att_k_std"] = float(k_valid.std(ddof=1))
            out["att_k_frac_pos"] = float((k_valid > 0).mean())
            idx = np.nonzero(valid)[0].astype(np.float64)
            out["att_k_slope"] = float(np.polyfit(idx, k_valid, 1)[0])
    return out
