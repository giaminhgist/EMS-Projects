"""Figure 6 — gaze density heatmaps (HC vs SZ) + category & stimulus importance.

(a) Fixation-density heatmaps on the four median-rule stimuli (T01.04, the
    same selection as Figure 2d): rows HC / SZ / difference (SZ − HC),
    columns = stimulus categories. Density = fixations per viewing subject
    (16 px bins, Gaussian-smoothed), overlaid on the grayscale stimulus. All
    HC/SZ panels share one color scale; the difference row uses a symmetric
    diverging scale centered at 0.
(b) Stimulus importance: per-stimulus leave-one-out AUC drop (mlp_attn
    EXP-PROP-003, seed 42, fold Set_0 val; data from T06.01), top 12 + bottom
    12 stimuli, colored by category.
(c) Category importance: leave-one-category-out AUC drop on the same model and
    fold — all stimuli of one category are masked from the subject
    representation; ΔAUC = base AUC − masked AUC.

Data provenance: fixations_cleaned.pkl filtered to partition == "train"
(subject ids 0-47 collide with the official test partition), labels via
rawdata.labels_series(). Style: Nature (bold lowercase panel labels, thin
spines, Okabe-Ito category colors).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FIG, TAB, CACHE, RAW, CAT_COLORS, savefig, panel_label,  # noqa: E402
                    image_list, category_of, make_datasets,
                    load_checkpoint_model)
from model_utils import forward_full, batch_tensors  # noqa: E402
from rawdata import labels_series  # noqa: E402

OUT = FIG
ATTN_ABL = "mlp_attn"  # EXP-PROP-003 (attention pooling; LOO model)

CAT_ORDER = ["social", "natural", "synthetic", "manipulated"]
CAT_FOLDERS = {"social": "Social Scenes", "natural": "Natural Scenes",
               "synthetic": "Synthetic Images",
               "manipulated": "Manipulated Images"}
IMG_ROOT = RAW / "Images"


def gaze_density(fix, stim, subj_ids, bins=(64, 48), sigma=1.5):
    """Smoothed fixation density (fixations per viewing subject) on
    [0, 1024) x [0, 768); returns (array[ny, nx] for imshow origin='lower',
    n_viewed, n_fixations)."""
    d = fix[(fix.IMAGE == stim) & fix.subject_id.isin(subj_ids)]
    n_viewed = int(d.subject_id.nunique())
    if n_viewed == 0:
        return np.zeros((bins[1], bins[0])), 0, 0
    H, _, _ = np.histogram2d(d.FIX_X, d.FIX_Y, bins=bins,
                             range=[[0, 1024], [0, 768]])
    H = gaussian_filter(H, sigma=sigma)
    return H.T / n_viewed, n_viewed, len(d)


def fig_xai_gaze():
    """Figure_6: (a) group gaze heatmaps + difference, (b) stimulus
    importance, (c) category importance."""
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    # --- gaze densities (panel a) --------------------------------------
    fix = pd.read_pickle(CACHE / "fixations_cleaned.pkl")
    fix = fix[fix.partition == "train"]  # ids 0-47 collide with test
    lab = labels_series()
    hc_ids = lab[lab == 0].index.to_numpy()
    sz_ids = lab[lab == 1].index.to_numpy()
    stims = pd.read_csv(TAB / "T01.04_selected_stimuli.csv")
    picks = {row.category: row.stimulus for row in stims.itertuples()}

    maps, stats = {}, []
    for cat in CAT_ORDER:
        stim = picks[cat]
        img = plt.imread(IMG_ROOT / CAT_FOLDERS[cat] / stim)
        maps[("img", cat)] = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        for key, ids in [("hc", hc_ids), ("sz", sz_ids)]:
            m, n_viewed, n_fix = gaze_density(fix, stim, ids)
            maps[(cat, key)] = m
            stats.append(dict(stimulus=stim, category=cat, group=key.upper(),
                              n_subjects=int(len(ids)), n_viewed=n_viewed,
                              n_fixations=n_fix,
                              fix_per_subject=float(n_fix / n_viewed)))
    vmax_dens = max(np.max(maps[(c, k)]) for c in CAT_ORDER for k in ("hc", "sz"))
    diffs = {c: maps[(c, "sz")] - maps[(c, "hc")] for c in CAT_ORDER}
    vmax_diff = max(np.max(np.abs(d)) for d in diffs.values())
    print(f"  heatmap density vmax = {vmax_dens:.3f} fix/subject, "
          f"diff vmax = {vmax_diff:.3f}")

    # --- leave-one-category-out + stimulus table (panels b, c) ---------
    imgs = image_list()
    cats = np.array([category_of(im) for im in imgs])
    model, _, _ = load_checkpoint_model(ATTN_ABL, 42, "Set_0")
    train_ds, val_ds, tr, va = make_datasets(ATTN_ABL, 42, "Set_0")
    D, mask = batch_tensors(val_ds)
    yv = np.array(val_ds.labels)
    with torch.no_grad():
        base = roc_auc_score(yv, forward_full(model, D, mask)["prob"].numpy())
    drops = {}
    for c in CAT_ORDER:
        m2 = mask.clone()
        m2[:, cats == c] = 0
        with torch.no_grad():
            fwd = forward_full(model, D, m2)
        drops[c] = float(base - roc_auc_score(yv, fwd["prob"].numpy()))
    t = pd.read_csv(TAB / "T06.01_stimulus_importance.csv")
    attn_by_cat = t.groupby("category")["mean_attn_pooled"].mean()
    print(f"  base AUC (mlp_attn, Set_0 val) = {base:.4f}")
    for c in CAT_ORDER:
        print(f"  leave-{c}-out ΔAUC = {drops[c]:+.4f} "
              f"(n_stim={int((cats == c).sum())}, mean attn {attn_by_cat[c]:.4f})")

    # --- layout ----------------------------------------------------------
    fig = plt.figure(figsize=(13.8, 9.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.52], hspace=0.34,
                          wspace=0.30, left=0.045, right=0.965, top=0.965,
                          bottom=0.05)
    gs_a = gs[0, :].subgridspec(3, 4, wspace=0.035, hspace=0.05)

    # (a) heatmap block: 3 rows (HC / SZ / SZ-HC) x 4 categories
    ax_first, ax_dens, ax_diff = None, [], []
    m_dens = m_diff = None
    for r, (key, row_label) in enumerate([("hc", "HC"), ("sz", "SZ"),
                                          ("diff", "SZ − HC")]):
        for ci, cat in enumerate(CAT_ORDER):
            ax = fig.add_subplot(gs_a[r, ci])
            ax.imshow(maps[("img", cat)], cmap="gray", vmin=0, vmax=1)
            if key == "diff":
                m_diff = ax.imshow(diffs[cat], cmap="RdBu_r", alpha=0.65,
                                   vmin=-vmax_diff, vmax=vmax_diff,
                                   extent=[0, 1024, 0, 768], origin="lower")
                ax_diff.append(ax)
            else:
                m_dens = ax.imshow(maps[(cat, key)], cmap="inferno",
                                   alpha=0.55, vmin=0, vmax=vmax_dens,
                                   extent=[0, 1024, 0, 768], origin="lower")
                ax_dens.append(ax)
            ax.set_xlim(0, 1024)
            ax.set_ylim(768, 0)
            ax.axis("off")
            if r == 0:
                ax.set_title(cat, fontsize=9, pad=3)
            if ci == 0:
                ax.text(-0.05, 0.5, row_label, rotation=90,
                        transform=ax.transAxes, va="center", ha="right",
                        fontsize=9)
            if r == 0 and ci == 0:
                ax_first = ax
    ax_first.text(0.02, 0.94, "a", transform=ax_first.transAxes,
                  fontweight="bold", fontsize=11, va="top", ha="left",
                  bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                            pad=1.5))
    fig.colorbar(m_dens, ax=ax_dens, location="right", pad=0.02, shrink=0.92,
                 fraction=0.035, label="fixations per subject (density)")
    fig.colorbar(m_diff, ax=ax_diff, location="right", pad=0.02, shrink=0.92,
                 fraction=0.035, label="Δ fixations per subject (SZ − HC)")

    # (b) stimulus importance — top/bottom 12 by LOO AUC drop
    ax = fig.add_subplot(gs[1, 0:2])
    order = np.argsort(t["loo_auc_drop_set0"].values)
    pick = np.concatenate([order[:12], order[-12:]])
    vals = t["loo_auc_drop_set0"].values[pick]
    cols = [CAT_COLORS[c] for c in t["category"].values[pick]]
    ax.barh(np.arange(24)[::-1], vals, color=cols)
    ax.set_yticks(np.arange(24)[::-1])
    ax.set_yticklabels([t["stimulus"].values[s][:6] for s in pick],
                       fontsize=7)
    ax.axvline(0, color="#333333", lw=0.7)
    ax.set_xlabel("ΔAUC (leave-one-stimulus-out, Set_0 val)")
    panel_label(ax, "b")
    ax.legend(handles=[Patch(color=CAT_COLORS[c], label=c) for c in CAT_ORDER],
              fontsize=7, loc="lower right")

    # (c) category importance — leave-one-category-out
    ax = fig.add_subplot(gs[1, 2])
    x = np.arange(4)
    vals_c = [drops[c] for c in CAT_ORDER]
    ax.bar(x, vals_c, color=[CAT_COLORS[c] for c in CAT_ORDER], width=0.62)
    for i, v in enumerate(vals_c):
        ax.text(i, v + (0.0006 if v >= 0 else -0.0006), f"{v:.4f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.axhline(0, color="#333333", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={int((cats == c).sum())})"
                        for c in CAT_ORDER], fontsize=8)
    ax.set_ylabel("ΔAUC (leave-one-category-out)")
    panel_label(ax, "c")

    for ax in fig.axes:
        for s in ax.spines.values():
            s.set_linewidth(0.6)

    savefig(fig, OUT, "Figure_6")
    pd.DataFrame(stats).to_csv(TAB / "T06.02_gaze_heatmap_stats.csv",
                               index=False)
    pd.DataFrame({"category": CAT_ORDER,
                  "n_stimuli": [int((cats == c).sum()) for c in CAT_ORDER],
                  "base_auc": [base] * 4,
                  "loo_drop": [drops[c] for c in CAT_ORDER],
                  "mean_attn_pooled": [float(attn_by_cat[c])
                                       for c in CAT_ORDER]}) \
        .to_csv(TAB / "T06.02_category_importance.csv", index=False)


if __name__ == "__main__":
    fig_xai_gaze()
    print("done Figure_6 (gaze XAI)")
