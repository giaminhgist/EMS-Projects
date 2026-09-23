"""Figure 4 — learned latent space vs the HC normative bank.

One merged figure: (a) joint PCA of bank-centered encodings, (b) subject-mean
PC1 distributions, (c) deviation by stimulus category, (d) subject x stimulus
deviation heatmap, (e) train-HC distance-to-bank diagnostic.

Notation follows the actual tensors (see cache/latent/*.npz exported by
model_utils.export_run):
  - z: encoder output BEFORE the comparator, [B, S, 128] (learned models only)
  - d: deviation AFTER the comparator / hard deviation, [B, S, dev_dim]
  - h: subject embedding AFTER pooling
  - bank_mu / bank_sigma: the checkpoint's buffer (bank refreshed at the start
    of the best-checkpoint epoch — used as-is, never refreshed here)
  - ref_z: training-fold HC re-encoded with the FINAL encoder weights
    (empirical reference; may differ slightly from the epoch-start bank)

All analyses are per model (fold/seed), never pooling latents of
independently trained encoders into one PCA.

ADAPTATION vs the reference suite: this project's protocol has NO
lambda_norm (removed before the 10-seed runs), so the old panels
"(c) effective rank lambda=0 vs 0.1" and "(e) lambda concentration"
are dropped. Panel (c) now shows subject-mean RMS deviation by stimulus
category (HC vs SZ); panel (e) is the remaining single-model
concentration check; the effective rank (participation ratio) of the
trained model's train-HC covariance is kept only as a printed/CSV
diagnostic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse, Patch
from scipy import stats
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FIG, TAB, CACHE, FOLDS, HC_COLOR, SZ_COLOR, REF_COLOR,  # noqa: E402
                    LEARNED_COLOR, HARD_COLOR, savefig, image_list, category_of)

OUT = FIG
LAT = CACHE / "latent"

MAIN_ABL = "mlp_deepset"  # EXP-PROP-001, the main proposal (best AUC)


def load_latent(ablation, seed, folds=FOLDS):
    """Concatenate the latent exports of several folds (same ablation/seed).

    Subject-level arrays (z, d, h, mask, label, subject_id) are concatenated;
    per-fold statistics (bank_mu/bank_sigma, ref_z/ref_mask) are kept per fold
    so each subject is centered with ITS fold's bank.
    """
    per_fold = []
    for f in folds:
        d = np.load(LAT / f"{ablation}__seed{seed}__fold{f}.npz", allow_pickle=False)
        arrs = {}
        for k in d.files:
            arr = d[k]
            if arr.dtype.kind in "OUS":
                arr = arr.item() if arr.ndim == 0 else arr
            arrs[k] = arr
        per_fold.append(arrs)
    res = {"_per_fold": per_fold}
    concat = ["z", "d", "h", "attn", "mask", "label", "subject_id", "prob"]
    for k in concat:
        if k in per_fold[0] and isinstance(per_fold[0][k], np.ndarray) \
                and per_fold[0][k].ndim > 0:
            res[k] = np.concatenate([p[k] for p in per_fold], axis=0)
        elif k in per_fold[0]:
            res[k] = np.array([p[k] for p in per_fold])
    res["fold_of"] = np.concatenate([
        np.full(len(p["label"]), i) for i, p in enumerate(per_fold)])
    return res


def center_per_fold(lat, kind="z"):
    """Per-stimulus centered encodings (z - that fold's bank_mu) over folds."""
    parts = []
    for i, p in enumerate(lat["_per_fold"]):
        if kind == "z":
            parts.append(p["z"] - p["bank_mu"][None])
        else:  # ref (train-HC re-encoded)
            parts.append(p["ref_z"] - p["bank_mu"][None])
    return np.concatenate(parts, axis=0)


def masks_per_fold(lat, kind="z"):
    parts = [p["mask"] if kind == "z" else p["ref_mask"]
             for p in lat["_per_fold"]]
    return np.concatenate(parts, axis=0)


def rms_std_residual(z, mu, sigma, mask):
    """Per (subject, stimulus) RMS standardized residual to the bank:
    sqrt( mean_k ((z-mu)/sigma)^2 ) over latent dims — comparable across
    dimensionalities (RMS per dimension)."""
    L = z.shape[-1]
    m = mask
    r = np.sqrt((((z - mu[None]) / sigma[None]) ** 2).sum(-1) / L)
    r[~m] = np.nan
    return r


def subject_rms(r):
    return np.nanmean(r, axis=1)


# --------------------------------------------------------------------- #
# Figure_4 learned latent space vs the HC normative bank
# --------------------------------------------------------------------- #
def fig_normative_latent():
    """One figure, five panels: (a) joint PCA, (b) subject-mean PC1,
    (c) deviation by stimulus category, (d) subject x stimulus deviation
    heatmap, (e) train-HC distance to the bank (concentrated, not 0)."""
    lat = load_latent(MAIN_ABL, 42)
    y = lat["label"]

    # -- PCA (panels a, b): fit on the train-HC reference --------------
    zc = center_per_fold(lat, "z")
    zc_ref = center_per_fold(lat, "ref")
    mask = masks_per_fold(lat, "z")
    m_ref = masks_per_fold(lat, "ref")
    flat_ref = zc_ref[m_ref].reshape(-1, 128)
    flat_eval = zc[mask].reshape(-1, 128)
    pca = PCA(n_components=2, random_state=0).fit(flat_ref)
    print("  PCA explained variance (fit on train-HC reference):",
          np.round(pca.explained_variance_ratio_, 3))
    proj_ref = pca.transform(flat_ref)
    proj_eval = pca.transform(flat_eval)
    proj_eval_full = np.full((zc.shape[0] * zc.shape[1], 2), np.nan)
    proj_eval_full[mask.reshape(-1)] = proj_eval
    proj_eval_full = proj_eval_full.reshape(zc.shape[0], zc.shape[1], 2)
    subj_mean = np.array([np.nanmean(proj_eval_full[i][mask[i]], axis=0)
                          for i in range(len(zc))])
    hc = subj_mean[y == 0]
    sz = subj_mean[y == 1]

    # -- RMS standardized residual matrix (panel c) --------------------
    r_parts = []
    for p in lat["_per_fold"]:
        r_parts.append(rms_std_residual(p["z"], p["bank_mu"], p["bank_sigma"],
                                        p["mask"]))
    r = np.concatenate(r_parts, axis=0)
    imgs = image_list()
    cats = np.array([category_of(im) for im in imgs])
    order = np.argsort(cats, kind="stable")
    r_sorted = r[:, order]
    idx_hc = np.where(y == 0)[0]
    idx_sz = np.where(y == 1)[0]

    def pick(idx):
        m = subject_rms(r)[idx]
        top = idx[np.argsort(m)[-15:]]
        bot = idx[np.argsort(m)[:15]]
        return np.concatenate([bot, top])
    heat_rows = np.concatenate([pick(idx_hc), pick(idx_sz)])

    # -- diagnostics (panel e + CSV): single model, no lambda_norm -----
    dists, zcs = [], []
    for p in lat["_per_fold"]:
        dist = np.sqrt(((p["ref_z"] - p["bank_mu"][None]) ** 2).sum(-1))
        dist[~p["ref_mask"]] = np.nan
        dists.append(dist)
        zcs.append((p["ref_z"] - p["bank_mu"][None])[p["ref_mask"]])
    dist = np.concatenate(dists, axis=0)
    subj_dist = np.nanmean(dist, axis=1)
    cov = np.cov(np.concatenate(zcs, axis=0).T)
    eig = np.linalg.eigvalsh(cov)
    eig = eig[eig > 1e-12]
    pr = float((eig.sum() ** 2) / (eig ** 2).sum())  # participation ratio
    print(f"  effective rank (participation ratio) of train-HC covariance: {pr:.1f}")
    print(f"  train-HC per-subject mean ||z-mu||: "
          f"{subj_dist.mean():.2f} ± {subj_dist.std():.2f} (min {subj_dist.min():.2f}, max {subj_dist.max():.2f})")
    print(f"  eval subject-mean RMS residual: HC {np.nanmean(subject_rms(r)[y==0]):.2f} "
          f"vs SZ {np.nanmean(subject_rms(r)[y==1]):.2f}")

    # -- layout --------------------------------------------------------
    fig = plt.figure(figsize=(13.8, 8.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1],
                          height_ratios=[1, 1.15], hspace=0.32, wspace=0.3,
                          left=0.06, right=0.975, top=0.93, bottom=0.06)

    # (a) joint PCA — training-HC reference density + eval subject means
    ax = fig.add_subplot(gs[0, 0])
    ax.hist2d(proj_ref[:, 0], proj_ref[:, 1], bins=80, cmap="Greys",
              vmax=60, zorder=0)
    ax.scatter(hc[:, 0], hc[:, 1], s=16, color=HC_COLOR, alpha=0.75,
               label=f"eval HC subjects (n={len(hc)})", zorder=2)
    ax.scatter(sz[:, 0], sz[:, 1], s=16, color=SZ_COLOR, alpha=0.75,
               label=f"eval SZ subjects (n={len(sz)})", zorder=2)
    for pts, color in [(hc, HC_COLOR), (sz, SZ_COLOR)]:
        mean = pts.mean(0)
        cov = np.cov(pts.T)
        eigval, eigvec = np.linalg.eigh(cov)
        ang = np.degrees(np.arctan2(eigvec[1, 0], eigvec[0, 0]))
        w, h = 2 * np.sqrt(eigval)
        ax.add_patch(Ellipse(mean, w, h, angle=ang, fill=False, ec=color, lw=2))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("(a) Joint PCA (z − μ_s, per stimulus)", fontsize=9.5)
    ax.legend(fontsize=7.5)

    # (b) HC/SZ separation along PC1
    ax = fig.add_subplot(gs[0, 1])
    for pts, color, lab in [(hc, HC_COLOR, "HC"), (sz, SZ_COLOR, "SZ")]:
        ax.hist(pts[:, 0], bins=24, color=color, alpha=0.55, label=lab,
                density=True)
    ax.set_xlabel("PC1 (subject mean)")
    ax.set_ylabel("density")
    ax.set_title("(b) Subject-mean PC1 distributions", fontsize=9.5)
    ax.legend(fontsize=8)

    # (c) subject-mean RMS residual by stimulus category — HC vs SZ
    ax = fig.add_subplot(gs[0, 2])
    cat_lab = ["social", "natural", "synthetic", "manipulated"]
    cat_mean = {c: np.nanmean(r[:, cats == c], axis=1) for c in cat_lab}
    pos = np.arange(4)
    for gi, (grp, color) in enumerate([(y == 0, HC_COLOR), (y == 1, SZ_COLOR)]):
        vp = ax.violinplot([cat_mean[c][grp] for c in cat_lab],
                           positions=pos + (gi - 0.5) * 0.24, widths=0.22,
                           showmedians=True, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(color)
            b.set_alpha(0.55)
            b.set_edgecolor("none")
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.2)
    ax.set_xticks(pos)
    ax.set_xticklabels(cat_lab, fontsize=8)
    ax.set_ylabel("subject-mean RMS ‖(z−μ)/σ‖")
    ax.set_title("(c) Deviation by stimulus category", fontsize=9.5)
    ax.legend(handles=[Patch(facecolor=HC_COLOR, alpha=0.55, label="HC"),
                       Patch(facecolor=SZ_COLOR, alpha=0.55, label="SZ")],
              fontsize=7.5, loc="upper left")

    # (d) subject x stimulus deviation heatmap (horizontal, wide)
    gs_d = gs[1, 0:2].subgridspec(1, 2, width_ratios=[1, 0.06], wspace=0.03)
    ax = fig.add_subplot(gs_d[0])
    cm = ax.imshow(r_sorted[heat_rows], aspect="auto", cmap="viridis", vmax=6)
    ax.axhline(len(heat_rows) / 2 - 0.5, color="white", lw=2.5)
    ax.text(-4, 14.5, "HC", fontsize=11, fontweight="bold", color=HC_COLOR,
            rotation=90, va="center")
    ax.text(-4, len(heat_rows) - 15.5, "SZ", fontsize=11, fontweight="bold",
            color=SZ_COLOR, rotation=90, va="center")
    sorted_cats = np.sort(cats)
    cpos = np.where(sorted_cats[1:] != sorted_cats[:-1])[0]
    for p in cpos:
        ax.axvline(p + 0.5, color="white", lw=1.2)
    cat_mid = {}
    for c in ["social", "natural", "synthetic", "manipulated"]:
        pos = np.where(np.sort(cats) == c)[0]
        cat_mid[c] = pos.mean()
    ax.set_xticks([cat_mid[c] for c in ["social", "natural", "synthetic", "manipulated"]])
    ax.set_xticklabels(["social", "natural", "synthetic", "manipulated"],
                       rotation=0, fontsize=8)
    ax.set_yticks([])
    ax.set_title("(d) Subject × stimulus deviation (60 evaluation subjects,\n"
                 "top 15 / bottom 15 per group; missing pairs blank)",
                 fontsize=9.5)
    cax = fig.add_subplot(gs_d[1])
    plt.colorbar(cm, cax=cax, label="RMS ‖(z−μ)/σ‖")

    # (e) train-HC distance to the bank — concentration check
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(subj_dist, bins=22, color=LEARNED_COLOR, alpha=0.55, density=True)
    ax.set_xlabel("train-HC mean ‖z − μ‖ per subject")
    ax.set_ylabel("density")
    ax.set_title("(e) Train HC distance to bank", fontsize=9.5)

    fig.suptitle("Learned latent space vs the HC normative bank "
                 "(mlp_deepset EXP-PROP-001, out-of-fold seed 42)", fontsize=11, y=0.985)
    savefig(fig, OUT, "Figure_4")
    pd.DataFrame({"subject_id": lat["subject_id"], "label": y,
                  "pc1_mean": subj_mean[:, 0], "pc2_mean": subj_mean[:, 1]}) \
        .to_csv(TAB / "T05.01_pca.csv", index=False)
    np.save(TAB / "T05.01_deviation_matrix.npy", r_sorted[heat_rows])
    pd.DataFrame([{"category": c, "n_stimuli": int((cats == c).sum()),
                   "hc_mean": float(cat_mean[c][y == 0].mean()),
                   "hc_std": float(cat_mean[c][y == 0].std()),
                   "sz_mean": float(cat_mean[c][y == 1].mean()),
                   "sz_std": float(cat_mean[c][y == 1].std()),
                   "welch_p": float(stats.ttest_ind(cat_mean[c][y == 0],
                                                    cat_mean[c][y == 1],
                                                    equal_var=False).pvalue)}
                  for c in cat_lab]) \
        .to_csv(TAB / "T05.01_category_deviation.csv", index=False)
    pd.DataFrame({"ablation": [MAIN_ABL],
                  "effective_rank": [pr],
                  "train_hc_mean_dist": [subj_dist.mean()],
                  "eval_hc_mean_rms": [np.nanmean(subject_rms(r)[y == 0])],
                  "eval_sz_mean_rms": [np.nanmean(subject_rms(r)[y == 1])]}) \
        .to_csv(TAB / "T05.01_norm_diagnostics.csv", index=False)


if __name__ == "__main__":
    fig_normative_latent()
    print("done Figure_4 (normative latent)")
