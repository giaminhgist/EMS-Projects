"""Figure 5 — feature + stimulus importance.

Model explained: mlp_deepset (EXP-PROP-001, seed 42) — the best-AUC
configuration; hard z_mean (EXP-PROP-004) added for feature-space comparison.
Feature rankings are computed on fold Set_1 val; leave-one-stimulus-out
effects on fold Set_0 val. Attention panels use mlp_attn (EXP-PROP-003) —
the only configuration with attention pooling. Permutation operates on
feature trajectories: the whole (S,) vector of a feature is swapped between
subjects, keeping the stimulus axis intact; the bank and all weights stay
frozen.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FIG, TAB, CACHE, HC_COLOR, SZ_COLOR, LEARNED_COLOR,  # noqa: E402
                    HARD_COLOR, FEATURE_NAMES, FEATURE_GROUPS, GROUP_LABELS,
                    feature_group_of, savefig, image_list, category_of,
                    make_datasets, load_checkpoint_model)
from model_utils import forward_full, batch_tensors  # noqa: E402
from make_latent_figs import load_latent  # noqa: E402

OUT = FIG

LEARNED_ABL = "mlp_deepset"  # EXP-PROP-001 (main)
ATTN_ABL = "mlp_attn"        # EXP-PROP-003 (attention pooling)
HARD_ABL = "z_mean"          # EXP-PROP-004 (hard deviation)


def model_on(ds, model, device="cpu"):
    D, mask = batch_tensors(ds)
    with torch.no_grad():
        fwd = forward_full(model, D.to(device), mask.to(device))
    y = np.array(ds.labels)
    return fwd["prob"].numpy(), y


def auc_of(prob, y):
    return roc_auc_score(y, prob)


# --------------------------------------------------------------------- #
# permutation importance
# --------------------------------------------------------------------- #
def permute_importance(ablation, seed, fold, n_repeats=10, families=False):
    model, _, _ = load_checkpoint_model(ablation, seed, fold)
    train_ds, val_ds, tr, va = make_datasets(ablation, seed, fold)
    D, mask = batch_tensors(val_ds)
    y = np.array(val_ds.labels)
    base_prob, _ = model_on(val_ds, model)
    base_auc = auc_of(base_prob, y)
    units = (list(FEATURE_GROUPS) if families
             else list(range(D.shape[-1])))
    importances = {}
    for u in units:
        cols = ([FEATURE_NAMES.index(f) for f in FEATURE_GROUPS[u]]
                if families else [u])
        aucs = []
        rng = np.random.default_rng(seed * 1000 + (u if isinstance(u, int) else 7))
        for _ in range(n_repeats):
            Dp = D.clone()
            perm = rng.permutation(D.shape[0])
            for c in cols:
                Dp[:, :, c] = D[perm, :, c]
            with torch.no_grad():
                fwd = forward_full(model, Dp, mask)
            aucs.append(auc_of(fwd["prob"].numpy(), y))
        importances[u] = dict(mean=float(base_auc - np.mean(aucs)),
                              std=float(np.std(aucs)))
    return importances, base_auc


def fig_importance():
    """Figure_5: (a) feature-family permutation importance, (b) top-15 features,
    (c) leave-one-stimulus-out AUC drop vs attention, (d) top-30 stimuli by
    attention, (e) attention by category x group."""
    # --- permutation importance (panels a, b) -------------------------
    fam_learned, _ = permute_importance(LEARNED_ABL, 42, "Set_1",
                                        n_repeats=8, families=True)
    fam_hard, _ = permute_importance(HARD_ABL, 42, "Set_1",
                                     n_repeats=8, families=True)
    imp_l, _ = permute_importance(LEARNED_ABL, 42, "Set_1", n_repeats=10)

    # --- stimulus importance (panels c, d, e) -------------------------
    lat = load_latent(ATTN_ABL, 42)
    attn = lat["attn"]          # (160, S) pooled val, out-of-fold models
    y = lat["label"]
    imgs = image_list()
    cats = np.array([category_of(im) for im in imgs])
    cat_color = {c: plt.cm.tab10(i) for i, c in
                 enumerate(["social", "natural", "synthetic", "manipulated"])}
    mean_attn = attn.mean(0)
    model, _, _ = load_checkpoint_model(ATTN_ABL, 42, "Set_0")
    train_ds, val_ds, tr, va = make_datasets(ATTN_ABL, 42, "Set_0")
    D, mask = batch_tensors(val_ds)
    yv = np.array(val_ds.labels)
    base = auc_of(*model_on(val_ds, model))
    dloos = []
    for s in range(100):
        m2 = mask.clone()
        m2[:, s] = 0
        with torch.no_grad():
            fwd = forward_full(model, D, m2)
        dloos.append(base - auc_of(fwd["prob"].numpy(), yv))
    with torch.no_grad():
        f = forward_full(model, D, mask)
    attn0 = f["attn"].numpy().mean(0)
    r = np.corrcoef(attn0, dloos)[0, 1]

    # --- layout ---------------------------------------------------------
    fig = plt.figure(figsize=(13.8, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.35,
                          wspace=0.42, left=0.06, right=0.975, top=0.955,
                          bottom=0.06)

    # (a) family-level permutation importance, learned vs hard
    ax = fig.add_subplot(gs[0, 0])
    groups = list(FEATURE_GROUPS)
    x = np.arange(len(groups))
    for offset, fam, color, lab in [(-0.18, fam_learned, LEARNED_COLOR,
                                     "mlp_deepset (learned)"),
                                    (0.18, fam_hard, HARD_COLOR, "z_mean (hard)")]:
        vals = [fam[g]["mean"] for g in groups]
        errs = [fam[g]["std"] for g in groups]
        ax.bar(x + offset, vals, width=0.36, color=color, yerr=errs,
               capsize=2.5, label=lab)
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("spa_", "").replace("_", "\n")
                        for g in groups], fontsize=8)
    ax.set_ylabel("ΔAUC when feature family is permuted\n(across subjects)")
    ax.set_title("(a) Feature-family permutation importance\n"
                 "(ranked on fold Set_1 val)", fontsize=9.5)
    ax.legend(fontsize=7.5)

    # (b) top-15 features, learned model
    ax = fig.add_subplot(gs[0, 1])
    feats = FEATURE_NAMES
    rows = sorted(imp_l.items(), key=lambda kv: -kv[1]["mean"])[:15]
    names = [feats[i] for i, _ in rows]
    vals = [v["mean"] for _, v in rows]
    errs = [v["std"] for _, v in rows]
    colors = [plt.cm.tab10(list(FEATURE_GROUPS).index(feature_group_of(n)))
              for n in names]
    ax.barh(range(15)[::-1], vals, xerr=errs, color=colors, capsize=2.5)
    ax.set_yticks(range(15)[::-1])
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("ΔAUC (permuted, 10 repeats ± SD)")
    ax.set_title("(b) Top-15 features — mlp_deepset\n(fold Set_1 val, n=40)",
                 fontsize=9.5)

    # (c) leave-one-stimulus-out AUC change vs attention (Set_0 val)
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(attn0, dloos, c=[cat_color[c] for c in cats], s=26, alpha=0.85)
    ax.set_xlabel("mean attention (fold Set_0 model)")
    ax.set_ylabel("AUC drop when stimulus removed")
    ax.set_title(f"(c) Leave-one-stimulus-out vs attention\n"
                 f"(Set_0 val, n=40; Pearson r = {r:.2f})", fontsize=9.5)

    # (d) top-30 stimuli by mean attention, colored by category
    ax = fig.add_subplot(gs[1, 0:2])
    order = np.argsort(-mean_attn)
    top30 = order[:30]
    for i, s in enumerate(top30):
        ax.bar(i, mean_attn[s], color=cat_color[cats[s]])
    ax.set_xticks(range(30))
    ax.set_xticklabels([imgs[s][:6] for s in top30], rotation=90, fontsize=6)
    ax.set_ylabel("mean attention weight")
    ax.set_title("(d) Top-30 stimuli by mean attention\n"
                 "(mlp_attn, out-of-fold)", fontsize=9.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cat_color[c], label=c)
                       for c in ["social", "natural", "synthetic",
                                 "manipulated"]], fontsize=7)

    # (e) attention by category and group
    ax = fig.add_subplot(gs[1, 2])
    cat_names = ["social", "natural", "synthetic", "manipulated"]
    for gi, (lab, color) in enumerate([(0, HC_COLOR), (1, SZ_COLOR)]):
        vals = [attn[y == lab][:, cats == c].mean() for c in cat_names]
        ax.bar(x[:4] + (gi - 0.5) * 0.35, vals, width=0.35, color=color,
               label="HC" if lab == 0 else "SZ")
    ax.set_xticks(x[:4])
    ax.set_xticklabels(cat_names, fontsize=8.5)
    ax.set_ylabel("mean attention")
    ax.set_title("(e) Attention by category × group", fontsize=9.5)
    ax.legend(fontsize=8)

    savefig(fig, OUT, "Figure_5")
    pd.DataFrame({f"{LEARNED_ABL}_{k}": v for k, v in imp_l.items()}).T \
        .to_csv(TAB / "T06.01a_importance_learned.csv")
    pd.DataFrame({f"{abl}_{g}": fam[g] for abl, fam in
                  [(LEARNED_ABL, fam_learned), (HARD_ABL, fam_hard)]
                  for g in groups}).T \
        .to_csv(TAB / "T06.01b_importance_families.csv")
    pd.DataFrame({"stimulus": imgs, "category": cats,
                  "mean_attn_pooled": mean_attn, "loo_auc_drop_set0": dloos}) \
        .to_csv(TAB / "T06.01_stimulus_importance.csv", index=False)


if __name__ == "__main__":
    fig_importance()
    print("done Figure_5 (importance)")
