"""Generate figure_manifest.csv and the contact-sheet gallery for Figures 1-5."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PRES, REPO  # noqa: E402

# figure file -> metadata
META = [
    # (file, group, title, science question, script, sources,
    #  protocol, seed/fold, sample note)
    ("Figure_1", "01_dataset", "Dataset overview: subjects, stimuli, official folds",
     "Who is in the dataset, what do they view, how are folds composed?",
     "make_dataset_figs.py", "metadata.csv; Images/*; fixations cache",
     "n/a", "n/a", "160 subjects / 100 stimuli"),
    ("Figure_2", "01_dataset", "Gaze signatures: subject-level distributions + scanpaths",
     "How do groups differ in gaze statistics, and what do their scanpaths look like?",
     "make_dataset_figs.py", "fixations cache (cleaned, train); Images/*",
     "n/a", "n/a", "n=80 HC / 80 SZ subjects; 4 stimuli x 2 subjects"),
    ("Figure_3", "02_features", "HC-SZ effect sizes for all 45 features + rainclouds + category profiles",
     "Which features discriminate groups, and how does the signal depend on category?",
     "make_feature_figs.py", "stimulus_features_train.pkl",
     "n/a", "n/a", "n=80/80 subjects; 4 features x 4 categories"),
    ("Figure_4", "05_latent_distribution", "Learned latent space vs the HC normative bank (PCA + heatmap + rank/distance diagnostics)",
     "Where do SZ live relative to the HC norm, and is the norm stable?",
     "make_latent_figs.py", "cache/latent mlp_deepset seed42 (z, bank, ref_z)",
     "P1 out-of-fold", "seed 42", "160 eval + train-HC ref subjects x 100 stimuli"),
    ("Figure_5", "06_importance_xai", "Feature + stimulus importance (permutation, attention, leave-one-out)",
     "Which features and stimuli drive the prediction?",
     "make_xai_figs.py", "checkpoints mlp_deepset + z_mean seed42 (permutation on Set_1 val); mlp_attn exports; LOO on Set_0 val model",
     "P1 fold Set_1 val / Set_0 val", "seed 42", "40 val subjects, 8-10 repeats"),
]


def build_manifest():
    rows = []
    for name, grp, title, question, script, sources, proto, seed, n in META:
        rows.append({
            "figure": name,
            "group": grp,
            "title": title,
            "science_question": question,
            "script": f"figures/scripts/{script}",
            "png": f"figures/figure/{name}.png",
            "svg": f"figures/figure/{name}.svg",
            "source_data": sources,
            "protocol": proto,
            "seed_fold": seed,
            "sample_note": n,
        })
    df = pd.DataFrame(rows)
    # verify files exist on disk (paths are repo-relative)
    df["png"] = [p if (REPO / p).exists() else "MISSING" for p in df["png"]]
    df.to_csv(PRES / "figure_manifest.csv", index=False)
    missing = df[df.png == "MISSING"]
    if len(missing):
        print("WARNING missing figures:\n", missing[["figure", "title"]])
    print(f"figure_manifest.csv: {len(df)} figures")
    return df


def build_gallery():
    df = build_manifest()
    n = len(df)
    cols = 3
    rows_n = int(np.ceil(n / cols))
    with PdfPages(PRES / "figures_gallery.pdf") as pdf:
        fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 5.2, rows_n * 3.6))
        axes = np.atleast_2d(axes)
        for i, row in enumerate(df.iterrows()):
            r = row[1]
            ax = axes[i // cols, i % cols]
            ax.axis("off")
            if r.png != "MISSING":
                img = plt.imread(REPO / r.png)
                ax.imshow(img)
            ax.set_title(f"{r.figure} — {r.group}\n{r.title}",
                         fontsize=7.5, pad=4)
        for j in range(n, rows_n * cols):
            axes[j // cols, j % cols].axis("off")
        fig.suptitle("EMS-Projects presentation figures — contact sheet", fontsize=14, y=0.995)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    print("figures_gallery.pdf written")


if __name__ == "__main__":
    build_manifest()
    build_gallery()
