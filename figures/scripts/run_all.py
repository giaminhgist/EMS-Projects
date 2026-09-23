#!/usr/bin/env python
"""Entry point — regenerate the presentation figure suite (Figures 1-5).

Usage (from the repo root):
    .venv/bin/python figures/scripts/run_all.py [--groups 01,02,05,06] [--skip-cache]

Groups:
  01 dataset (raw fixations, cleaned; needs figures/cache/fixations_cleaned.pkl)
  02 hand-crafted features
  05 latent distribution (needs cache/latent exports)
  06 importance + XAI (needs cache/latent exports)

Caching:
  - raw fixations are cached by rawdata.py (figures/cache/fixations_cleaned.pkl)
  - latent exports are cached in figures/cache/latent/ by model_utils.export_run
  - both are skipped when present; pass --force-cache to rebuild.
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
PY = sys.executable or str(REPO / ".venv" / "bin" / "python")

# latent exports needed by the remaining figures (05/06 latent + importance)
LATENT_RUNS = [
    ("mlp_deepset", 42, ["Set_0", "Set_1", "Set_2", "Set_3"]),
    ("mlp_attn", 42, ["Set_0", "Set_1", "Set_2", "Set_3"]),
]


def run(cmd):
    print("$", " ".join(map(str, cmd)))
    r = subprocess.run([str(c) for c in cmd], cwd=REPO)
    if r.returncode != 0:
        print(f"FAILED: {cmd}")
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="01,02,05,06",
                    help="comma-separated figure groups to regenerate")
    ap.add_argument("--force-cache", action="store_true",
                    help="rebuild fixation cache even if present")
    args = ap.parse_args()
    groups = set(g.strip() for g in args.groups.split(",") if g.strip())

    if args.force_cache or not (REPO / "figures/cache/fixations_cleaned.pkl").exists():
        run([PY, SCRIPTS / "rawdata.py"])

    if groups & {"05", "06"}:
        for abl, seed, folds in LATENT_RUNS:
            for f in folds:
                out = REPO / "figures/cache/latent" / f"{abl}__seed{seed}__fold{f}.npz"
                if not out.exists():
                    run([PY, SCRIPTS / "model_utils.py", "--ablation", abl,
                         "--seed", str(seed), "--fold", f])

    scripts = {
        "01": "make_dataset_figs.py",
        "02": "make_feature_figs.py",
        "05": "make_latent_figs.py",
        "06": "make_xai_figs.py",
    }
    for g in sorted(groups):
        run([PY, SCRIPTS / scripts[g]])
    print("all figure groups done")


if __name__ == "__main__":
    main()
