"""Consolidated 10-seed summary for the roman_features/ isolated ablation.

Convention (identical to the original suite, EXPERIMENTS_RESULTS.md):
  - per seed = mean of the 4 fold metrics;
  - reported as mean +/- sample SD (ddof=1) over the 10 seed means
    (the locked src/registry.SEEDS);
  - deltas are PAIRED per seed (all runs share the same seeds and folds),
    reported as mean +/- SD of the per-seed deltas.

Writes (only under roman_features/results/):
  - summary.csv            one row per ablation: 10-seed aggregates
  - per_seed_metrics.csv   ablation x seed: per-seed fold-mean metrics
  - deltas.csv             all required pairwise deltas (paired per seed)
  - summary.md             consolidated report (primary vs sensitivity)

Usage (from repo root):
    .venv/bin/python roman_features/code/summarize.py
    .venv/bin/python roman_features/code/summarize.py --seeds 2026   # single seed
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROMAN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROMAN.parent / "src"))
sys.path.insert(0, str(_ROMAN / "code"))

from metrics import mean_std, METRIC_KEYS  # noqa: E402
from registry import SEEDS                 # noqa: E402
from dataset import load_extended_features  # noqa: E402
from run_ablation import feature_cols, MATRIX, RESULTS  # noqa: E402

PRIMARY = {r["ablation_id"] for r in MATRIX["runs"] if r.get("primary")}
DELTAS = [("A1_freq20", "A0_base45"), ("A2_k", "A0_base45"),
          ("A3_freq20_k", "A1_freq20"), ("A3_freq20_k", "A2_k"),
          ("A3_freq20_k", "A0_base45"), ("A4_freq10", "A1_freq20")]
METRIC_DISPLAY = METRIC_KEYS + ["pre"]


def fmt(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def missingness(feature_set):
    ext = load_extended_features()
    cols = feature_cols(feature_set)
    sub = ext[cols]
    masked = int(sub.isna().all(axis=1).sum())
    n_nan = int(sub.isna().sum().sum())
    return {"n_rows": len(sub), "n_masked_rows": masked,
            "n_cells": sub.shape[0] * sub.shape[1], "n_nan_cells": n_nan,
            "nan_frac": n_nan / max(sub.shape[0] * sub.shape[1], 1)}


def collect(seeds):
    """Return (rows, per_seed) with 10-seed aggregates and per-seed metrics."""
    run_defs = [r for r in MATRIX["runs"] if r.get("run", True)]
    per_seed = []          # one row per (ablation, seed)
    for r in run_defs:
        for seed in seeds:
            p = RESULTS / r["ablation_id"] / f"seed{seed}" / "folds_summary.json"
            rec = {"ablation_id": r["ablation_id"], "feature_set": r["feature_set"],
                   "input_dim": r["input_dim"], "seed": seed}
            if not p.exists():
                rec.update({"status": "missing"})
                per_seed.append(rec)
                continue
            d = json.loads(p.read_text())
            agg = dict(mean_std(d["fold_metrics"], keys=METRIC_DISPLAY))
            rec.update({"status": "ok",
                        "best_epochs": [m["best_epoch"] for m in d["fold_metrics"]],
                        "epochs_trained": [m["epochs_trained"] for m in d["fold_metrics"]],
                        "fold_metrics": d["fold_metrics"]})
            for k in METRIC_DISPLAY:
                rec[f"{k}_mean"], rec[f"{k}_sd"] = agg[k]
            per_seed.append(rec)

    ok_seed = [r for r in per_seed if r["status"] == "ok"]
    by_key = {(r["ablation_id"], r["seed"]): r for r in ok_seed}
    rows = []
    for r in run_defs:
        sub = [s for s in ok_seed if s["ablation_id"] == r["ablation_id"]]
        row = {"ablation_id": r["ablation_id"], "feature_set": r["feature_set"],
               "input_dim": r["input_dim"], "n_seeds": len(sub),
               "status": "ok" if sub else "missing"}
        if not sub:
            rows.append(row)
            continue
        for k in METRIC_DISPLAY:
            vals = [s[f"{k}_mean"] for s in sub]
            row[f"{k}_mean"] = float(np.mean(vals))
            row[f"{k}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        row["seed_auc_means"] = {s["seed"]: s["auc_mean"] for s in sub}
        row.update(missingness(r["feature_set"]))
        rows.append(row)
    return rows, per_seed, by_key


def paired_deltas(per_seed_ok):
    """Paired per-seed deltas: mean +/- SD (ddof=1) over seeds."""
    by_key = {(r["ablation_id"], r["seed"]): r for r in per_seed_ok}
    seeds = sorted({s for _, s in by_key})
    out = []
    for hi, lo in DELTAS:
        rec = {"comparison": f"{hi} - {lo}", "n_seeds": 0}
        per = []
        for s in seeds:
            if (hi, s) not in by_key or (lo, s) not in by_key:
                continue
            per.append({k: by_key[(hi, s)][f"{k}_mean"] - by_key[(lo, s)][f"{k}_mean"]
                        for k in METRIC_DISPLAY})
        rec["n_seeds"] = len(per)
        for k in METRIC_DISPLAY:
            vals = [p[k] for p in per]
            rec[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
            rec[f"{k}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        out.append(rec)
    return out


def a0_repro(seeds):
    """Per-seed A0 comparison against the original suite outputs."""
    out = []
    for seed in seeds:
        p = RESULTS / "A0_base45" / f"seed{seed}" / "folds_summary.json"
        ref = (_ROMAN.parent / "outputs" / "experiments" / "EXP-PROP-001" /
               "full45" / f"seed{seed}" / "folds_summary.json")
        if not p.exists() or not ref.exists():
            out.append({"seed": seed, "status": "missing"})
            continue
        mine = json.loads(p.read_text())
        orig = json.loads(ref.read_text())
        same = orig["fold_metrics"] == mine["fold_metrics"]
        out.append({"seed": seed, "status": "EXACT" if same else "MISMATCH"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="subset of the locked seeds (default: all 10)")
    args = ap.parse_args()
    seeds = args.seeds if args.seeds is not None else list(SEEDS)
    rows, per_seed, _ = collect(seeds)
    ok = [r for r in rows if r["status"] == "ok"]
    by_id = {r["ablation_id"]: r for r in ok}
    a0 = by_id.get("A0_base45")
    ok_seed = [r for r in per_seed if r["status"] == "ok"]
    deltas = paired_deltas(ok_seed)
    repro = a0_repro(seeds)

    # ---- summary.csv (one row per ablation, 10-seed aggregates) ----
    csv_cols = ["ablation_id", "feature_set", "input_dim", "n_seeds",
                "auc_mean", "auc_sd", "acc_mean", "acc_sd",
                "balanced_acc_mean", "balanced_acc_sd", "sen_mean", "sen_sd",
                "spec_mean", "spec_sd", "pre_mean", "pre_sd", "f1_mean", "f1_sd",
                "n_rows", "n_masked_rows", "n_cells", "n_nan_cells", "nan_frac"]
    pd.DataFrame(rows)[csv_cols].to_csv(RESULTS / "summary.csv", index=False)

    # ---- per_seed_metrics.csv ----
    pd.DataFrame([{k: v for k, v in r.items() if k != "fold_metrics"}
                  for r in per_seed]).to_csv(RESULTS / "per_seed_metrics.csv",
                                             index=False)

    # ---- deltas.csv (paired per-seed deltas) ----
    pd.DataFrame(deltas).to_csv(RESULTS / "deltas.csv", index=False)

    # ---- summary.md ----
    L = []
    L.append("# roman_features/ — direction-frequency & K-index ablation "
             "(10 seeds)")
    L.append("")
    L.append(f"Locked seeds {SEEDS}, official 4-fold protocol (Set_0..Set_3), "
             "EXP-PROP-001 model configuration (learned deviation, mlp "
             "comparator, deepset pooling), identical hyperparameters across "
             "all runs (epochs 150, patience 30, AdamW lr 1e-3, wd 1e-4, "
             "batch 16, dropout 0.3, threshold 0.5). Per seed = mean of the 4 "
             "fold metrics; reported as mean ± sample SD (ddof=1) over the "
             "seed means. Deltas are paired per seed. Best checkpoint = "
             "highest outer-validation AUC (earliest epoch wins ties). "
             "CPU-only, OMP/MKL/OPENBLAS=1.")
    L.append("")
    L.append("## 1. Primary factorial ablation (A0-A3)")
    L.append("")
    L.append("| run | feature set | dim | AUC | ACC | balACC | SEN | SPEC | PRE | F1 | ΔAUC vs A0 | ΔbalACC vs A0 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(ok, key=lambda x: x["ablation_id"]):
        if r["ablation_id"] not in PRIMARY:
            continue
        d_auc = (r["auc_mean"] - a0["auc_mean"]) if a0 is not None else None
        d_bal = (r["balanced_acc_mean"] - a0["balanced_acc_mean"]) if a0 is not None else None
        L.append(f"| {r['ablation_id']} | {r['feature_set']} | {r['input_dim']} | "
                 f"{fmt(r['auc_mean'])} ± {fmt(r['auc_sd'])} | {fmt(r['acc_mean'])} | "
                 f"{fmt(r['balanced_acc_mean'])} | {fmt(r['sen_mean'])} | "
                 f"{fmt(r['spec_mean'])} | {fmt(r['pre_mean'])} | {fmt(r['f1_mean'])} | "
                 f"{fmt(d_auc)} | {fmt(d_bal)} |")
    L.append("")
    L.append("### Pairwise deltas (paired per seed, all required comparisons)")
    L.append("")
    L.append("| comparison | ΔAUC | ΔACC | ΔbalACC | ΔSEN | ΔSPEC | ΔPRE | ΔF1 | n seeds |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for d in deltas:
        L.append(f"| {d['comparison']} | {fmt(d['auc_mean'])} ± {fmt(d['auc_sd'])} | "
                 f"{fmt(d['acc_mean'])} | {fmt(d['balanced_acc_mean'])} | "
                 f"{fmt(d['sen_mean'])} | {fmt(d['spec_mean'])} | {fmt(d['pre_mean'])} | "
                 f"{fmt(d['f1_mean'])} | {d['n_seeds']} |")
    L.append("")
    L.append("## 2. Sensitivity: 10-degree vs 20-degree direction bins")
    L.append("")
    d41 = next((d for d in deltas if d["comparison"] == "A4_freq10 - A1_freq20"), None)
    if d41 is not None:
        L.append(f"A4_freq10 - A1_freq20 (paired per seed): ΔAUC = "
                 f"{fmt(d41['auc_mean'])} ± {fmt(d41['auc_sd'])} pp, "
                 f"ΔbalACC = {fmt(d41['balanced_acc_mean'])} ± "
                 f"{fmt(d41['balanced_acc_sd'])} pp.")
    L.append("")
    L.append("## 3. Per-seed fold-mean AUC")
    L.append("")
    seed_cols = sorted({s["seed"] for s in ok_seed})
    head = "| run " + "".join(f"| {s} " for s in seed_cols) + "|"
    L.append(head)
    L.append("|" + "---|" * (len(seed_cols) + 1))
    for r in sorted(ok, key=lambda x: x["ablation_id"]):
        m = r["seed_auc_means"]
        L.append("| " + r["ablation_id"] +
                 "".join(f"| {fmt(m.get(s))} " for s in seed_cols) + "|")
    L.append("")
    L.append("## 4. Missingness and valid feature values")
    L.append("")
    L.append("| run | dim | rows | masked rows | cells | NaN cells | NaN % |")
    L.append("|---|---|---|---|---|---|---|")
    for r in ok:
        L.append(f"| {r['ablation_id']} | {r['input_dim']} | {r['n_rows']} | "
                 f"{r['n_masked_rows']} | {r['n_cells']} | {r['n_nan_cells']} | "
                 f"{r['nan_frac'] * 100:.2f} |")
    L.append("")
    L.append("## 5. A0 reproduction check vs the original suite (all seeds)")
    L.append("")
    L.append("| seed | status |")
    L.append("|---|---|")
    for r in repro:
        L.append(f"| {r['seed']} | {r['status']} |")
    n_exact = sum(1 for r in repro if r["status"] == "EXACT")
    L.append("")
    L.append(f"A0 reproduces the original EXP-PROP-001/full45 runs exactly in "
             f"{n_exact}/{len(repro)} seeds (fold metrics, best epochs, epochs "
             "trained identical).")
    L.append("")
    L.append("Generated by roman_features/code/summarize.py")
    (RESULTS / "summary.md").write_text("\n".join(L) + "\n")
    print(f"wrote summary.csv, per_seed_metrics.csv, deltas.csv, summary.md "
          f"({len(ok)} runs ok, {len(ok_seed)} run-seeds, {len(deltas)} deltas)")


if __name__ == "__main__":
    main()
