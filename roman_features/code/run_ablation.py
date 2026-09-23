"""Runner for the roman_features/ isolated feature ablation (A0-A4, seed 2026).

Reuses, unchanged: trainer.trainer.train_model, proposal.model.NormativeModel,
metrics.summarize, data.common.official_folds, the EXP-PROP-001 model
configuration (learned deviation, mlp comparator, deepset pooling) and the
locked PROPOSAL_HP hyperparameters. The ONLY trainer-side change is the
AblationRunConfig subclass, which redirects cfg.run_dir() from outputs/ into
roman_features/results/ (or results/_smoke/ for smoke runs) — nothing is
ever written to outputs/ or processed_dataset/.

The environment reproduces the original experiment runner's subprocess env
(OMP/MKL/OPENBLAS threads = 1) so the A0 run is directly comparable to
outputs/experiments/EXP-PROP-001/full45/seed2026.

Usage (from repo root):
    .venv/bin/python roman_features/code/run_ablation.py \
        --ablation-id A0_base45 --feature-set base45 --seed 2026 --fold all
    # smoke: 2 epochs, patience 5, one fold, written under results/_smoke/
    .venv/bin/python roman_features/code/run_ablation.py --ablation-id A0_base45 \
        --feature-set base45 --seed 2026 --fold Set_0 --smoke --epochs 2 --patience 5
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Reproduce the original run_experiments.py subprocess environment: the
# original 10-seed suite ran with single-threaded BLAS, which affects float
# reduction order and therefore numeric reproducibility.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

_ROMAN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROMAN.parent / "src"))      # EMS-Projects/src
sys.path.insert(0, str(_ROMAN / "code"))

from data.common import official_folds                # noqa: E402
from trainer.config import RunConfig                  # noqa: E402
from trainer.trainer import train_model               # noqa: E402
from metrics import summarize                         # noqa: E402
from proposal.model import NormativeModel             # noqa: E402
from dataset import (ExtendedNormativeDataset,        # noqa: E402
                     extended_feature_names)

RESULTS = _ROMAN / "results"
MATRIX = json.loads((_ROMAN / "configs" / "ablation_matrix.json").read_text())


def _base45_cols():
    """Authoritative base45 column order (processed feature_names.txt)."""
    names = [ln.strip() for ln in
             (_ROMAN.parent / "processed_dataset" / "feature_names.txt")
             .read_text().splitlines() if ln.strip()]
    return names


def feature_cols(feature_set):
    """Ordered column list for a named feature set (see ablation_matrix.json)."""
    ext = extended_feature_names()
    base = _base45_cols()
    assert ext[:len(base)] == base, "cache base45 order differs from processed schema"
    freq20 = ext[len(base):len(base) + 18]
    freq10 = ext[len(base) + 18:len(base) + 18 + 36]
    if feature_set == "base45":
        return base
    if feature_set == "base45_freq20":
        return base + freq20
    if feature_set == "base45_k":
        return base + ["att_k_mean"]
    if feature_set == "base45_freq20_k":
        return base + freq20 + ["att_k_mean"]
    if feature_set == "base45_freq10":
        return base + freq10
    if feature_set == "base45_freq10_k":
        return base + freq10 + ["att_k_mean"]
    raise KeyError(f"unknown feature set {feature_set}")


@dataclass
class AblationRunConfig(RunConfig):
    """RunConfig whose run_dir() points inside roman_features/results/.

    The trainer calls cfg.run_dir(create=True) and cfg.save(run_dir) — this
    subclass redirects both to the isolation folder. `smoke` routes the run
    under results/_smoke/ so no smoke artifact can be mistaken for a result.
    """

    smoke: bool = False


    def run_dir(self, create=True):
        root = RESULTS / "_smoke" if self.smoke else RESULTS
        d = root / self.experiment / f"seed{self.seed}" / f"fold{self.fold}"
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d


def _log(fp, msg):
    print(msg, flush=True)
    fp.write(msg + "\n")
    fp.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-id", required=True, help="e.g. A0_base45")
    ap.add_argument("--feature-set", required=True,
                    help="e.g. base45, base45_freq20, base45_k, base45_freq20_k, base45_freq10")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fold", default="all", help="all | Set_0..Set_3")
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke)")
    ap.add_argument("--patience", type=int, default=None, help="override (smoke)")
    ap.add_argument("--smoke", action="store_true",
                    help="write under results/_smoke/ (short runs only)")
    args = ap.parse_args()

    run_def = next(r for r in MATRIX["runs"]
                   if r["ablation_id"] == args.ablation_id)
    assert run_def["feature_set"] == args.feature_set, \
        f"{args.ablation_id} is defined for {run_def['feature_set']}"
    hp = dict(MATRIX["hyperparameters"])
    if args.epochs is not None:
        hp["epochs"] = args.epochs
    if args.patience is not None:
        hp["patience"] = args.patience
    model_cfg = dict(MATRIX["model"])

    cols = feature_cols(args.feature_set)
    input_dim = len(cols)
    assert input_dim == run_def["input_dim"], \
        f"input dim {input_dim} != matrix {run_def['input_dim']}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = AblationRunConfig(
        experiment=args.ablation_id, feature_set=args.feature_set,
        ablation="mlp_deepset", seed=args.seed,
        epochs=hp["epochs"], patience=hp["patience"], lr=hp["lr"],
        weight_decay=hp["weight_decay"], batch_size=hp["batch_size"],
        dropout=hp["dropout"],
        extra={**model_cfg, "input_dim": input_dim, "smoke": args.smoke,
               "role": run_def["role"], "hypothesis": run_def["hypothesis"]},
        smoke=args.smoke)

    folds = list(official_folds()) if args.fold == "all" else [
        (args.fold, *next(f for f in official_folds() if f[0] == args.fold)[1:])]
    seed_dir = cfg.run_dir(create=True).parent
    logf = open(seed_dir / "run.log", "a")
    _log(logf, f"=== roman_features ablation run {args.ablation_id} "
               f"(feature_set={args.feature_set}, dim={input_dim}, seed={args.seed}, "
               f"device={device}, folds={[f[0] for f in folds]}) ===")
    _log(logf, f"hp={hp} model={model_cfg} smoke={args.smoke}")

    # resolved config + exact ordered feature names (once per seed run)
    resolved = {
        "ablation_id": args.ablation_id, "feature_set": args.feature_set,
        "input_dim": input_dim, "seed": args.seed, "fold": args.fold,
        "model": model_cfg, "hyperparameters": hp, "device": device,
        "role": run_def["role"], "hypothesis": run_def["hypothesis"],
        "smoke": args.smoke, "fold_names": [f[0] for f in folds],
        "feature_names": cols,
    }
    (seed_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2))
    (seed_dir / "feature_names.txt").write_text("\n".join(cols) + "\n")

    fold_metrics = []
    for fold_name, train_ids, val_ids in folds:
        cfg.fold = fold_name
        train_ds = ExtendedNormativeDataset(train_ids, train_ids,
                                            feature_cols=cols)
        val_ds = ExtendedNormativeDataset(val_ids, train_ids,
                                          feature_cols=cols)
        factory = (lambda: NormativeModel(
            d_in=input_dim, comparator=model_cfg["comparator"],
            pool=model_cfg["pool"], dropout=cfg.dropout,
            deviation=model_cfg["deviation"], n_stim=model_cfg["n_stim"]))
        summary = train_model(factory, train_ds, val_ds, cfg, device)
        # per-fold copy of the exact feature list used
        (cfg.run_dir(create=False) / "feature_names.txt").write_text(
            "\n".join(cols) + "\n")
        fold_metrics.append({**summary["best_val_metrics"], "fold": fold_name,
                             "best_epoch": summary["best_epoch"],
                             "epochs_trained": summary["epochs_trained"],
                             "stopping_reason": summary["stopping_reason"]})
        _log(logf, f"{fold_name}: AUC={summary['best_val_metrics']['auc']:.4f} "
                   f"Acc={summary['best_val_metrics']['acc']:.4f} "
                   f"Sen={summary['best_val_metrics']['sen']:.4f} "
                   f"Spec={summary['best_val_metrics']['spec']:.4f} "
                   f"F1={summary['best_val_metrics']['f1']:.4f} "
                   f"(best epoch {summary['best_epoch']}, "
                   f"{summary['epochs_trained']} epochs, "
                   f"{summary['stopping_reason']})")

    agg = summarize(fold_metrics)
    out = {"experiment": args.ablation_id, "ablation": "mlp_deepset",
           "feature_set": args.feature_set, "input_dim": input_dim,
           "seed": args.seed, "folds": [m["fold"] for m in fold_metrics],
           "fold_metrics": fold_metrics, "mean_metrics": agg}
    if len(folds) > 1:
        (seed_dir / "folds_summary.json").write_text(
            json.dumps(out, indent=2, default=str))
    _log(logf, json.dumps(agg, indent=1))
    logf.close()


if __name__ == "__main__":
    main()
