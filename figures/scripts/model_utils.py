"""Inference utilities: run trained checkpoints, capture z / d / h / attention.

All exports preserve the checkpoint's bank buffers (never refreshed), so
predictions and intermediate tensors correspond to the committed runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ABLATION_META, CACHE, FOLDS, make_datasets,  # noqa: E402
                    load_checkpoint_model, pick_run_dir)

EPS = 1e-6


def batch_tensors(ds, indices=None):
    """Stack (D, mask) of a NormativeDataset into (B,S,D), (B,S)."""
    if indices is None:
        indices = list(range(len(ds)))
    Ds, Ms = [], []
    for i in indices:
        (X, mask), y, sid = ds[i]
        Ds.append(X)
        Ms.append(mask)
    return torch.stack(Ds), torch.stack(Ms)


def forward_full(model, D, mask):
    """Full forward with intermediate tensors captured.

    Returns dict(z, d, h, prob, attn) — z/attn only exist for learned/attention.
    """
    out = {}
    deviation = model.deviation
    if deviation == "learned":
        z, _ = model._encode(D, mask)
        out["z"] = z
        d = model._deviate(z, mask)
    else:
        d = D
    d = d * mask[..., None]
    out["d"] = d
    if model.pool == "attention":
        score = model.scorer(d) + (mask - 1)[..., None] * 1e9
        a = torch.softmax(score, dim=1).squeeze(-1)   # (B, S)
        out["attn"] = a
    h = model._pool(d, mask)
    out["h"] = h
    out["prob"] = torch.sigmoid(model.head(h)).view(-1)
    return out


def export_run(ablation, seed, fold, device="cpu"):
    """Export z/d/h/prob/attn for the val fold + re-encoded train-HC reference.

    Returns the npz path.
    """
    meta = ABLATION_META[ablation]
    out_path = CACHE / "latent" / f"{ablation}__seed{seed}__fold{fold}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path

    model, run_dir, cfg = load_checkpoint_model(ablation, seed, fold, device)
    train_ds, val_ds, train_ids, val_ids = make_datasets(ablation, seed, fold)

    # ---- validation fold ----
    D, mask = batch_tensors(val_ds)
    D, mask = D.to(device), mask.to(device)
    with torch.no_grad():
        fwd = forward_full(model, D, mask)
    labels = np.array(val_ds.labels, dtype=np.int64)
    sids = np.array(val_ds.subjects, dtype=np.int64)

    npz = {
        "subject_id": sids,
        "label": labels,
        "mask": mask.cpu().numpy().astype(bool),
        "h": fwd["h"].cpu().numpy(),
        "d": fwd["d"].cpu().numpy(),
        "prob": fwd["prob"].cpu().numpy(),
        "deviation": meta["deviation"],
        "ablation": ablation,
        "seed": seed,
        "fold": fold,
        "run_dir": str(run_dir),
    }
    if "z" in fwd:
        npz["z"] = fwd["z"].cpu().numpy()
    if "attn" in fwd:
        npz["attn"] = fwd["attn"].cpu().numpy()
    if meta["deviation"] == "learned":
        npz["bank_mu"] = model.bank_mu.cpu().numpy()
        npz["bank_sigma"] = model.bank_sigma.cpu().numpy()
        # re-encoded training-HC empirical reference (final encoder weights)
        hc_idx = train_ds.hc_indices()
        Dref, mref = batch_tensors(train_ds, hc_idx)
        Dref, mref = Dref.to(device), mref.to(device)
        with torch.no_grad():
            z_ref, _ = model._encode(Dref, mref)
        npz["ref_subject_id"] = np.array([train_ds.subjects[i] for i in hc_idx])
        npz["ref_z"] = z_ref.cpu().numpy()
        npz["ref_mask"] = mref.cpu().numpy().astype(bool)
    np.savez_compressed(out_path, **npz)
    print(f"  exported {out_path.name}  (val n={len(sids)})")
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", default="mlp_deepset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", default="all", help="Set_0..3 or 'all'")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    folds = FOLDS if args.fold == "all" else [args.fold]
    for f in folds:
        export_run(args.ablation, args.seed, f, args.device)
