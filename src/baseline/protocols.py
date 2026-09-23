"""Official split protocol used by all baselines.

The original EMS protocol: Train_Valid.xlsx assigns the 160 labelled
subjects to 4 folds (Set_0..Set_3). For each fold: train on the other 3
(120 subjects), validate on the fold (40 subjects). The official test labels
are withheld by the authors, so no test metrics can be computed — the
Phase-3 matrix reports validation metrics only.

(The legacy random 120/40 protocol P2 is not part of the current matrix.)
"""
import pandas as pd

from baseline.common import PROCESSED


def load_metadata():
    meta = pd.read_csv(PROCESSED / "metadata.csv", index_col="subject_id")
    return meta


def train_subjects():
    meta = load_metadata()
    return meta[meta.partition == "train"].index.tolist()


def protocol1_folds():
    """Yield (fold_name, train_ids, val_ids) following the official 4 folds."""
    meta = load_metadata()
    meta = meta[meta.partition == "train"]
    folds = {}
    for name in ["Set_0", "Set_1", "Set_2", "Set_3"]:
        folds[name] = meta[meta.official_fold == name].index.tolist()
    for name, val_ids in folds.items():
        train_ids = [s for n, ids in folds.items() if n != name for s in ids]
        yield name, train_ids, val_ids
