"""Common constants and data-loading utilities for the EMS-Projects project."""
from pathlib import Path
import glob
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # repo root, portable across machines
RAW = ROOT / "original_dataset" / "EMS"
TRAIN_FIX_DIR = RAW / "Train_Valid" / "Fixations"
TEST_FIX_DIR = RAW / "Test" / "Fixations"
IMAGES_DIR = RAW / "Images"
PROCESSED = ROOT / "processed_dataset"
DOCS_EDA = ROOT / "docs" / "EDA"

SCREEN_W, SCREEN_H = 1024, 768  # Eyelink display resolution used in EMS

# stimulus name -> category (derived from the official Images/ folder layout)
CATEGORY_PREFIXES = {
    "act_": "social", "por_": "social", "soc_": "social",
    "ind_": "natural", "land_": "natural", "outman_": "natural", "sat_": "natural",
    "art_": "synthetic", "cat_": "synthetic", "pat_": "synthetic",
    "low_": "manipulated", "mood_": "manipulated", "noi_": "manipulated",
    "patch_": "manipulated", "rand_": "manipulated",
}


def image_category(name: str) -> str:
    """Map an image file name (with or without .jpg) to its stimulus category."""
    stem = name.replace(".jpg", "").replace(".jpeg", "")
    for prefix, cat in CATEGORY_PREFIXES.items():
        if stem.startswith(prefix):
            return cat
    raise KeyError(f"unknown stimulus prefix in {name}")


def image_categories() -> pd.Series:
    """Series mapping every stimulus image name -> category (100 images)."""
    names = sorted(p.name for p in IMAGES_DIR.glob("*/*.jpg"))
    return pd.Series({n: image_category(n) for n in names}).sort_index()


def subject_label(subject_id: int) -> int:
    """EMS convention: ids < 200 are HC (label 0), ids >= 200 are SZ (label 1)."""
    return 0 if int(subject_id) < 200 else 1


def load_subject(subject_id, partition="train"):
    """Load one subject's fixation xlsx into a DataFrame with extra columns."""
    fix_dir = TRAIN_FIX_DIR if partition == "train" else TEST_FIX_DIR
    fname = f"{int(subject_id):03d}.xlsx" if partition == "train" else f"Test_{subject_id}.xlsx"
    df = pd.read_excel(fix_dir / fname)
    df["subject_id"] = subject_id
    df["category"] = df["IMAGE"].map(image_category)
    return df


def load_all(partition="train", subjects=None):
    """Load all subject files of a partition into one long DataFrame.

    partition: 'train' -> 160 Train_Valid subjects, 'test' -> 48 official test subjects.
    subjects: optional list of subject ids to restrict loading.
    """
    fix_dir = TRAIN_FIX_DIR if partition == "train" else TEST_FIX_DIR
    files = sorted(fix_dir.glob("*.xlsx"))
    if subjects is not None:
        files = [f for f in files if int(f.stem.split("_")[-1]) in set(subjects)]
    frames = []
    for f in files:
        sid = int(f.stem.split("_")[-1])
        df = pd.read_excel(f)
        df["subject_id"] = sid
        df["category"] = df["IMAGE"].map(image_category)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def official_folds() -> pd.DataFrame:
    """Official 4-fold assignment (Train_Valid.xlsx): Set_0 .. Set_3, 40 subjects each."""
    df = pd.read_excel(RAW / "Train_Valid.xlsx")
    folds = {}
    for col in df.columns:
        folds[col] = [int(x) for x in df[col].dropna().tolist()]
    return folds


def train_subject_ids():
    return sorted(int(f.stem) for f in TRAIN_FIX_DIR.glob("*.xlsx"))


def test_subject_ids():
    return sorted(int(f.stem.split("_")[-1]) for f in TEST_FIX_DIR.glob("*.xlsx"))


def subject_labels(subject_ids=None):
    """Series subject_id -> label for train/valid subjects (ids < 200 = HC)."""
    ids = train_subject_ids() if subject_ids is None else subject_ids
    return pd.Series({s: subject_label(s) for s in ids}, name="label").sort_index()
