"""Shared paths and utilities for the baseline suite."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root, portable across machines
PROCESSED = ROOT / "processed_dataset"
RESULTS = ROOT / "outputs" / "experiments"  # run outputs
NUM_STIMULI = 100
CATEGORIES = ["social", "natural", "synthetic", "manipulated"]

# stimulus filename prefix -> category (same mapping as EMS-Projects/src/common.py)
PREFIX_CATEGORY = {
    "act": "social", "por": "social", "soc": "social",
    "ind": "natural", "land": "natural", "outman": "natural", "sat": "natural",
    "art": "synthetic", "cat": "synthetic", "pat": "synthetic",
    "low": "manipulated", "mood": "manipulated", "noi": "manipulated",
    "patch": "manipulated", "rand": "manipulated",
}


def image_category(name):
    return PREFIX_CATEGORY[name.replace(".jpg", "").split("_")[0]]
