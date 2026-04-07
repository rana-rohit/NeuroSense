"""
splits.py
Responsible for: Generating and saving reproducible train/val/test splits.
Supports: random split, subject-stratified split, LOSO split generation.
"""

import os
import json
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split

from src.data.loader import N_SUBJECTS, N_VIDEOS
from src.utils.logger import get_logger

logger = get_logger("splits")


# ── Random window-level split ─────────────────────────────────────────────────

def random_split_indices(
    n_samples   : int,
    val_ratio   : float = 0.15,
    test_ratio  : float = 0.15,
    seed        : int   = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Random train / val / test index split (window-level).

    Returns:
        train_idx, val_idx, test_idx
    """
    idx = np.arange(n_samples)
    train_idx, tmp_idx = train_test_split(
        idx, test_size=(val_ratio + test_ratio), random_state=seed
    )
    ratio_adjusted = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=ratio_adjusted, random_state=seed
    )
    logger.info(
        f"Random split | train={len(train_idx)} "
        f"val={len(val_idx)} test={len(test_idx)}"
    )
    return train_idx, val_idx, test_idx


# ── Subject-stratified split ──────────────────────────────────────────────────

def subject_split(
    subject_ids  : np.ndarray,
    val_subjects : Optional[List[int]] = None,
    test_subjects: Optional[List[int]] = None,
    val_ratio    : float = 0.15,
    test_ratio   : float = 0.15,
    seed         : int   = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split indices so that no subject appears in more than one split.
    Prevents data leakage across subjects.

    Args:
        subject_ids  : (n_samples,) array of subject IDs per window
        val_subjects : explicit list of 1-indexed subject IDs for val
        test_subjects: explicit list of 1-indexed subject IDs for test
        val_ratio    : fraction of subjects for val (if not explicit)
        test_ratio   : fraction of subjects for test (if not explicit)
        seed         : random seed

    Returns:
        train_idx, val_idx, test_idx  (window-level indices)
    """
    unique_subs = np.unique(subject_ids)
    rng = np.random.default_rng(seed)

    if test_subjects is None:
        n_test = max(1, int(len(unique_subs) * test_ratio))
        test_subjects = list(
            rng.choice(unique_subs, size=n_test, replace=False)
        )

    remaining = [s for s in unique_subs if s not in test_subjects]

    if val_subjects is None:
        n_val = max(1, int(len(unique_subs) * val_ratio))
        val_subjects = list(
            rng.choice(remaining, size=n_val, replace=False)
        )

    train_subjects = [
        s for s in unique_subs
        if s not in test_subjects and s not in val_subjects
    ]

    train_idx = np.where(np.isin(subject_ids, train_subjects))[0]
    val_idx   = np.where(np.isin(subject_ids, val_subjects))[0]
    test_idx  = np.where(np.isin(subject_ids, test_subjects))[0]

    logger.info(
        f"Subject split | "
        f"train_subs={sorted(train_subjects)} "
        f"val_subs={sorted(val_subjects)} "
        f"test_subs={sorted(test_subjects)}"
    )
    logger.info(
        f"               windows → "
        f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
    )
    return train_idx, val_idx, test_idx


# ── LOSO fold generator ───────────────────────────────────────────────────────

def loso_folds(
    subject_ids: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate Leave-One-Subject-Out fold index pairs.

    Returns:
        list of (train_idx, test_idx) tuples, one per subject
    """
    folds = []
    for sub in np.unique(subject_ids):
        train_idx = np.where(subject_ids != sub)[0]
        test_idx  = np.where(subject_ids == sub)[0]
        folds.append((train_idx, test_idx))
    logger.info(f"LOSO folds generated: {len(folds)} folds")
    return folds


# ── Save / Load split files ───────────────────────────────────────────────────

def save_splits(
    train_idx : np.ndarray,
    val_idx   : np.ndarray,
    test_idx  : np.ndarray,
    target    : str,
    split_dir : str = "data/splits/",
    split_type: str = "subject",
):
    os.makedirs(split_dir, exist_ok=True)
    base = os.path.join(split_dir, f"{split_type}_{target}")
    np.save(f"{base}_train.npy", train_idx)
    np.save(f"{base}_val.npy",   val_idx)
    np.save(f"{base}_test.npy",  test_idx)

    meta = {
        "target"    : target,
        "split_type": split_type,
        "n_train"   : int(len(train_idx)),
        "n_val"     : int(len(val_idx)),
        "n_test"    : int(len(test_idx)),
    }
    with open(f"{base}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Splits saved → {base}_*.npy")


def load_splits(
    target    : str,
    split_dir : str = "data/splits/",
    split_type: str = "subject",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = os.path.join(split_dir, f"{split_type}_{target}")
    train_idx = np.load(f"{base}_train.npy")
    val_idx   = np.load(f"{base}_val.npy")
    test_idx  = np.load(f"{base}_test.npy")
    logger.info(
        f"Splits loaded | train={len(train_idx)} "
        f"val={len(val_idx)} test={len(test_idx)}"
    )
    return train_idx, val_idx, test_idx