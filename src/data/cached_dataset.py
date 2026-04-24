"""
cached_dataset.py
Responsible for: PyTorch Dataset that reads from preprocessed .npy files
                 instead of raw DREAMER.mat. Much faster DataLoader.

Use this AFTER running save_processed.py.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple

from src.data.loader import LABEL_COLS, N_SUBJECTS, N_VIDEOS
from src.utils.logger import get_logger

logger = get_logger("cached_dataset")


def _eeg_path(processed_dir: str, sub: int, vid: int) -> str:
    return os.path.join(processed_dir, f"sub{sub:02d}_vid{vid:02d}_eeg.npy")

def _ecg_path(processed_dir: str, sub: int, vid: int) -> str:
    return os.path.join(processed_dir, f"sub{sub:02d}_vid{vid:02d}_ecg.npy")


class CachedDREAMERDataset(Dataset):
    """
    Fast Dataset backed by preprocessed .npy files.

    Args:
        processed_dir : path to data/processed/
        target        : 'valence' | 'arousal' | 'dominance'
        subject_ids   : list of 1-indexed subject IDs (None = all)
        threshold     : binarisation threshold (default 3.0)

    Returns per __getitem__:
        eeg   : torch.Tensor (win_eeg, 14)   float32
        ecg   : torch.Tensor (win_ecg, 2)    float32
        label : torch.Tensor scalar           long
    """

    def __init__(
        self,
        processed_dir : str  = "data/processed/",
        target        : str  = "valence",
        subject_ids   : Optional[List[int]] = None,
        threshold     : float = 3.0,
    ):
        if target not in LABEL_COLS:
            raise ValueError(f"target must be one of {LABEL_COLS}")

        self.processed_dir = processed_dir
        self.target        = target
        self.threshold     = threshold

        # Load labels CSV
        labels_path = os.path.join(processed_dir, "labels.csv")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"labels.csv not found in {processed_dir}. "
                "Run src/data/save_processed.py first."
            )

        df = pd.read_csv(labels_path)

        # Filter subjects
        if subject_ids is not None:
            df = df[df["subject"].isin(subject_ids)].reset_index(drop=True)

        label_col = f"{target}_bin"
        if label_col not in df.columns:
            raise KeyError(f"Column '{label_col}' not in labels.csv")

        # Build flat sample list
        self.samples: List[dict] = []

        for _, row in df.iterrows():
            sub = int(row["subject"])
            vid = int(row["video"])
            lbl = int(row[label_col])

            ef = _eeg_path(processed_dir, sub, vid)
            cf = _ecg_path(processed_dir, sub, vid)

            if not os.path.exists(ef) or not os.path.exists(cf):
                logger.warning(
                    f"Missing npy files: sub={sub} vid={vid}, skipping"
                )
                continue

            # Load to get window count (lazy — don't load data yet)
            eeg_shape = np.load(ef, mmap_mode="r").shape
            n_windows = eeg_shape[0]

            for w in range(n_windows):
                self.samples.append({
                    "subject"  : sub,
                    "video"    : vid,
                    "window"   : w,
                    "label"    : lbl,
                    "eeg_file" : ef,
                    "ecg_file" : cf,
                })

        logger.info(
            f"CachedDREAMERDataset | target={target} | "
            f"windows={len(self.samples)}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s   = self.samples[idx]
        eeg = np.load(s["eeg_file"], mmap_mode="r")[s["window"]]
        ecg = np.load(s["ecg_file"], mmap_mode="r")[s["window"]]
        return (
            torch.from_numpy(eeg.copy()).float(),
            torch.from_numpy(ecg.copy()).float(),
            torch.tensor(s["label"], dtype=torch.long),
        )

    def class_weights(self) -> torch.Tensor:
        labels = np.array([s["label"] for s in self.samples])
        classes, counts = np.unique(labels, return_counts=True)
        weights = 1.0 / counts.astype(float)
        weights = weights / weights.sum()
        w = torch.zeros(int(classes.max()) + 1)
        for i, c in enumerate(classes):
            w[c] = float(weights[i])
        return w

    def summary(self):
        labels = [s["label"] for s in self.samples]
        n_neg  = len(labels) - n_pos
        print(f"  Target      : {self.target}")
        print(f"  Total wins  : {len(self.samples)}")
        print(f"  Class 0     : {n_neg}  ({100*n_neg/len(labels):.1f}%)")
        print(f"  Class 1     : {n_pos}  ({100*n_pos/len(labels):.1f}%)")