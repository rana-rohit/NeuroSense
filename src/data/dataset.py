"""
dataset.py
Responsible for: PyTorch Dataset wrapping preprocessed EEG & ECG segments.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional

from src.data.loader import (
    load_dreamer_mat, get_subject_data,
    get_trial_signals, get_trial_labels,
    N_SUBJECTS, N_VIDEOS, LABEL_COLS
)
from src.data.preprocessor import process_trial


# ── Label binarisation ────────────────────────────────────────────────────────

def binarize_labels(labels: Dict[str, float],
                    threshold: float = 3.0,
                    targets: List[str] = LABEL_COLS) -> Dict[str, int]:
    """
    Convert continuous scores (1–5) to binary (0=Low, 1=High).
    Score > threshold → 1, else → 0
    """
    return {k: int(labels[k] > threshold) for k in targets}


# ── Dataset ───────────────────────────────────────────────────────────────────

class DREAMERDataset(Dataset):
    """
    PyTorch Dataset for DREAMER EEG + ECG emotion recognition.

    Each sample is one time-window segment from one trial.
    Labels are binarised valence / arousal / dominance.

    Args:
        mat_path    : path to DREAMER.mat
        target      : one of 'valence', 'arousal', 'dominance'
        subject_ids : list of 1-indexed subject IDs to include (None = all)
        window_sec  : segment window length (seconds)
        overlap_sec : segment overlap (seconds)
        norm_method : 'zscore' | 'minmax'
        threshold   : binarisation threshold (default 3.0)
        mode        : 'stimuli' (default) or 'baseline'

    Returns per __getitem__:
        eeg : torch.Tensor (window_samples_eeg, 14)   float32
        ecg : torch.Tensor (window_samples_ecg, 2)    float32
        label: torch.Tensor scalar                     long
    """

    def __init__(
        self,
        mat_path    : str,
        target      : str  = "valence",
        subject_ids : Optional[List[int]] = None,
        window_sec  : float = 4.0,
        overlap_sec : float = 2.0,
        norm_method : str   = "zscore",
        threshold   : float = 3.0,
        mode        : str   = "stimuli",
    ):
        if target not in LABEL_COLS:
            raise ValueError(f"target must be one of {LABEL_COLS}")

        self.target      = target
        self.threshold   = threshold
        self.window_sec  = window_sec
        self.overlap_sec = overlap_sec

        # Load raw data
        dreamer = load_dreamer_mat(mat_path)

        # Resolve subject list (convert 1-indexed → 0-indexed)
        if subject_ids is None:
            sub_indices = list(range(N_SUBJECTS))
        else:
            sub_indices = [s - 1 for s in subject_ids]

        # Build sample list
        self.samples: List[Dict] = []     # {eeg, ecg, label}

        for sub_idx in sub_indices:
            subject = get_subject_data(dreamer, sub_idx)

            for vid_idx in range(N_VIDEOS):
                try:
                    eeg_s, ecg_s = get_trial_signals(subject, vid_idx, mode="stimuli")
                    eeg_b, ecg_b = get_trial_signals(subject, vid_idx, mode="baseline")
                    raw_labels   = get_trial_labels(subject, vid_idx)
                    bin_labels   = binarize_labels(raw_labels, threshold)
                    label        = bin_labels[target]

                    eeg_segs, ecg_segs = process_trial(
                        eeg_s, ecg_s, eeg_b, ecg_b,
                        window_sec=window_sec,
                        overlap_sec=overlap_sec,
                        norm_method=norm_method,
                    )

                    for w in range(eeg_segs.shape[0]):
                        self.samples.append({
                            "eeg"       : eeg_segs[w],    # (win_eeg, 14)
                            "ecg"       : ecg_segs[w],    # (win_ecg, 2)
                            "label"     : label,
                            "subject"   : sub_idx + 1,
                            "video"     : vid_idx + 1,
                        })

                except Exception as e:
                    print(f"  ⚠ Skipped sub={sub_idx+1} vid={vid_idx+1}: {e}")
                    continue

        print(f"✅ DREAMERDataset | target={target} | "
              f"subjects={len(sub_indices)} | windows={len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        eeg   = torch.from_numpy(s["eeg"]).float()    # (win_eeg, 14)
        ecg   = torch.from_numpy(s["ecg"]).float()    # (win_ecg, 2)
        label = torch.tensor(s["label"], dtype=torch.long)
        return eeg, ecg, label

    def class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for imbalanced data."""
        labels = np.array([s["label"] for s in self.samples])
        classes, counts = np.unique(labels, return_counts=True)
        weights = 1.0 / counts.astype(float)
        weights = weights / weights.sum()
        w = torch.zeros(len(classes))
        for i, c in enumerate(classes):
            w[c] = weights[i]
        return w

    def summary(self):
        labels = [s["label"] for s in self.samples]
        n_pos  = sum(labels)
        n_neg  = len(labels) - n_pos
        print(f"  Target    : {self.target}")
        print(f"  Windows   : {len(self.samples)}")
        print(f"  Class 0   : {n_neg}  ({100*n_neg/len(labels):.1f}%)")
        print(f"  Class 1   : {n_pos}  ({100*n_pos/len(labels):.1f}%)")