"""
loader.py
Responsible for: Loading raw DREAMER.mat and returning structured data.
"""

import os
import numpy as np
from typing import Optional

try:
    import mat73
    _MAT_LOADER = "mat73"
except ImportError:
    import scipy.io as _sio
    _MAT_LOADER = "scipy"


# ── Constants ─────────────────────────────────────────────────────────────────
EEG_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7",
    "O1",  "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]
N_EEG_CHANNELS = 14
N_ECG_CHANNELS = 2
EEG_FS         = 128   # Hz
ECG_FS         = 256   # Hz
N_SUBJECTS     = 23
N_VIDEOS       = 18
LABEL_COLS     = ["valence", "arousal", "dominance"]


# ── Loader ────────────────────────────────────────────────────────────────────
def load_dreamer_mat(path: str) -> dict:
    """
    Load DREAMER.mat file.

    Args:
        path: Absolute or relative path to DREAMER.mat

    Returns:
        dreamer: dict with keys EEG_SamplingRate, ECG_SamplingRate, Data, etc.

    Raises:
        FileNotFoundError: if path does not exist
        RuntimeError: if loading fails
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"DREAMER.mat not found at: {path}")

    try:
        if _MAT_LOADER == "mat73":
            raw = mat73.loadmat(path)
        else:
            import scipy.io as sio
            raw = sio.loadmat(path, simplify_cells=True)
        return raw["DREAMER"]
    except Exception as e:
        raise RuntimeError(f"Failed to load DREAMER.mat: {e}")


def get_subject_data(dreamer: dict, subject_idx: int) -> dict:
    """
    Return data dict for a single subject (0-indexed).

    Args:
        dreamer    : output of load_dreamer_mat()
        subject_idx: 0 to 22

    Returns:
        dict with keys: EEG, ECG, ScoreValence, ScoreArousal, ScoreDominance
    """
    if not (0 <= subject_idx < N_SUBJECTS):
        raise IndexError(f"subject_idx must be 0–{N_SUBJECTS-1}, got {subject_idx}")
    return dreamer["Data"][subject_idx]


def get_trial_signals(subject: dict, video_idx: int, mode: str = "stimuli"):
    """
    Extract EEG and ECG arrays for one trial.

    Args:
        subject  : output of get_subject_data()
        video_idx: 0 to 17
        mode     : 'stimuli' or 'baseline'

    Returns:
        eeg: np.ndarray shape (samples, 14)
        ecg: np.ndarray shape (samples, 2)
    """
    if mode not in ("stimuli", "baseline"):
        raise ValueError("mode must be 'stimuli' or 'baseline'")
    if not (0 <= video_idx < N_VIDEOS):
        raise IndexError(f"video_idx must be 0–{N_VIDEOS-1}, got {video_idx}")

    eeg = np.array(subject["EEG"][mode][video_idx], dtype=np.float32)
    ecg = np.array(subject["ECG"][mode][video_idx], dtype=np.float32)
    return eeg, ecg


def get_trial_labels(subject: dict, video_idx: int) -> dict:
    """
    Return valence, arousal, dominance scores for a trial.

    Args:
        subject  : output of get_subject_data()
        video_idx: 0 to 17

    Returns:
        dict with keys valence, arousal, dominance (float, 1–5)
    """
    return {
        "valence"  : float(np.array(subject["ScoreValence"]).flatten()[video_idx]),
        "arousal"  : float(np.array(subject["ScoreArousal"]).flatten()[video_idx]),
        "dominance": float(np.array(subject["ScoreDominance"]).flatten()[video_idx]),
    }


def build_trial_index(dreamer: dict) -> list:
    """
    Build a flat list of all (subject_idx, video_idx) trial references.

    Returns:
        list of dicts: [{subject, video, valence, arousal, dominance}, ...]
    """
    trials = []
    for sub_idx in range(N_SUBJECTS):
        subject = get_subject_data(dreamer, sub_idx)
        for vid_idx in range(N_VIDEOS):
            labels = get_trial_labels(subject, vid_idx)
            trials.append({
                "subject_idx": sub_idx,
                "video_idx"  : vid_idx,
                **labels
            })
    return trials


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/DREAMER.mat"
    dreamer = load_dreamer_mat(path)
    trials  = build_trial_index(dreamer)
    print(f"✅ Loaded {len(trials)} trials from {N_SUBJECTS} subjects × {N_VIDEOS} videos")

    sub0   = get_subject_data(dreamer, 0)
    eeg, ecg = get_trial_signals(sub0, 0)
    labels   = get_trial_labels(sub0, 0)
    print(f"   EEG shape : {eeg.shape}  ECG shape : {ecg.shape}")
    print(f"   Labels    : {labels}")