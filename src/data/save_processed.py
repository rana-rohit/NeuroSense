"""
save_processed.py
Responsible for: Preprocessing all DREAMER trials once and caching
                 to disk as .npz files. Avoids re-processing on every
                 training run.

Usage:
    python src/data/save_processed.py --config configs/default.yaml

Outputs (in data/processed/):
    subject_XX_video_YY_eeg.npy   — shape (n_windows, win_eeg, 14)
    subject_XX_video_YY_ecg.npy   — shape (n_windows, win_ecg, 2)
    labels.csv                    — all trial labels + binary flags
    metadata.json                 — config snapshot used for processing
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
))

from src.utils.config      import load_config
from src.utils.logger      import get_logger
from src.data.loader       import (
    load_dreamer_mat, get_subject_data,
    get_trial_signals, get_trial_labels,
    N_SUBJECTS, N_VIDEOS, LABEL_COLS
)
from src.data.preprocessor import process_trial

logger = get_logger("save_processed")


# ── Filename helpers ──────────────────────────────────────────────────────────

def eeg_path(out_dir: str, sub: int, vid: int) -> str:
    return os.path.join(out_dir, f"sub{sub:02d}_vid{vid:02d}_eeg.npy")

def ecg_path(out_dir: str, sub: int, vid: int) -> str:
    return os.path.join(out_dir, f"sub{sub:02d}_vid{vid:02d}_ecg.npy")


# ── Main processor ────────────────────────────────────────────────────────────

def preprocess_and_save(config: dict, overwrite: bool = False):
    """
    Load DREAMER, preprocess all trials, save to data/processed/.

    Args:
        config   : loaded YAML config dict
        overwrite: re-process even if files already exist
    """
    dc       = config["data"]
    lc       = config["labels"]
    out_dir  = dc["processed_path"]
    os.makedirs(out_dir, exist_ok=True)

    dreamer  = load_dreamer_mat(dc["raw_path"])
    threshold = float(lc["threshold"])

    label_records = []
    skipped       = 0
    saved         = 0

    for sub_idx in tqdm(range(N_SUBJECTS), desc="Subjects"):
        subject = get_subject_data(dreamer, sub_idx)
        sub_id  = sub_idx + 1

        for vid_idx in range(N_VIDEOS):
            vid_id   = vid_idx + 1
            eeg_file = eeg_path(out_dir, sub_id, vid_id)
            ecg_file = ecg_path(out_dir, sub_id, vid_id)

            # Skip if already processed
            if not overwrite and os.path.exists(eeg_file):
                raw_labels = get_trial_labels(subject, vid_idx)
                bin_labels = {k: int(v > threshold)
                              for k, v in raw_labels.items()}
                label_records.append({
                    "subject": sub_id, "video": vid_id,
                    **raw_labels, **{f"{k}_bin": v
                                     for k, v in bin_labels.items()}
                })
                continue

            try:
                eeg_s, ecg_s = get_trial_signals(subject, vid_idx, "stimuli")
                eeg_b, ecg_b = get_trial_signals(subject, vid_idx, "baseline")
                raw_labels   = get_trial_labels(subject, vid_idx)

                eeg_segs, ecg_segs = process_trial(
                    eeg_s, ecg_s, eeg_b, ecg_b,
                    eeg_fs=float(dc["sampling_rate_eeg"]),
                    ecg_fs=float(dc["sampling_rate_ecg"]),
                    window_sec=float(dc["segment_length"]),
                    overlap_sec=float(dc["overlap"]),
                    norm_method=dc["norm_method"],
                )

                np.save(eeg_file, eeg_segs)
                np.save(ecg_file, ecg_segs)

                bin_labels = {k: int(v > threshold)
                              for k, v in raw_labels.items()}
                label_records.append({
                    "subject": sub_id, "video": vid_id,
                    **raw_labels,
                    **{f"{k}_bin": v for k, v in bin_labels.items()}
                })
                saved += 1

            except Exception as e:
                logger.warning(
                    f"Skipped sub={sub_id} vid={vid_id}: {e}"
                )
                skipped += 1

    # Save labels CSV
    df_labels = pd.DataFrame(label_records)
    labels_path = os.path.join(out_dir, "labels.csv")
    df_labels.to_csv(labels_path, index=False)
    logger.info(f"Labels saved → {labels_path}")

    # Save metadata snapshot
    meta = {
        "sampling_rate_eeg": dc["sampling_rate_eeg"],
        "sampling_rate_ecg": dc["sampling_rate_ecg"],
        "segment_length"   : dc["segment_length"],
        "overlap"          : dc["overlap"],
        "norm_method"      : dc["norm_method"],
        "threshold"        : threshold,
        "n_subjects"       : N_SUBJECTS,
        "n_videos"         : N_VIDEOS,
        "trials_saved"     : saved,
        "trials_skipped"   : skipped,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        f"✅ Processing complete | "
        f"saved={saved} skipped={skipped} total={saved+skipped}"
    )
    return df_labels


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess DREAMER and cache to disk"
    )
    parser.add_argument("--config",    type=str,
                        default="configs/default.yaml")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process even if files exist")
    args = parser.parse_args()

    cfg = load_config(args.config)
    preprocess_and_save(cfg, overwrite=args.overwrite)