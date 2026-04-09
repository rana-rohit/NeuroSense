"""
predict.py
Responsible for: Production inference — load a trained model and
                 predict emotion from raw EEG + ECG signals.

Usage:
    python src/inference/predict.py \
        --eeg_path  data/raw/sample_eeg.npy \
        --ecg_path  data/raw/sample_ecg.npy \
        --model_path outputs/models/best_valence_FusionModel.pt \
        --target    valence \
        --config    configs/default.yaml
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
))

from src.utils.config      import load_config
from src.utils.logger      import get_logger
from src.data.preprocessor import process_trial
from src.models.deep_model import build_model, FusionModel

logger = get_logger("predict")


# ── Signal validator ──────────────────────────────────────────────────────────

def validate_signals(eeg: np.ndarray, ecg: np.ndarray):
    """
    Basic shape and value checks before inference.

    Raises:
        ValueError on invalid input
    """
    if eeg.ndim != 2 or eeg.shape[1] != 14:
        raise ValueError(
            f"EEG must be (samples, 14), got {eeg.shape}"
        )
    if ecg.ndim != 2 or ecg.shape[1] != 2:
        raise ValueError(
            f"ECG must be (samples, 2), got {ecg.shape}"
        )
    if np.any(np.isnan(eeg)) or np.any(np.isinf(eeg)):
        raise ValueError("EEG contains NaN or Inf values")
    if np.any(np.isnan(ecg)) or np.any(np.isinf(ecg)):
        raise ValueError("ECG contains NaN or Inf values")

    min_eeg_sec = 5.0
    min_ecg_sec = 5.0
    if eeg.shape[0] < min_eeg_sec * 128:
        raise ValueError(
            f"EEG too short: {eeg.shape[0]} samples "
            f"(need ≥ {int(min_eeg_sec*128)})"
        )
    logger.info(
        f"Signal validated — EEG: {eeg.shape} | ECG: {ecg.shape}"
    )


# ── Predictor class ───────────────────────────────────────────────────────────

class EmotionPredictor:
    """
    End-to-end emotion predictor.

    Args:
        model_path  : path to saved .pt checkpoint
        model_type  : 'fusion' | 'eegnet' | 'cnn' | 'cnnlstm'
        target      : 'valence' | 'arousal' | 'dominance'
        config      : loaded YAML config dict
    """

    LABEL_MAP = {0: "Low", 1: "High"}

    def __init__(
        self,
        model_path : str,
        model_type : str,
        target     : str,
        config     : dict,
    ):
        self.config     = config
        self.target     = target
        self.device     = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.eeg_fs     = float(config["data"]["sampling_rate_eeg"])
        self.ecg_fs     = float(config["data"]["sampling_rate_ecg"])
        self.window_sec = float(config["data"]["segment_length"])
        self.overlap_sec= float(config["data"]["overlap"])
        self.norm_method= config["data"]["norm_method"]
        self.threshold  = float(config["labels"]["threshold"])

        # Load model
        self.model = build_model(model_type, config)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            f"EmotionPredictor ready | target={target} "
            f"device={self.device} model={model_type}"
        )

    @torch.no_grad()
    def predict(
        self,
        eeg_stim  : np.ndarray,
        ecg_stim  : np.ndarray,
        eeg_base  : Optional[np.ndarray] = None,
        ecg_base  : Optional[np.ndarray] = None,
    ) -> dict:
        """
        Predict emotion from raw EEG + ECG signals.

        Args:
            eeg_stim  : stimuli EEG  (samples, 14)  float32
            ecg_stim  : stimuli ECG  (samples, 2)   float32
            eeg_base  : baseline EEG (samples, 14)  — uses zero if None
            ecg_base  : baseline ECG (samples, 2)   — uses zero if None

        Returns:
            dict:
                target      : emotion dimension
                prediction  : 'High' or 'Low'
                class_id    : 0 or 1
                confidence  : probability of predicted class
                prob_high   : probability of High emotion
                prob_low    : probability of Low emotion
                window_preds: per-window predictions list
        """
        validate_signals(eeg_stim, ecg_stim)

        # Synthetic zero baseline if not provided
        if eeg_base is None:
            eeg_base = np.zeros_like(eeg_stim[:128])
        if ecg_base is None:
            ecg_base = np.zeros_like(ecg_stim[:256])

        # Preprocess + segment
        eeg_segs, ecg_segs = process_trial(
            eeg_stim.astype(np.float32),
            ecg_stim.astype(np.float32),
            eeg_base.astype(np.float32),
            ecg_base.astype(np.float32),
            eeg_fs=self.eeg_fs,
            ecg_fs=self.ecg_fs,
            window_sec=self.window_sec,
            overlap_sec=self.overlap_sec,
            norm_method=self.norm_method,
        )

        # Batch inference over all windows
        all_probs = []
        for w in range(eeg_segs.shape[0]):
            eeg_t = torch.from_numpy(eeg_segs[w]).float().unsqueeze(0).to(self.device)
            ecg_t = torch.from_numpy(ecg_segs[w]).float().unsqueeze(0).to(self.device)

            if hasattr(self.model, "ecg_branch"):
                logits = self.model(eeg_t, ecg_t)
            else:
                logits = self.model(eeg_t)

            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            all_probs.append(probs)

        all_probs   = np.stack(all_probs)           # (n_windows, 2)
        mean_probs  = all_probs.mean(axis=0)        # averaged across windows
        class_id    = int(np.argmax(mean_probs))

        window_preds = [
            {
                "window"    : i,
                "prediction": self.LABEL_MAP[int(np.argmax(p))],
                "prob_high" : round(float(p[1]), 4),
            }
            for i, p in enumerate(all_probs)
        ]

        result = {
            "target"      : self.target,
            "prediction"  : self.LABEL_MAP[class_id],
            "class_id"    : class_id,
            "confidence"  : round(float(mean_probs[class_id]), 4),
            "prob_high"   : round(float(mean_probs[1]), 4),
            "prob_low"    : round(float(mean_probs[0]), 4),
            "n_windows"   : eeg_segs.shape[0],
            "window_preds": window_preds,
        }

        logger.info(
            f"Prediction: {result['target']} = {result['prediction']} "
            f"(conf={result['confidence']}  prob_high={result['prob_high']})"
        )
        return result

    def predict_all_targets(
        self,
        eeg_stim  : np.ndarray,
        ecg_stim  : np.ndarray,
        model_paths: dict,
        model_type : str,
        eeg_base   : Optional[np.ndarray] = None,
        ecg_base   : Optional[np.ndarray] = None,
    ) -> dict:
        """
        Predict all three emotion dimensions using separate models.

        Args:
            model_paths: {"valence": "path.pt", "arousal": ..., "dominance": ...}

        Returns:
            dict with predictions for each target
        """
        results = {}
        for tgt, mpath in model_paths.items():
            self.target = tgt
            self.model.load_state_dict(
                torch.load(mpath, map_location=self.device)
            )
            results[tgt] = self.predict(
                eeg_stim, ecg_stim, eeg_base, ecg_base
            )
        return results


# Add missing Optional import
from typing import Optional, Dict


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Emotion inference from EEG + ECG"
    )
    parser.add_argument("--eeg_path",
                        type=str, required=True,
                        help="Path to EEG .npy file (samples, 14)")
    parser.add_argument("--ecg_path",
                        type=str, required=True,
                        help="Path to ECG .npy file (samples, 2)")
    parser.add_argument("--model_path",
                        type=str, required=True,
                        help="Path to saved .pt checkpoint")
    parser.add_argument("--model_type",
                        type=str, default="fusion",
                        choices=["eegnet","cnn","cnnlstm","fusion"])
    parser.add_argument("--target",
                        type=str, default="valence",
                        choices=["valence","arousal","dominance"])
    parser.add_argument("--config",
                        type=str, default="configs/default.yaml")
    parser.add_argument("--out",
                        type=str, default=None,
                        help="Optional path to save prediction JSON")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    eeg  = np.load(args.eeg_path).astype(np.float32)
    ecg  = np.load(args.ecg_path).astype(np.float32)

    predictor = EmotionPredictor(
        model_path = args.model_path,
        model_type = args.model_type,
        target     = args.target,
        config     = cfg,
    )

    result = predictor.predict(eeg, ecg)

    print("\n" + "="*45)
    print(f"  Emotion Target  : {result['target'].capitalize()}")
    print(f"  Prediction      : {result['prediction']}")
    print(f"  Confidence      : {result['confidence']:.1%}")
    print(f"  Prob(High)      : {result['prob_high']:.4f}")
    print(f"  Prob(Low)       : {result['prob_low']:.4f}")
    print(f"  Windows used    : {result['n_windows']}")
    print("="*45 + "\n")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result saved → {args.out}")