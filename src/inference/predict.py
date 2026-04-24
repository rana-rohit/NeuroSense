"""
predict.py — production inference from raw EEG + ECG.

Usage:
    python src/inference/predict.py \
        --eeg_path data/raw/sample_eeg.npy \
        --ecg_path data/raw/sample_ecg.npy \
        --model_path outputs/models/best_valence_FusionModel.pt \
        --model_type fusion --target valence \
        --config configs/default.yaml
"""

import os, sys, json, argparse
from typing import Optional, Dict

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
))

from src.utils.config      import load_config
from src.data.preprocessor import process_trial
from src.models.deep_model import build_model

def validate_signals(eeg: np.ndarray, ecg: np.ndarray):
    """Raise ValueError on bad shapes / values."""
    if eeg.ndim != 2 or eeg.shape[1] != 14:
        raise ValueError(f"EEG must be (samples, 14), got {eeg.shape}")
    if ecg.ndim != 2 or ecg.shape[1] != 2:
        raise ValueError(f"ECG must be (samples, 2), got {ecg.shape}")
    if np.any(np.isnan(eeg)) or np.any(np.isinf(eeg)):
        raise ValueError("EEG contains NaN or Inf values")
    if np.any(np.isnan(ecg)) or np.any(np.isinf(ecg)):
        raise ValueError("ECG contains NaN or Inf values")
    min_samples = int(5.0 * 128)
    if eeg.shape[0] < min_samples:
        raise ValueError(
            f"EEG too short: {eeg.shape[0]} samples (need ≥ {min_samples})"
        )


class EmotionPredictor:
    """
    End-to-end predictor.

    Args:
        model_path : path to saved .pt checkpoint
        model_type : 'fusion' | 'eegnet' | 'cnn' | 'cnnlstm'
        target     : 'valence' | 'arousal' | 'dominance'
        config     : loaded config dict
    """

    LABEL_MAP = {0: "Low", 1: "High"}

    def __init__(self, model_path: str, model_type: str,
                 target: str, config: dict):
        import torch as torch
        self.target      = target
        self.config      = config
        self.device      = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        dc = config["data"]
        self.eeg_fs      = float(dc["sampling_rate_eeg"])
        self.ecg_fs      = float(dc["sampling_rate_ecg"])
        self.window_sec  = float(dc["segment_length"])
        self.overlap_sec = float(dc["overlap"])
        self.norm_method = dc["norm_method"]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        self.model = build_model(model_type, config)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device).eval()
        print(f"✅ Predictor ready | target={target} device={self.device}")

    def predict(
        self,
        eeg_stim : np.ndarray,
        ecg_stim : np.ndarray,
        eeg_base : Optional[np.ndarray] = None,
        ecg_base : Optional[np.ndarray] = None,
    ) -> Dict:
        validate_signals(eeg_stim, ecg_stim)

        # fallback baselines = zeros matching 5s
        if eeg_base is None:
            eeg_base = np.zeros((5 * int(self.eeg_fs), 14), dtype=np.float32)
        if ecg_base is None:
            ecg_base = np.zeros((5 * int(self.ecg_fs), 2), dtype=np.float32)

        eeg_segs, ecg_segs = process_trial(
            eeg_stim.astype(np.float32), ecg_stim.astype(np.float32),
            eeg_base.astype(np.float32), ecg_base.astype(np.float32),
            eeg_fs=self.eeg_fs, ecg_fs=self.ecg_fs,
            window_sec=self.window_sec, overlap_sec=self.overlap_sec,
            norm_method=self.norm_method,
        )

        all_probs = []
        for w in range(eeg_segs.shape[0]):
            eeg_t = torch.from_numpy(eeg_segs[w]).float().unsqueeze(0).to(self.device)
            ecg_t = torch.from_numpy(ecg_segs[w]).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                if hasattr(self.model, "ecg_branch"):
                    logits = self.model(eeg_t, ecg_t)
                else:
                    logits = self.model(eeg_t)
            probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
            all_probs.append(probs)

        all_probs  = np.stack(all_probs)
        mean_probs = all_probs.mean(axis=0)
        class_id   = int(np.argmax(mean_probs))

        return {
            "target"     : self.target,
            "prediction" : self.LABEL_MAP[class_id],
            "class_id"   : class_id,
            "confidence" : round(float(mean_probs[class_id]), 4),
            "prob_high"  : round(float(mean_probs[1]), 4),
            "prob_low"   : round(float(mean_probs[0]), 4),
            "n_windows"  : int(eeg_segs.shape[0]),
            "window_preds": [
                {"window": i,
                 "prediction": self.LABEL_MAP[int(np.argmax(p))],
                 "prob_high": round(float(p[1]), 4)}
                for i, p in enumerate(all_probs)
            ],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg_path",   required=True)
    parser.add_argument("--ecg_path",   required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_type", default="fusion",
                        choices=["eegnet","cnn","cnnlstm","fusion"])
    parser.add_argument("--target",     default="valence",
                        choices=["valence","arousal","dominance"])
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--out",        default=None)
    args = parser.parse_args()

    cfg  = load_config(args.config)
    eeg  = np.load(args.eeg_path).astype(np.float32)
    ecg  = np.load(args.ecg_path).astype(np.float32)

    predictor = EmotionPredictor(args.model_path, args.model_type,
                                  args.target, cfg)
    result = predictor.predict(eeg, ecg)

    print(f"\n{'='*45}")
    print(f"  Target     : {result['target'].capitalize()}")
    print(f"  Prediction : {result['prediction']}")
    print(f"  Confidence : {result['confidence']:.1%}")
    print(f"  Prob High  : {result['prob_high']:.4f}")
    print(f"  Prob Low   : {result['prob_low']:.4f}")
    print(f"  Windows    : {result['n_windows']}")
    print(f"{'='*45}\n")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved → {args.out}")