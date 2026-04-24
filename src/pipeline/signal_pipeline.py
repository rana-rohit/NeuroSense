"""
pipeline/signal_pipeline.py
Orchestrates the full signal → prediction flow.

Flow:
  SignalInput → validate → preprocess → segment → infer → aggregate → PredictionRecord
"""

import time
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.utils.logger      import get_logger
from src.data.preprocessor import process_trial
from src.schemas.models    import (
    SignalInput, EmotionPrediction, PredictionRecord,
    WindowPrediction, EmotionLabel, EmotionDimension,
    SignalQuality,
)

logger = get_logger("signal_pipeline")

TARGETS = ["valence", "arousal", "dominance"]
MIN_DURATION_SEC = 5.0   # minimum signal length


# ── Signal quality checker ────────────────────────────────────────

def check_signal_quality(
    eeg: np.ndarray,
    ecg: np.ndarray,
    eeg_fs: float,
) -> Tuple[SignalQuality, str]:
    """
    Assess EEG + ECG signal quality.

    Checks:
      - NaN / Inf values
      - Flat channels (std < threshold → likely electrode dropout)
      - Duration adequacy
      - Amplitude plausibility

    Returns:
        (SignalQuality, notes_string)
    """
    notes = []

    # NaN / Inf
    if np.any(np.isnan(eeg)) or np.any(np.isinf(eeg)):
        return SignalQuality.POOR, "EEG contains NaN or Inf values"
    if np.any(np.isnan(ecg)) or np.any(np.isinf(ecg)):
        return SignalQuality.POOR, "ECG contains NaN or Inf values"

    # Duration
    duration_sec = eeg.shape[0] / eeg_fs
    if duration_sec < MIN_DURATION_SEC:
        return SignalQuality.POOR, f"Signal too short: {duration_sec:.1f}s"

    # Flat channels (electrode dropout)
    ch_stds = eeg.std(axis=0)
    flat_channels = np.sum(ch_stds < 0.01)
    if flat_channels > 4:
        notes.append(f"{flat_channels} flat EEG channels detected")
        return SignalQuality.POOR, "; ".join(notes)
    elif flat_channels > 0:
        notes.append(f"{flat_channels} potentially flat EEG channel(s)")

    # Amplitude range plausibility (EEG typically 10-500 µV)
    eeg_range = float(np.ptp(eeg))
    if eeg_range > 5000:
        notes.append(f"EEG amplitude unusually high ({eeg_range:.0f} µV)")

    quality = SignalQuality.DEGRADED if notes else SignalQuality.GOOD
    return quality, "; ".join(notes) if notes else "OK"


# ── Inference engine wrapper ──────────────────────────────────────

class InferenceEngine:
    """
    Wraps one or more trained models for multi-target inference.

    Supports:
        - Deep model (FusionModel / EEGNet)
        - Baseline sklearn pipeline

    Args:
        model_paths: {"valence": "path.pt", "arousal": ..., "dominance": ...}
        model_type : "fusion" | "eegnet" | "rf" | "svm" (baseline)
        config     : loaded YAML config dict
    """

    def __init__(
        self,
        model_paths : Dict[str, str],
        model_type  : str,
        config      : dict,
    ):
        self.model_type  = model_type
        self.config      = config
        self.models      = {}
        self._is_deep    = model_type not in ("rf", "svm", "logreg", "gbm")

        self._load_models(model_paths)

    def _load_models(self, model_paths: Dict[str, str]):
        if self._is_deep:
            import torch
            from src.models.deep_model import build_model
            self.device = torch.device(
                "cuda" if __import__("torch").cuda.is_available() else "cpu"
            )
            for target, path in model_paths.items():
                m = build_model(self.model_type, self.config)
                m.load_state_dict(
                    __import__("torch").load(path, map_location=self.device)
                )
                m.to(self.device).eval()
                self.models[target] = m
                logger.info(f"Loaded deep model [{target}] ← {path}")
        else:
            import joblib
            for target, path in model_paths.items():
                self.models[target] = joblib.load(path)
                logger.info(f"Loaded baseline model [{target}] ← {path}")

    def predict_windows(
        self,
        eeg_segs : np.ndarray,   # (n_windows, win_eeg, 14)
        ecg_segs : np.ndarray,   # (n_windows, win_ecg, 2)
        eeg_fs   : float = 128.0,
        ecg_fs   : float = 256.0,
    ) -> Dict[str, List[dict]]:
        """
        Run inference on all windows for all targets.

        Returns:
            {target: [{"prob_high": float, "label": str}, ...]}
        """
        results = {}

        if self._is_deep:
            import torch
            for target, model in self.models.items():
                win_results = []
                model.eval()
                with __import__("torch").no_grad():
                    for w in range(eeg_segs.shape[0]):
                        eeg_t = torch.from_numpy(
                            eeg_segs[w]).float().unsqueeze(0).to(self.device)
                        ecg_t = torch.from_numpy(
                            ecg_segs[w]).float().unsqueeze(0).to(self.device)
                        if hasattr(model, "ecg_branch"):
                            logits = model(eeg_t, ecg_t)
                        else:
                            logits = model(eeg_t)
                        probs = __import__("torch").softmax(
                            logits, dim=1).squeeze(0).cpu().numpy()
                        win_results.append({
                            "prob_high": float(probs[1]),
                            "prob_low" : float(probs[0]),
                            "label"    : "High" if probs[1] > 0.5 else "Low",
                        })
                results[target] = win_results
        else:
            from src.features.eeg_features import extract_eeg_features
            from src.features.ecg_features import extract_ecg_features
            n = min(eeg_segs.shape[0], ecg_segs.shape[0])
            X = np.stack([
                np.concatenate([
                    extract_eeg_features(eeg_segs[w], fs=eeg_fs),
                    extract_ecg_features(ecg_segs[w], fs=ecg_fs),
                ])
                for w in range(n)
            ])
            for target, model in self.models.items():
                probs = model.predict_proba(X)
                results[target] = [
                    {"prob_high": float(p[1]), "prob_low": float(p[0]),
                     "label": "High" if p[1] > 0.5 else "Low"}
                    for p in probs
                ]

        return results


# ── Main pipeline ─────────────────────────────────────────────────

class SignalPipeline:
    """
    Full signal → EmotionPrediction + PredictionRecord pipeline.

    Usage:
        pipeline = SignalPipeline(engine, config)
        prediction, record = pipeline.run(signal_input)
    """

    def __init__(self, engine: InferenceEngine, config: dict):
        self.engine = engine
        self.config = config
        dc = config["data"]
        self.eeg_fs      = float(dc["sampling_rate_eeg"])
        self.ecg_fs      = float(dc["sampling_rate_ecg"])
        self.window_sec  = float(dc["segment_length"])
        self.overlap_sec = float(dc["overlap"])
        self.norm_method = dc["norm_method"]

    def run(
        self,
        signal: SignalInput,
    ) -> Tuple[EmotionPrediction, PredictionRecord]:
        """
        Execute full pipeline.

        Returns:
            (EmotionPrediction, PredictionRecord)
        """
        t_start = time.perf_counter()

        # 1. Convert to numpy
        eeg = np.array(signal.eeg_data, dtype=np.float32)
        ecg = np.array(signal.ecg_data, dtype=np.float32)

        # 2. Signal quality check
        quality, quality_notes = check_signal_quality(eeg, ecg, self.eeg_fs)
        if quality == SignalQuality.POOR:
            logger.warning(
                f"Poor signal quality [{signal.session_id}]: {quality_notes}"
            )

        # 3. Baseline (use zeros if not provided)
        eeg_base = (np.array(signal.eeg_baseline, dtype=np.float32)
                    if signal.eeg_baseline
                    else np.zeros((int(self.eeg_fs * 5), 14), dtype=np.float32))
        ecg_base = (np.array(signal.ecg_baseline, dtype=np.float32)
                    if signal.ecg_baseline
                    else np.zeros((int(self.ecg_fs * 5), 2), dtype=np.float32))

        # 4. Preprocess + segment
        try:
            eeg_segs, ecg_segs = process_trial(
                eeg, ecg, eeg_base, ecg_base,
                eeg_fs=self.eeg_fs, ecg_fs=self.ecg_fs,
                window_sec=self.window_sec, overlap_sec=self.overlap_sec,
                norm_method=self.norm_method,
            )
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise ValueError(f"Signal preprocessing failed: {e}")

        n_windows = eeg_segs.shape[0]
        logger.info(
            f"Session {signal.session_id} | "
            f"windows={n_windows} quality={quality.value}"
        )

        # 5. Inference
        window_results = self.engine.predict_windows(
            eeg_segs, ecg_segs, self.eeg_fs, self.ecg_fs
        )

        # 6. Aggregate per target
        aggregated = {}
        for target, win_list in window_results.items():
            probs_high = np.array([w["prob_high"] for w in win_list])
            mean_prob  = float(np.mean(probs_high))
            aggregated[target] = {
                "prob_high"  : round(mean_prob, 4),
                "prob_low"   : round(1 - mean_prob, 4),
                "label"      : "High" if mean_prob > 0.5 else "Low",
                "confidence" : round(abs(mean_prob - 0.5) * 2, 4),
            }

        # 7. Build window-level prediction list
        all_window_preds = []
        for target, win_list in window_results.items():
            for i, w in enumerate(win_list):
                all_window_preds.append(WindowPrediction(
                    window_index = i,
                    dimension    = EmotionDimension(target),
                    label        = EmotionLabel(w["label"]),
                    prob_high    = round(w["prob_high"], 4),
                    prob_low     = round(w["prob_low"],  4),
                    confidence   = round(abs(w["prob_high"] - 0.5) * 2, 4),
                ))

        processing_ms = round((time.perf_counter() - t_start) * 1000, 2)

        # 8. Build EmotionPrediction
        pred = EmotionPrediction(
            session_id     = signal.session_id,
            user_id        = signal.user_id,
            timestamp      = signal.timestamp,
            valence        = EmotionLabel(aggregated["valence"]["label"]),
            arousal        = EmotionLabel(aggregated["arousal"]["label"]),
            dominance      = EmotionLabel(aggregated["dominance"]["label"]),
            valence_conf   = aggregated["valence"]["confidence"],
            arousal_conf   = aggregated["arousal"]["confidence"],
            dominance_conf = aggregated["dominance"]["confidence"],
            valence_prob   = aggregated["valence"]["prob_high"],
            arousal_prob   = aggregated["arousal"]["prob_high"],
            dominance_prob = aggregated["dominance"]["prob_high"],
            n_windows      = n_windows,
            window_preds   = all_window_preds,
            signal_quality = quality,
            quality_notes  = quality_notes,
            processing_ms  = processing_ms,
        )

        # 9. Build PredictionRecord (flat, for DB)
        record = PredictionRecord(
            prediction_id  = pred.prediction_id,
            session_id     = pred.session_id,
            user_id        = pred.user_id,
            timestamp      = pred.timestamp,
            valence        = pred.valence.value,
            arousal        = pred.arousal.value,
            dominance      = pred.dominance.value,
            valence_prob   = pred.valence_prob,
            arousal_prob   = pred.arousal_prob,
            dominance_prob = pred.dominance_prob,
            n_windows      = pred.n_windows,
            signal_quality = pred.signal_quality.value,
            model_version  = pred.model_version,
            processing_ms  = pred.processing_ms,
        )

        logger.info(
            f"Prediction complete [{signal.session_id}] | "
            f"V={pred.valence.value} A={pred.arousal.value} "
            f"D={pred.dominance.value} | {processing_ms}ms"
        )

        return pred, record