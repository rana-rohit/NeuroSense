"""
tests/test_inference.py
Tests for src/inference/predict.py and signal quality checks.
Run: python -m pytest tests/test_inference.py -v
"""

import pytest
import numpy as np
import os, sys, json, tempfile

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

SKIP_TORCH = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="torch not installed"
)

EEG_SAMPLES = 60 * 128
ECG_SAMPLES = 60 * 256


@pytest.fixture
def dummy_eeg():
    return np.random.randn(EEG_SAMPLES, 14).astype(np.float32)

@pytest.fixture
def dummy_ecg():
    return np.random.randn(ECG_SAMPLES, 2).astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# SIGNAL VALIDATOR TESTS
# ══════════════════════════════════════════════════════════════════

class TestValidateSignals:

    def test_valid_signals_pass(self, dummy_eeg, dummy_ecg):
        from src.inference.predict import validate_signals
        validate_signals(dummy_eeg, dummy_ecg)   # should not raise

    def test_wrong_eeg_channels_raises(self, dummy_ecg):
        from src.inference.predict import validate_signals
        bad_eeg = np.random.randn(EEG_SAMPLES, 10).astype(np.float32)
        with pytest.raises(ValueError, match="EEG must be"):
            validate_signals(bad_eeg, dummy_ecg)

    def test_wrong_ecg_channels_raises(self, dummy_eeg):
        from src.inference.predict import validate_signals
        bad_ecg = np.random.randn(ECG_SAMPLES, 5).astype(np.float32)
        with pytest.raises(ValueError, match="ECG must be"):
            validate_signals(dummy_eeg, bad_ecg)

    def test_nan_eeg_raises(self, dummy_eeg, dummy_ecg):
        from src.inference.predict import validate_signals
        bad = dummy_eeg.copy(); bad[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            validate_signals(bad, dummy_ecg)

    def test_inf_eeg_raises(self, dummy_eeg, dummy_ecg):
        from src.inference.predict import validate_signals
        bad = dummy_eeg.copy(); bad[0, 0] = float("inf")
        with pytest.raises(ValueError, match="NaN or Inf"):
            validate_signals(bad, dummy_ecg)

    def test_nan_ecg_raises(self, dummy_eeg, dummy_ecg):
        from src.inference.predict import validate_signals
        bad = dummy_ecg.copy(); bad[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            validate_signals(dummy_eeg, bad)

    def test_short_eeg_raises(self, dummy_ecg):
        from src.inference.predict import validate_signals
        short = np.random.randn(100, 14).astype(np.float32)
        with pytest.raises(ValueError, match="too short"):
            validate_signals(short, dummy_ecg)

    def test_1d_eeg_raises(self, dummy_ecg):
        from src.inference.predict import validate_signals
        with pytest.raises(ValueError):
            validate_signals(np.random.randn(EEG_SAMPLES).astype(np.float32), dummy_ecg)


# ══════════════════════════════════════════════════════════════════
# EMOTION PREDICTOR INTEGRATION (requires torch)
# ══════════════════════════════════════════════════════════════════

@SKIP_TORCH
class TestEmotionPredictor:

    @pytest.fixture
    def predictor(self, tmp_path):
        from src.inference.predict import EmotionPredictor
        from src.models.deep_model import FusionModel

        model = FusionModel(branch_dim=64)
        ckpt  = str(tmp_path / "model.pt")
        torch.save(model.state_dict(), ckpt)

        config = {
            "data": {
                "sampling_rate_eeg": 128,
                "sampling_rate_ecg": 256,
                "segment_length"   : 4,
                "overlap"          : 2,
                "norm_method"      : "zscore",
            },
            "labels"  : {"threshold": 3.0},
            "model"   : {"dropout": 0.3, "branch_dim": 64},
            "training": {},
        }
        return EmotionPredictor(ckpt, "fusion", "valence", config)

    def test_predict_returns_dict(self, predictor, dummy_eeg, dummy_ecg):
        result = predictor.predict(dummy_eeg, dummy_ecg)
        assert isinstance(result, dict)

    def test_result_has_all_keys(self, predictor, dummy_eeg, dummy_ecg):
        result = predictor.predict(dummy_eeg, dummy_ecg)
        required = ["target","prediction","class_id","confidence",
                    "prob_high","prob_low","n_windows","window_preds"]
        for k in required:
            assert k in result, f"Missing key: {k}"

    def test_prediction_is_high_or_low(self, predictor, dummy_eeg, dummy_ecg):
        result = predictor.predict(dummy_eeg, dummy_ecg)
        assert result["prediction"] in ("High", "Low")

    def test_probabilities_sum_to_one(self, predictor, dummy_eeg, dummy_ecg):
        result = predictor.predict(dummy_eeg, dummy_ecg)
        total  = result["prob_high"] + result["prob_low"]
        assert abs(total - 1.0) < 1e-4

    def test_confidence_in_range(self, predictor, dummy_eeg, dummy_ecg):
        result = predictor.predict(dummy_eeg, dummy_ecg)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_n_windows_positive(self, predictor, dummy_eeg, dummy_ecg):
        result = predictor.predict(dummy_eeg, dummy_ecg)
        assert result["n_windows"] > 0

    def test_window_preds_is_list(self, predictor, dummy_eeg, dummy_ecg):
        result = predictor.predict(dummy_eeg, dummy_ecg)
        assert isinstance(result["window_preds"], list)
        assert len(result["window_preds"]) == result["n_windows"]

    def test_predict_with_explicit_baseline(self, predictor, dummy_eeg, dummy_ecg):
        eeg_base = np.random.randn(10 * 128, 14).astype(np.float32)
        ecg_base = np.random.randn(10 * 256, 2).astype(np.float32)
        result   = predictor.predict(dummy_eeg, dummy_ecg, eeg_base, ecg_base)
        assert result["n_windows"] > 0

    def test_target_field_correct(self, predictor, dummy_eeg, dummy_ecg):
        result = predictor.predict(dummy_eeg, dummy_ecg)
        assert result["target"] == "valence"

    def test_nonexistent_checkpoint_raises(self, tmp_path, dummy_eeg, dummy_ecg):
        from src.inference.predict import EmotionPredictor
        config = {
            "data": {"sampling_rate_eeg":128,"sampling_rate_ecg":256,
                     "segment_length":4,"overlap":2,"norm_method":"zscore"},
            "labels":{"threshold":3.0},
            "model":{"dropout":0.3,"branch_dim":64},
            "training":{},
        }
        with pytest.raises(FileNotFoundError):
            EmotionPredictor("/no/such/file.pt", "fusion", "valence", config)

    def test_output_json_serialisable(self, predictor, dummy_eeg, dummy_ecg):
        result = predictor.predict(dummy_eeg, dummy_ecg)
        # Should not raise
        serialised = json.dumps(result)
        assert len(serialised) > 0