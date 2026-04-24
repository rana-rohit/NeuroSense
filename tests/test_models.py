"""
tests/test_models.py
Unit tests for src/models/deep_model.py and src/models/baseline.py
Run: python -m pytest tests/test_models.py -v
"""

import pytest
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))

# ── Skip deep model tests if torch unavailable ────────────────────
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

SKIP_TORCH = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="torch not installed"
)

BATCH     = 4
EEG_TIME  = 512
ECG_TIME  = 1024
EEG_CH    = 14
ECG_CH    = 2
N_CLASSES = 2


# ══════════════════════════════════════════════════════════════════
# BASELINE MODEL TESTS
# ══════════════════════════════════════════════════════════════════

class TestBaselineModels:

    @pytest.fixture
    def Xy(self):
        np.random.seed(42)
        X = np.random.randn(200, 280).astype(np.float32)
        y = np.random.randint(0, 2, 200)
        return X, y

    @pytest.mark.parametrize("model_type", ["logreg", "svm", "rf", "gbm"])
    def test_build_returns_pipeline(self, model_type):
        from src.models.baseline import build_baseline
        from sklearn.pipeline import Pipeline
        model = build_baseline(model_type)
        assert isinstance(model, Pipeline)

    def test_invalid_model_type_raises(self):
        from src.models.baseline import build_baseline
        with pytest.raises(ValueError, match="Unknown model_type"):
            build_baseline("xgboost_xyz")

    @pytest.mark.parametrize("model_type", ["logreg", "rf"])
    def test_fit_predict(self, model_type, Xy):
        from src.models.baseline import build_baseline
        X, y = Xy
        model = build_baseline(model_type)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (200,)
        assert set(preds).issubset({0, 1})

    @pytest.mark.parametrize("model_type", ["logreg", "rf"])
    def test_predict_proba_shape(self, model_type, Xy):
        from src.models.baseline import build_baseline
        X, y = Xy
        model = build_baseline(model_type)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (200, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_evaluate_returns_metrics(self, Xy):
        from src.models.baseline import build_baseline, evaluate
        X, y = Xy
        model = build_baseline("logreg")
        model.fit(X, y)
        metrics = evaluate(model, X, y, split="test")
        assert "accuracy"  in metrics
        assert "f1"        in metrics
        assert "roc_auc"   in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["roc_auc"]  <= 1.0

    def test_save_and_load_model(self, Xy, tmp_path):
        from src.models.baseline import build_baseline, save_model, load_model
        X, y = Xy
        model = build_baseline("logreg")
        model.fit(X, y)
        path  = str(tmp_path / "model.pkl")
        save_model(model, path)
        assert os.path.exists(path)
        loaded = load_model(path)
        preds1 = model.predict(X)
        preds2 = loaded.predict(X)
        assert np.array_equal(preds1, preds2)

    def test_load_nonexistent_raises(self):
        from src.models.baseline import load_model
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/path/model.pkl")

    def test_build_feature_matrix_shape(self):
        from src.models.baseline import build_feature_matrix
        eeg_segs = np.random.randn(5, 512, 14).astype(np.float32)
        ecg_segs = np.random.randn(5, 1024, 2).astype(np.float32)
        X = build_feature_matrix(eeg_segs, ecg_segs)
        assert X.shape == (5, 280)

    def test_build_feature_matrix_no_nan(self):
        from src.models.baseline import build_feature_matrix
        eeg_segs = np.random.randn(3, 512, 14).astype(np.float32)
        ecg_segs = np.random.randn(3, 1024, 2).astype(np.float32)
        X = build_feature_matrix(eeg_segs, ecg_segs)
        assert not np.any(np.isnan(X))


# ══════════════════════════════════════════════════════════════════
# DEEP MODEL TESTS
# ══════════════════════════════════════════════════════════════════

@SKIP_TORCH
class TestEEGNet:

    def test_output_shape_time_first(self):
        from src.models.deep_model import EEGNet
        model = EEGNet(n_channels=EEG_CH, n_samples=EEG_TIME, n_classes=N_CLASSES)
        x   = torch.randn(BATCH, EEG_TIME, EEG_CH)
        out = model(x)
        assert out.shape == (BATCH, N_CLASSES)

    def test_output_shape_ch_first(self):
        from src.models.deep_model import EEGNet
        model = EEGNet(n_channels=EEG_CH, n_samples=EEG_TIME, n_classes=N_CLASSES)
        x   = torch.randn(BATCH, EEG_CH, EEG_TIME)
        out = model(x)
        assert out.shape == (BATCH, N_CLASSES)

    def test_no_nan_output(self):
        from src.models.deep_model import EEGNet
        model = EEGNet(n_channels=EEG_CH, n_samples=EEG_TIME)
        x   = torch.randn(BATCH, EEG_TIME, EEG_CH)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_batch_size_1(self):
        from src.models.deep_model import EEGNet
        model = EEGNet(n_channels=EEG_CH, n_samples=EEG_TIME)
        x   = torch.randn(1, EEG_TIME, EEG_CH)
        out = model(x)
        assert out.shape == (1, N_CLASSES)


@SKIP_TORCH
class TestCNN1DBranch:

    def test_eeg_branch_output(self):
        from src.models.deep_model import CNN1DBranch
        branch = CNN1DBranch(in_channels=EEG_CH, out_dim=128)
        x   = torch.randn(BATCH, EEG_CH, EEG_TIME)
        out = branch(x)
        assert out.shape == (BATCH, 128)

    def test_ecg_branch_output(self):
        from src.models.deep_model import CNN1DBranch
        branch = CNN1DBranch(in_channels=ECG_CH, out_dim=64)
        x   = torch.randn(BATCH, ECG_CH, ECG_TIME)
        out = branch(x)
        assert out.shape == (BATCH, 64)

    def test_no_nan(self):
        from src.models.deep_model import CNN1DBranch
        branch = CNN1DBranch(in_channels=EEG_CH, out_dim=128)
        x   = torch.randn(BATCH, EEG_CH, EEG_TIME)
        out = branch(x)
        assert not torch.isnan(out).any()


@SKIP_TORCH
class TestCNNLSTMBranch:

    def test_output_shape(self):
        from src.models.deep_model import CNNLSTMBranch
        branch = CNNLSTMBranch(in_channels=EEG_CH, out_dim=128)
        x   = torch.randn(BATCH, EEG_CH, EEG_TIME)
        out = branch(x)
        assert out.shape == (BATCH, 128)

    def test_ecg_input(self):
        from src.models.deep_model import CNNLSTMBranch
        branch = CNNLSTMBranch(in_channels=ECG_CH, out_dim=64)
        x   = torch.randn(BATCH, ECG_CH, ECG_TIME)
        out = branch(x)
        assert out.shape == (BATCH, 64)


@SKIP_TORCH
class TestFusionModel:

    @pytest.fixture
    def model(self):
        from src.models.deep_model import FusionModel
        return FusionModel(
            n_eeg_channels=EEG_CH, n_ecg_channels=ECG_CH,
            branch_type="cnn", branch_dim=64, n_classes=N_CLASSES,
        )

    def test_output_shape(self, model):
        eeg = torch.randn(BATCH, EEG_TIME, EEG_CH)
        ecg = torch.randn(BATCH, ECG_TIME, ECG_CH)
        out = model(eeg, ecg)
        assert out.shape == (BATCH, N_CLASSES)

    def test_no_nan_output(self, model):
        eeg = torch.randn(BATCH, EEG_TIME, EEG_CH)
        ecg = torch.randn(BATCH, ECG_TIME, ECG_CH)
        out = model(eeg, ecg)
        assert not torch.isnan(out).any()

    def test_cnnlstm_branch(self):
        from src.models.deep_model import FusionModel
        model = FusionModel(branch_type="cnnlstm", branch_dim=64)
        eeg = torch.randn(BATCH, EEG_TIME, EEG_CH)
        ecg = torch.randn(BATCH, ECG_TIME, ECG_CH)
        out = model(eeg, ecg)
        assert out.shape == (BATCH, N_CLASSES)

    def test_attention_weights_sum_to_one(self, model):
        attn_out = {}
        def hook(m, inp, out):
            attn_out["w"] = out
        model.attention.register_forward_hook(hook)
        eeg = torch.randn(BATCH, EEG_TIME, EEG_CH)
        ecg = torch.randn(BATCH, ECG_TIME, ECG_CH)
        model(eeg, ecg)
        w = attn_out["w"]
        assert w.shape == (BATCH, 2)
        assert torch.allclose(w.sum(dim=1), torch.ones(BATCH), atol=1e-5)

    def test_batch_size_1(self, model):
        eeg = torch.randn(1, EEG_TIME, EEG_CH)
        ecg = torch.randn(1, ECG_TIME, ECG_CH)
        out = model(eeg, ecg)
        assert out.shape == (1, N_CLASSES)

    def test_gradients_flow(self, model):
        eeg  = torch.randn(BATCH, EEG_TIME, EEG_CH)
        ecg  = torch.randn(BATCH, ECG_TIME, ECG_CH)
        out  = model(eeg, ecg)
        loss = out.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient: {name}"
                assert not torch.isnan(p.grad).any(), f"NaN gradient: {name}"

    def test_param_count_reasonable(self, model):
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert 50_000 < n < 5_000_000, f"Unexpected param count: {n:,}"


@SKIP_TORCH
class TestBuildModelFactory:

    def _config(self):
        return {
            "data"    : {"segment_length": 4, "sampling_rate_eeg": 128,
                         "sampling_rate_ecg": 256},
            "model"   : {"dropout": 0.3, "branch_dim": 64},
            "training": {},
        }

    @pytest.mark.parametrize("model_type", ["eegnet", "cnn", "cnnlstm", "fusion"])
    def test_factory_returns_module(self, model_type):
        from src.models.deep_model import build_model
        model = build_model(model_type, self._config())
        assert isinstance(model, nn.Module)

    def test_factory_invalid_raises(self):
        from src.models.deep_model import build_model
        with pytest.raises(ValueError, match="Unknown model_type"):
            build_model("transformer_v99", self._config())

    @pytest.mark.parametrize("model_type", ["cnn", "cnnlstm", "fusion"])
    def test_factory_forward_pass(self, model_type):
        from src.models.deep_model import build_model
        model = build_model(model_type, self._config())
        eeg   = torch.randn(2, EEG_TIME, EEG_CH)
        ecg   = torch.randn(2, ECG_TIME, ECG_CH)
        out   = model(eeg, ecg)
        assert out.shape == (2, N_CLASSES)