"""
test_models.py
Unit tests for deep model forward passes and trainer smoke tests.
Run: python -m pytest tests/test_models.py -v
"""

import pytest
import torch
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))

from src.models.deep_model import (
    EEGNet, CNN1DBranch, CNNLSTMBranch, FusionModel, build_model
)

BATCH      = 4
EEG_TIME   = 512    # 4s × 128 Hz
ECG_TIME   = 1024   # 4s × 256 Hz
EEG_CH     = 14
ECG_CH     = 2
N_CLASSES  = 2


# ── EEGNet ────────────────────────────────────────────────────────────────────

def test_eegnet_output_shape():
    model  = EEGNet(n_channels=EEG_CH, n_samples=EEG_TIME, n_classes=N_CLASSES)
    x      = torch.randn(BATCH, EEG_TIME, EEG_CH)
    out    = model(x)
    assert out.shape == (BATCH, N_CLASSES)

def test_eegnet_no_nan():
    model = EEGNet(n_channels=EEG_CH, n_samples=EEG_TIME)
    x     = torch.randn(BATCH, EEG_TIME, EEG_CH)
    out   = model(x)
    assert not torch.isnan(out).any()


# ── CNN1DBranch ───────────────────────────────────────────────────────────────

def test_cnn1d_eeg_shape():
    branch = CNN1DBranch(in_channels=EEG_CH, out_dim=128)
    x      = torch.randn(BATCH, EEG_CH, EEG_TIME)
    out    = branch(x)
    assert out.shape == (BATCH, 128)

def test_cnn1d_ecg_shape():
    branch = CNN1DBranch(in_channels=ECG_CH, out_dim=64)
    x      = torch.randn(BATCH, ECG_CH, ECG_TIME)
    out    = branch(x)
    assert out.shape == (BATCH, 64)


# ── CNNLSTMBranch ─────────────────────────────────────────────────────────────

def test_cnnlstm_eeg_shape():
    branch = CNNLSTMBranch(in_channels=EEG_CH, out_dim=128)
    x      = torch.randn(BATCH, EEG_CH, EEG_TIME)
    out    = branch(x)
    assert out.shape == (BATCH, 128)

def test_cnnlstm_ecg_shape():
    branch = CNNLSTMBranch(in_channels=ECG_CH, out_dim=64)
    x      = torch.randn(BATCH, ECG_CH, ECG_TIME)
    out    = branch(x)
    assert out.shape == (BATCH, 64)


# ── FusionModel ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("branch_type", ["cnn", "cnnlstm"])
def test_fusion_output_shape(branch_type):
    model = FusionModel(
        n_eeg_channels=EEG_CH,
        n_ecg_channels=ECG_CH,
        branch_type=branch_type,
        branch_dim=64,
        n_classes=N_CLASSES,
    )
    eeg = torch.randn(BATCH, EEG_TIME, EEG_CH)
    ecg = torch.randn(BATCH, ECG_TIME, ECG_CH)
    out = model(eeg, ecg)
    assert out.shape == (BATCH, N_CLASSES)

@pytest.mark.parametrize("branch_type", ["cnn", "cnnlstm"])
def test_fusion_no_nan(branch_type):
    model = FusionModel(
        n_eeg_channels=EEG_CH,
        n_ecg_channels=ECG_CH,
        branch_type=branch_type,
        branch_dim=64,
    )
    eeg = torch.randn(BATCH, EEG_TIME, EEG_CH)
    ecg = torch.randn(BATCH, ECG_TIME, ECG_CH)
    out = model(eeg, ecg)
    assert not torch.isnan(out).any()

def test_fusion_attention_weights_sum_to_one():
    model = FusionModel(branch_dim=64)
    eeg   = torch.randn(BATCH, EEG_TIME, EEG_CH)
    ecg   = torch.randn(BATCH, ECG_TIME, ECG_CH)

    # Hook to capture attention output
    attn_out = {}
    def hook(m, inp, out):
        attn_out["w"] = out

    model.attention.register_forward_hook(hook)
    model(eeg, ecg)

    w = attn_out["w"]
    assert w.shape == (BATCH, 2)
    assert torch.allclose(w.sum(dim=1), torch.ones(BATCH), atol=1e-5)


# ── build_model factory ───────────────────────────────────────────────────────

@pytest.mark.parametrize("model_type", ["eegnet", "cnn", "cnnlstm", "fusion"])
def test_build_model_factory(model_type):
    config = {
        "data"    : {
            "segment_length"   : 4,
            "sampling_rate_eeg": 128,
            "sampling_rate_ecg": 256,
        },
        "model"   : {"dropout": 0.3, "branch_dim": 64},
        "training": {"batch_size": 4},
    }
    model = build_model(model_type, config)
    assert isinstance(model, torch.nn.Module)

def test_build_model_invalid():
    config = {
        "data"    : {"segment_length": 4, "sampling_rate_eeg": 128,
                     "sampling_rate_ecg": 256},
        "model"   : {"dropout": 0.3, "branch_dim": 64},
        "training": {},
    }
    with pytest.raises(ValueError):
        build_model("transformer_xyz", config)


# ── Parameter count sanity ────────────────────────────────────────────────────

def test_fusion_param_count():
    model  = FusionModel(branch_dim=128)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Should be between 100K and 5M for this config
    assert 100_000 < n_params < 5_000_000, \
        f"Unexpected param count: {n_params:,}"


# ── Gradient flow check ───────────────────────────────────────────────────────

def test_gradients_flow_fusion():
    model = FusionModel(branch_dim=64)
    eeg   = torch.randn(BATCH, EEG_TIME, EEG_CH, requires_grad=False)
    ecg   = torch.randn(BATCH, ECG_TIME, ECG_CH, requires_grad=False)

    out  = model(eeg, ecg)
    loss = out.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, \
                f"No gradient for: {name}"
            assert not torch.isnan(param.grad).any(), \
                f"NaN gradient for: {name}"