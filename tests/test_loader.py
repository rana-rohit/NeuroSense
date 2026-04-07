"""
test_loader.py
Basic unit tests for loader and preprocessor.
Run: python -m pytest tests/ -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocessor import (
    bandpass_filter, notch_filter, normalize_signal,
    baseline_correction, segment_signal, process_trial
)
from src.features.eeg_features import extract_eeg_features, feature_names as eeg_names
from src.features.ecg_features import extract_ecg_features, feature_names as ecg_names


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_eeg():
    """Synthetic EEG: 60s at 128Hz, 14 channels"""
    np.random.seed(42)
    return np.random.randn(60 * 128, 14).astype(np.float32)

@pytest.fixture
def dummy_ecg():
    """Synthetic ECG: 60s at 256Hz, 2 channels"""
    np.random.seed(42)
    return np.random.randn(60 * 256, 2).astype(np.float32)


# ── Preprocessor tests ────────────────────────────────────────────────────────

def test_bandpass_shape(dummy_eeg):
    out = bandpass_filter(dummy_eeg, fs=128.0)
    assert out.shape == dummy_eeg.shape

def test_bandpass_1d():
    sig = np.random.randn(1024).astype(np.float32)
    out = bandpass_filter(sig, fs=128.0)
    assert out.shape == sig.shape

def test_notch_shape(dummy_eeg):
    out = notch_filter(dummy_eeg, fs=128.0, freq=50.0)
    assert out.shape == dummy_eeg.shape

def test_normalize_zscore(dummy_eeg):
    out = normalize_signal(dummy_eeg, method="zscore")
    assert out.shape == dummy_eeg.shape
    assert abs(out.mean()) < 0.1

def test_normalize_minmax(dummy_eeg):
    out = normalize_signal(dummy_eeg, method="minmax")
    assert out.min() >= -0.01
    assert out.max() <= 1.01

def test_normalize_invalid():
    with pytest.raises(ValueError):
        normalize_signal(np.random.randn(100, 14).astype(np.float32), method="bad")

def test_baseline_correction(dummy_eeg):
    baseline = np.random.randn(10 * 128, 14).astype(np.float32)
    out = baseline_correction(dummy_eeg, baseline)
    assert out.shape == dummy_eeg.shape

def test_segment_shape(dummy_eeg):
    segs = segment_signal(dummy_eeg, fs=128.0, window_sec=4.0, overlap_sec=2.0)
    assert segs.ndim == 3
    assert segs.shape[1] == 4 * 128
    assert segs.shape[2] == 14

def test_segment_count(dummy_eeg):
    # 60s signal, 4s window, 2s step → ~29 windows
    segs = segment_signal(dummy_eeg, fs=128.0, window_sec=4.0, overlap_sec=2.0)
    assert segs.shape[0] >= 28

def test_process_trial(dummy_eeg, dummy_ecg):
    base_eeg = np.random.randn(10 * 128, 14).astype(np.float32)
    base_ecg = np.random.randn(10 * 256, 2).astype(np.float32)
    eeg_s, ecg_s = process_trial(dummy_eeg, dummy_ecg, base_eeg, base_ecg)
    assert eeg_s.ndim == 3
    assert ecg_s.ndim == 3
    assert eeg_s.shape[0] == ecg_s.shape[0]   # windows aligned


# ── Feature tests ─────────────────────────────────────────────────────────────

def test_eeg_feature_shape(dummy_eeg):
    seg  = dummy_eeg[:4*128]            # 4s segment
    feat = extract_eeg_features(seg, fs=128.0)
    assert feat.ndim == 1
    assert len(feat) == 258

def test_eeg_feature_names():
    assert len(eeg_names()) == 258

def test_ecg_feature_shape(dummy_ecg):
    seg  = dummy_ecg[:4*256]
    feat = extract_ecg_features(seg, fs=256.0)
    assert feat.ndim == 1
    assert len(feat) == 22

def test_ecg_feature_names():
    assert len(ecg_names()) == 22

def test_no_nan_eeg(dummy_eeg):
    seg  = dummy_eeg[:4*128]
    feat = extract_eeg_features(seg, fs=128.0)
    assert not np.any(np.isnan(feat))

def test_no_nan_ecg(dummy_ecg):
    seg  = dummy_ecg[:4*256]
    feat = extract_ecg_features(seg, fs=256.0)
    assert not np.any(np.isnan(feat))