"""
tests/test_loader.py
Unit tests for src/data/loader.py and src/data/preprocessor.py
Run: python -m pytest tests/test_loader.py -v
"""

import pytest
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))

from src.data.preprocessor import (
    bandpass_filter, notch_filter, normalize_signal,
    baseline_correction, segment_signal, process_trial,
)
from src.data.loader import (
    get_subject_data, get_trial_signals, get_trial_labels,
    build_trial_index, EEG_CHANNELS, LABEL_COLS,
)

EEG_FS, ECG_FS = 128, 256


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def dummy_eeg():
    np.random.seed(42)
    return np.random.randn(60 * EEG_FS, 14).astype(np.float32)

@pytest.fixture
def dummy_ecg():
    np.random.seed(42)
    return np.random.randn(60 * ECG_FS, 2).astype(np.float32)

@pytest.fixture
def dummy_eeg_base():
    return np.random.randn(10 * EEG_FS, 14).astype(np.float32)

@pytest.fixture
def dummy_ecg_base():
    return np.random.randn(10 * ECG_FS, 2).astype(np.float32)

@pytest.fixture
def fake_dreamer():
    """Minimal DREAMER-like dict structure for loader tests."""
    N_S, N_V = 3, 4

    def make_subject(sub_idx):
        return {
            "ScoreValence"  : np.array([1., 2., 3., 4.][:N_V]),
            "ScoreArousal"  : np.array([3., 4., 2., 5.][:N_V]),
            "ScoreDominance": np.array([2., 3., 4., 1.][:N_V]),
            "EEG": {
                "stimuli" : [np.random.randn(60*EEG_FS, 14).astype(np.float32)] * N_V,
                "baseline": [np.random.randn(10*EEG_FS, 14).astype(np.float32)] * N_V,
            },
            "ECG": {
                "stimuli" : [np.random.randn(60*ECG_FS, 2).astype(np.float32)] * N_V,
                "baseline": [np.random.randn(10*ECG_FS, 2).astype(np.float32)] * N_V,
            },
        }

    return {
        "EEG_SamplingRate"   : 128,
        "ECG_SamplingRate"   : 256,
        "noOfSubjects"       : N_S,
        "noOfVideoSequences" : N_V,
        "EEG_Electrodes"     : EEG_CHANNELS,
        "Data"               : [make_subject(i) for i in range(N_S)],
        "_N_S": N_S,
        "_N_V": N_V,
    }


# ══════════════════════════════════════════════════════════════════
# PREPROCESSOR TESTS
# ══════════════════════════════════════════════════════════════════

class TestBandpassFilter:

    def test_output_shape_2d(self, dummy_eeg):
        out = bandpass_filter(dummy_eeg, EEG_FS)
        assert out.shape == dummy_eeg.shape

    def test_output_shape_1d(self):
        sig = np.random.randn(1024).astype(np.float32)
        out = bandpass_filter(sig, EEG_FS)
        assert out.shape == sig.shape

    def test_dtype_float32(self, dummy_eeg):
        out = bandpass_filter(dummy_eeg, EEG_FS)
        assert out.dtype == np.float32

    def test_no_nan(self, dummy_eeg):
        out = bandpass_filter(dummy_eeg, EEG_FS)
        assert not np.any(np.isnan(out))

    def test_attenuates_high_freq(self):
        """Signal above cutoff should be attenuated."""
        fs    = 128.0
        t     = np.arange(0, 10, 1/fs)
        # Pure 60 Hz tone — well above 45 Hz cutoff
        sig   = np.sin(2 * np.pi * 60 * t).astype(np.float32)
        filt  = bandpass_filter(sig, fs, high=45.0)
        # RMS of filtered should be << RMS of original
        assert np.sqrt(np.mean(filt**2)) < 0.1 * np.sqrt(np.mean(sig**2))

    def test_custom_cutoffs(self, dummy_eeg):
        out = bandpass_filter(dummy_eeg, EEG_FS, low=1.0, high=30.0)
        assert out.shape == dummy_eeg.shape


class TestNotchFilter:

    def test_output_shape(self, dummy_eeg):
        out = notch_filter(dummy_eeg, EEG_FS, freq=50.0)
        assert out.shape == dummy_eeg.shape

    def test_1d_input(self):
        sig = np.random.randn(1024).astype(np.float32)
        out = notch_filter(sig, EEG_FS)
        assert out.shape == sig.shape

    def test_no_nan(self, dummy_eeg):
        out = notch_filter(dummy_eeg, EEG_FS)
        assert not np.any(np.isnan(out))

    def test_attenuates_target_frequency(self):
        """50 Hz tone should be attenuated after notch at 50 Hz."""
        fs  = 256.0
        t   = np.arange(0, 5, 1/fs)
        sig = np.sin(2 * np.pi * 50 * t).astype(np.float32)
        out = notch_filter(sig, fs, freq=50.0, quality=30.0)
        assert np.std(out) < 0.2 * np.std(sig)


class TestNormalizeSignal:

    def test_zscore_shape(self, dummy_eeg):
        out = normalize_signal(dummy_eeg, method="zscore")
        assert out.shape == dummy_eeg.shape

    def test_zscore_mean_near_zero(self, dummy_eeg):
        out = normalize_signal(dummy_eeg, method="zscore")
        assert np.abs(out.mean()) < 0.1

    def test_zscore_std_near_one(self, dummy_eeg):
        out = normalize_signal(dummy_eeg, method="zscore")
        assert abs(out.std() - 1.0) < 0.2

    def test_minmax_shape(self, dummy_eeg):
        out = normalize_signal(dummy_eeg, method="minmax")
        assert out.shape == dummy_eeg.shape

    def test_minmax_range(self, dummy_eeg):
        out = normalize_signal(dummy_eeg, method="minmax")
        assert float(out.min()) >= -0.01
        assert float(out.max()) <=  1.01

    def test_no_nan_zscore(self, dummy_eeg):
        out = normalize_signal(dummy_eeg, method="zscore")
        assert not np.any(np.isnan(out))

    def test_no_nan_minmax(self, dummy_eeg):
        out = normalize_signal(dummy_eeg, method="minmax")
        assert not np.any(np.isnan(out))

    def test_flat_channel_no_division_error(self):
        """Flat channel (std=0) should not produce NaN or Inf."""
        sig = np.ones((512, 14), dtype=np.float32)
        sig[:, 0] = 0.0  # one all-zero channel
        out = normalize_signal(sig, method="zscore")
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_invalid_method_raises(self, dummy_eeg):
        with pytest.raises(ValueError, match="Unknown normalization"):
            normalize_signal(dummy_eeg, method="l2norm")


class TestBaselineCorrection:

    def test_output_shape(self, dummy_eeg, dummy_eeg_base):
        out = baseline_correction(dummy_eeg, dummy_eeg_base)
        assert out.shape == dummy_eeg.shape

    def test_subtracts_baseline_mean(self):
        stim = np.ones((512, 14), dtype=np.float32) * 5.0
        base = np.ones((128, 14), dtype=np.float32) * 3.0
        out  = baseline_correction(stim, base)
        assert np.allclose(out, 2.0, atol=1e-5)

    def test_dtype_float32(self, dummy_eeg, dummy_eeg_base):
        out = baseline_correction(dummy_eeg, dummy_eeg_base)
        assert out.dtype == np.float32


class TestSegmentSignal:

    def test_output_ndim(self, dummy_eeg):
        segs = segment_signal(dummy_eeg, EEG_FS, window_sec=4.0, overlap_sec=2.0)
        assert segs.ndim == 3

    def test_window_length(self, dummy_eeg):
        segs = segment_signal(dummy_eeg, EEG_FS, window_sec=4.0, overlap_sec=2.0)
        assert segs.shape[1] == 4 * EEG_FS

    def test_channel_count(self, dummy_eeg):
        segs = segment_signal(dummy_eeg, EEG_FS)
        assert segs.shape[2] == 14

    def test_window_count_correct(self, dummy_eeg):
        # 60s signal, 4s window, 2s step → (60-4)/2 + 1 = 29
        segs = segment_signal(dummy_eeg, EEG_FS, window_sec=4.0, overlap_sec=2.0)
        assert segs.shape[0] == 29

    def test_no_overlap(self, dummy_eeg):
        segs = segment_signal(dummy_eeg, EEG_FS, window_sec=4.0, overlap_sec=0.0)
        assert segs.shape[0] == 15  # 60/4 = 15

    def test_dtype_float32(self, dummy_eeg):
        segs = segment_signal(dummy_eeg, EEG_FS)
        assert segs.dtype == np.float32

    def test_signal_too_short_raises(self):
        short = np.random.randn(10, 14).astype(np.float32)
        with pytest.raises(ValueError, match="too short|Signal too short"):
            segment_signal(short, EEG_FS, window_sec=4.0)

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            segment_signal(np.random.randn(512, 14).astype(np.float32),
                           EEG_FS, window_sec=4.0, overlap_sec=5.0)


class TestProcessTrial:

    def test_output_shapes(self, dummy_eeg, dummy_ecg,
                            dummy_eeg_base, dummy_ecg_base):
        eeg_s, ecg_s = process_trial(
            dummy_eeg, dummy_ecg, dummy_eeg_base, dummy_ecg_base,
            eeg_fs=EEG_FS, ecg_fs=ECG_FS,
        )
        assert eeg_s.ndim == 3
        assert ecg_s.ndim == 3

    def test_channel_dims(self, dummy_eeg, dummy_ecg,
                           dummy_eeg_base, dummy_ecg_base):
        eeg_s, ecg_s = process_trial(
            dummy_eeg, dummy_ecg, dummy_eeg_base, dummy_ecg_base,
        )
        assert eeg_s.shape[2] == 14
        assert ecg_s.shape[2] == 2

    def test_window_counts_aligned(self, dummy_eeg, dummy_ecg,
                                    dummy_eeg_base, dummy_ecg_base):
        eeg_s, ecg_s = process_trial(
            dummy_eeg, dummy_ecg, dummy_eeg_base, dummy_ecg_base,
        )
        assert eeg_s.shape[0] == ecg_s.shape[0]

    def test_no_nan_in_output(self, dummy_eeg, dummy_ecg,
                               dummy_eeg_base, dummy_ecg_base):
        eeg_s, ecg_s = process_trial(
            dummy_eeg, dummy_ecg, dummy_eeg_base, dummy_ecg_base,
        )
        assert not np.any(np.isnan(eeg_s))
        assert not np.any(np.isnan(ecg_s))

    def test_custom_window_overlap(self, dummy_eeg, dummy_ecg,
                                    dummy_eeg_base, dummy_ecg_base):
        eeg_s, ecg_s = process_trial(
            dummy_eeg, dummy_ecg, dummy_eeg_base, dummy_ecg_base,
            window_sec=2.0, overlap_sec=1.0,
        )
        assert eeg_s.shape[1] == 2 * EEG_FS
        assert ecg_s.shape[1] == 2 * ECG_FS

    @pytest.mark.parametrize("method", ["zscore", "minmax"])
    def test_norm_methods(self, dummy_eeg, dummy_ecg,
                           dummy_eeg_base, dummy_ecg_base, method):
        eeg_s, ecg_s = process_trial(
            dummy_eeg, dummy_ecg, dummy_eeg_base, dummy_ecg_base,
            norm_method=method,
        )
        assert eeg_s.shape[2] == 14


# ══════════════════════════════════════════════════════════════════
# LOADER TESTS
# ══════════════════════════════════════════════════════════════════

class TestLoaderFunctions:

    def test_get_subject_data_valid(self, fake_dreamer):
        import src.data.loader as ldr
        orig_n = ldr.N_SUBJECTS
        ldr.N_SUBJECTS = fake_dreamer["_N_S"]
        sub = get_subject_data(fake_dreamer, 0)
        ldr.N_SUBJECTS = orig_n
        assert "EEG" in sub
        assert "ECG" in sub

    def test_get_subject_data_out_of_range(self, fake_dreamer):
        import src.data.loader as ldr
        orig_n = ldr.N_SUBJECTS
        ldr.N_SUBJECTS = fake_dreamer["_N_S"]
        with pytest.raises(IndexError):
            get_subject_data(fake_dreamer, 99)
        ldr.N_SUBJECTS = orig_n

    def test_get_trial_signals_stimuli_shape(self, fake_dreamer):
        import src.data.loader as ldr
        orig_n = ldr.N_SUBJECTS; orig_v = ldr.N_VIDEOS
        ldr.N_SUBJECTS = fake_dreamer["_N_S"]
        ldr.N_VIDEOS   = fake_dreamer["_N_V"]
        sub = get_subject_data(fake_dreamer, 0)
        eeg, ecg = get_trial_signals(sub, 0, mode="stimuli")
        assert eeg.shape == (60 * EEG_FS, 14)
        assert ecg.shape == (60 * ECG_FS, 2)
        ldr.N_SUBJECTS = orig_n; ldr.N_VIDEOS = orig_v

    def test_get_trial_signals_baseline_shape(self, fake_dreamer):
        import src.data.loader as ldr
        orig_n = ldr.N_SUBJECTS; orig_v = ldr.N_VIDEOS
        ldr.N_SUBJECTS = fake_dreamer["_N_S"]
        ldr.N_VIDEOS   = fake_dreamer["_N_V"]
        sub = get_subject_data(fake_dreamer, 0)
        eeg, ecg = get_trial_signals(sub, 0, mode="baseline")
        assert eeg.shape == (10 * EEG_FS, 14)
        assert ecg.shape == (10 * ECG_FS, 2)
        ldr.N_SUBJECTS = orig_n; ldr.N_VIDEOS = orig_v

    def test_get_trial_signals_invalid_mode(self, fake_dreamer):
        import src.data.loader as ldr
        orig_n = ldr.N_SUBJECTS; orig_v = ldr.N_VIDEOS
        ldr.N_SUBJECTS = fake_dreamer["_N_S"]
        ldr.N_VIDEOS   = fake_dreamer["_N_V"]
        sub = get_subject_data(fake_dreamer, 0)
        with pytest.raises(ValueError, match="mode must be"):
            get_trial_signals(sub, 0, mode="invalid")
        ldr.N_SUBJECTS = orig_n; ldr.N_VIDEOS = orig_v

    def test_get_trial_labels_keys(self, fake_dreamer):
        import src.data.loader as ldr
        orig_n = ldr.N_SUBJECTS; orig_v = ldr.N_VIDEOS
        ldr.N_SUBJECTS = fake_dreamer["_N_S"]
        ldr.N_VIDEOS   = fake_dreamer["_N_V"]
        sub    = get_subject_data(fake_dreamer, 0)
        labels = get_trial_labels(sub, 0)
        assert set(labels.keys()) == {"valence", "arousal", "dominance"}
        ldr.N_SUBJECTS = orig_n; ldr.N_VIDEOS = orig_v

    def test_get_trial_labels_float_values(self, fake_dreamer):
        import src.data.loader as ldr
        orig_n = ldr.N_SUBJECTS; orig_v = ldr.N_VIDEOS
        ldr.N_SUBJECTS = fake_dreamer["_N_S"]
        ldr.N_VIDEOS   = fake_dreamer["_N_V"]
        sub    = get_subject_data(fake_dreamer, 0)
        labels = get_trial_labels(sub, 0)
        for k, v in labels.items():
            assert isinstance(v, float), f"{k} not float: {type(v)}"
        ldr.N_SUBJECTS = orig_n; ldr.N_VIDEOS = orig_v

    def test_get_trial_labels_range(self, fake_dreamer):
        """All label values should be in 1–5 range."""
        import src.data.loader as ldr
        orig_n = ldr.N_SUBJECTS; orig_v = ldr.N_VIDEOS
        ldr.N_SUBJECTS = fake_dreamer["_N_S"]
        ldr.N_VIDEOS   = fake_dreamer["_N_V"]
        for sub_idx in range(fake_dreamer["_N_S"]):
            sub = get_subject_data(fake_dreamer, sub_idx)
            for vid_idx in range(fake_dreamer["_N_V"]):
                labels = get_trial_labels(sub, vid_idx)
                for k, v in labels.items():
                    assert 1.0 <= v <= 5.0, f"{k}={v} out of 1-5 range"
        ldr.N_SUBJECTS = orig_n; ldr.N_VIDEOS = orig_v

    def test_build_trial_index_length(self, fake_dreamer):
        import src.data.loader as ldr
        orig_n = ldr.N_SUBJECTS; orig_v = ldr.N_VIDEOS
        ldr.N_SUBJECTS = fake_dreamer["_N_S"]
        ldr.N_VIDEOS   = fake_dreamer["_N_V"]
        trials = build_trial_index(fake_dreamer)
        assert len(trials) == fake_dreamer["_N_S"] * fake_dreamer["_N_V"]
        ldr.N_SUBJECTS = orig_n; ldr.N_VIDEOS = orig_v

    def test_build_trial_index_keys(self, fake_dreamer):
        import src.data.loader as ldr
        orig_n = ldr.N_SUBJECTS; orig_v = ldr.N_VIDEOS
        ldr.N_SUBJECTS = fake_dreamer["_N_S"]
        ldr.N_VIDEOS   = fake_dreamer["_N_V"]
        trials = build_trial_index(fake_dreamer)
        required = {"subject_idx","video_idx","valence","arousal","dominance"}
        for trial in trials:
            assert required.issubset(set(trial.keys()))
        ldr.N_SUBJECTS = orig_n; ldr.N_VIDEOS = orig_v

    def test_label_cols_constant(self):
        assert LABEL_COLS == ["valence", "arousal", "dominance"]

    def test_eeg_channel_count(self):
        assert len(EEG_CHANNELS) == 14