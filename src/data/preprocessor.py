"""
preprocessor.py
Responsible for: Filtering, segmentation, normalisation of EEG & ECG signals.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from typing import Optional, List, Tuple


# ── EEG Preprocessing ─────────────────────────────────────────────────────────

def bandpass_filter(signal: np.ndarray, fs: float,
                    low: float = 0.5, high: float = 45.0,
                    order: int = 4) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter.

    Args:
        signal : (samples,) or (samples, channels)
        fs     : sampling frequency in Hz
        low    : lower cutoff (Hz)
        high   : upper cutoff (Hz)
        order  : filter order

    Returns:
        filtered signal, same shape as input
    """
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    if signal.ndim == 1:
        return filtfilt(b, a, signal).astype(np.float32)
    return np.apply_along_axis(lambda x: filtfilt(b, a, x), 0, signal).astype(np.float32)


def notch_filter(signal: np.ndarray, fs: float,
                 freq: float = 50.0, quality: float = 30.0) -> np.ndarray:
    """
    Apply notch filter to remove powerline noise (50 Hz or 60 Hz).

    Args:
        signal : (samples,) or (samples, channels)
        fs     : sampling frequency
        freq   : notch frequency (Hz)
        quality: Q-factor (higher = narrower notch)
    """
    b, a = iirnotch(freq / (fs / 2.0), quality)
    if signal.ndim == 1:
        return filtfilt(b, a, signal).astype(np.float32)
    return np.apply_along_axis(lambda x: filtfilt(b, a, x), 0, signal).astype(np.float32)


def preprocess_eeg(eeg: np.ndarray, fs: float = 128.0,
                   bp_low: float = 0.5, bp_high: float = 45.0,
                   notch_freq: Optional[float] = 50.0) -> np.ndarray:
    """
    Full EEG preprocessing pipeline.
        1. Bandpass filter (0.5–45 Hz)
        2. Notch filter    (50 Hz)

    Args:
        eeg       : (samples, 14)
        fs        : EEG sampling rate
        bp_low    : bandpass low cutoff
        bp_high   : bandpass high cutoff
        notch_freq: set None to skip notch

    Returns:
        eeg_clean : (samples, 14)
    """
    eeg_f = bandpass_filter(eeg, fs, low=bp_low, high=bp_high)
    if notch_freq is not None:
        eeg_f = notch_filter(eeg_f, fs, freq=notch_freq)
    return eeg_f


# ── ECG Preprocessing ─────────────────────────────────────────────────────────

def preprocess_ecg(ecg: np.ndarray, fs: float = 256.0,
                   bp_low: float = 0.5, bp_high: float = 40.0) -> np.ndarray:
    """
    Full ECG preprocessing pipeline.
        1. Bandpass filter (0.5–40 Hz)

    Args:
        ecg : (samples, 2)
        fs  : ECG sampling rate

    Returns:
        ecg_clean : (samples, 2)
    """
    return bandpass_filter(ecg, fs, low=bp_low, high=bp_high)


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize_signal(signal: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Normalize signal per channel.

    Args:
        signal : (samples, channels) or (samples,)
        method : 'zscore' | 'minmax'

    Returns:
        normalized signal, same shape
    """
    if method == "zscore":
        mean = signal.mean(axis=0, keepdims=True)
        std  = signal.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        return ((signal - mean) / std).astype(np.float32)

    elif method == "minmax":
        mn = signal.min(axis=0, keepdims=True)
        mx = signal.max(axis=0, keepdims=True)
        rng = mx - mn
        rng[rng == 0] = 1.0
        return ((signal - mn) / rng).astype(np.float32)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


# ── Baseline Correction ───────────────────────────────────────────────────────

def baseline_correction(stimuli: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """
    Subtract per-channel mean of baseline from stimuli signal.

    Args:
        stimuli  : (samples_s, channels)
        baseline : (samples_b, channels)

    Returns:
        corrected: (samples_s, channels)
    """
    baseline_mean = baseline.mean(axis=0, keepdims=True)
    return (stimuli - baseline_mean).astype(np.float32)


# ── Segmentation ──────────────────────────────────────────────────────────────

def segment_signal(signal: np.ndarray, fs: float,
                   window_sec: float = 4.0,
                   overlap_sec: float = 2.0) -> np.ndarray:
    """
    Segment a signal into overlapping windows.

    Args:
        signal      : (samples, channels)
        fs          : sampling frequency
        window_sec  : window length in seconds
        overlap_sec : overlap between windows in seconds

    Returns:
        segments: (n_windows, window_samples, channels)
    """
    window_samples  = int(window_sec  * fs)
    step_samples    = int((window_sec - overlap_sec) * fs)

    if step_samples <= 0:
        raise ValueError("overlap_sec must be less than window_sec")

    n_samples = signal.shape[0]
    segments  = []

    start = 0
    while start + window_samples <= n_samples:
        segments.append(signal[start : start + window_samples])
        start += step_samples

    if len(segments) == 0:
        raise ValueError(
            f"Signal too short ({n_samples} samples) for window "
            f"of {window_samples} samples at fs={fs}"
        )

    return np.stack(segments, axis=0).astype(np.float32)


# ── Full Trial Pipeline ───────────────────────────────────────────────────────

def process_trial(eeg_stim: np.ndarray, ecg_stim: np.ndarray,
                  eeg_base: np.ndarray, ecg_base: np.ndarray,
                  eeg_fs: float = 128.0, ecg_fs: float = 256.0,
                  window_sec: float = 4.0, overlap_sec: float = 2.0,
                  norm_method: str = "zscore") -> Tuple[np.ndarray, np.ndarray]:
    """
    End-to-end preprocessing for one EEG+ECG trial.

    Steps:
        1. Filter EEG (bandpass + notch)
        2. Filter ECG (bandpass)
        3. Baseline correction (EEG + ECG)
        4. Normalise
        5. Segment into windows

    Args:
        eeg_stim   : raw EEG stimuli  (samples, 14)
        ecg_stim   : raw ECG stimuli  (samples, 2)
        eeg_base   : raw EEG baseline (samples, 14)
        ecg_base   : raw ECG baseline (samples, 2)
        eeg_fs     : EEG sampling rate
        ecg_fs     : ECG sampling rate
        window_sec : segment window length (s)
        overlap_sec: segment overlap (s)
        norm_method: 'zscore' or 'minmax'

    Returns:
        eeg_segments : (n_windows, window_samples_eeg, 14)
        ecg_segments : (n_windows, window_samples_ecg, 2)
    """
    # 1. Filter
    eeg_f = preprocess_eeg(eeg_stim, fs=eeg_fs)
    ecg_f = preprocess_ecg(ecg_stim, fs=ecg_fs)

    eeg_bf = preprocess_eeg(eeg_base, fs=eeg_fs)
    ecg_bf = preprocess_ecg(ecg_base, fs=ecg_fs)

    # ── NEW IMPROVED NORMALIZATION PIPELINE ──

    # 2. Normalize baseline separately (IMPORTANT)
    eeg_bf_n = normalize_signal(eeg_bf, method=norm_method)
    ecg_bf_n = normalize_signal(ecg_bf, method=norm_method)

    eeg_f_n  = normalize_signal(eeg_f,  method=norm_method)
    ecg_f_n  = normalize_signal(ecg_f,  method=norm_method)

    # 3. Baseline correction (on normalized signals)
    eeg_c = baseline_correction(eeg_f_n, eeg_bf_n)
    ecg_c = baseline_correction(ecg_f_n, ecg_bf_n)

    # 4. Final normalization (stabilize)
    eeg_n = normalize_signal(eeg_c, method=norm_method)
    ecg_n = normalize_signal(ecg_c, method=norm_method)

    # 4. Segment
    eeg_seg = segment_signal(eeg_n, fs=eeg_fs,
                              window_sec=window_sec, overlap_sec=overlap_sec)
    ecg_seg = segment_signal(ecg_n, fs=ecg_fs,
                              window_sec=window_sec, overlap_sec=overlap_sec)

    # Align number of windows (take minimum)
    n_win = min(eeg_seg.shape[0], ecg_seg.shape[0])
    return eeg_seg[:n_win], ecg_seg[:n_win]