"""
ecg_features.py
Responsible for: Extracting HRV and statistical features from ECG segments.
"""

import numpy as np
from scipy.signal import welch, find_peaks, butter, filtfilt
from scipy.stats import skew, kurtosis
from typing import Tuple, List


# ── R-peak detection ──────────────────────────────────────────────────────────

def detect_r_peaks(ecg_channel: np.ndarray, fs: float = 256.0) -> np.ndarray:
    """
    Detect R-peaks in a single ECG channel.

    Args:
        ecg_channel: (samples,)
        fs         : ECG sampling rate

    Returns:
        r_peaks: indices of R-peaks
    """
    # Light bandpass 5–15 Hz to emphasise QRS
    nyq = fs / 2.0
    b, a = butter(2, [5 / nyq, 15 / nyq], btype="band")
    filtered = filtfilt(b, a, ecg_channel)

    min_distance = int(0.4 * fs)           # 400 ms refractory period
    height_thresh = np.std(filtered) * 0.5

    peaks, _ = find_peaks(filtered,
                           distance=min_distance,
                           height=height_thresh)
    return peaks


# ── HRV time-domain features ──────────────────────────────────────────────────

def hrv_time_domain(r_peaks: np.ndarray, fs: float = 256.0) -> np.ndarray:
    """
    HRV time-domain features from R-peaks.

    Returns:
        features: (8,)
        [mean_rr, std_rr, rmssd, nn50, pnn50, mean_hr, std_hr, cv_rr]
    """
    if len(r_peaks) < 3:
        return np.zeros(8, dtype=np.float32)

    rr      = np.diff(r_peaks) / fs * 1000   # ms
    hr      = 60000.0 / rr                    # bpm
    diff_rr = np.diff(rr)

    nn50  = int(np.sum(np.abs(diff_rr) > 50))
    pnn50 = nn50 / len(diff_rr) if len(diff_rr) > 0 else 0.0
    rmssd = float(np.sqrt(np.mean(diff_rr ** 2)))

    feats = np.array([
        np.mean(rr),                          # mean RR (ms)
        np.std(rr),                           # SDNN
        rmssd,                                # RMSSD
        float(nn50),                          # NN50
        pnn50,                                # pNN50
        np.mean(hr),                          # mean HR (bpm)
        np.std(hr),                           # HR std
        np.std(rr) / (np.mean(rr) + 1e-10),  # CV_RR
    ], dtype=np.float32)

    return feats


# ── HRV frequency-domain features ────────────────────────────────────────────

def hrv_frequency_domain(r_peaks: np.ndarray, fs: float = 256.0) -> np.ndarray:
    """
    HRV frequency-domain features (LF, HF, LF/HF).

    Returns:
        features: (4,)
        [lf_power, hf_power, lf_hf_ratio, total_power]
    """
    if len(r_peaks) < 4:
        return np.zeros(4, dtype=np.float32)

    rr      = np.diff(r_peaks) / fs          # seconds
    rr_ms   = rr * 1000

    # Uniform resampling of RR series at 4 Hz for spectral analysis
    fs_re   = 4.0
    t_orig  = np.cumsum(rr)
    t_re    = np.arange(t_orig[0], t_orig[-1], 1.0 / fs_re)
    rr_re   = np.interp(t_re, t_orig, rr_ms)

    if len(rr_re) < 8:
        return np.zeros(4, dtype=np.float32)

    freqs, psd = welch(rr_re, fs=fs_re, nperseg=min(len(rr_re), 64))

    lf_mask  = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask  = (freqs >= 0.15) & (freqs < 0.40)

    lf_power = float(np.trapezoid(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else 0.0
    hf_power = float(np.trapezoid(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else 0.0
    total    = lf_power + hf_power + 1e-10
    lf_hf    = lf_power / (hf_power + 1e-10)

    return np.array([lf_power, hf_power, lf_hf, total], dtype=np.float32)


# ── Signal statistical features ───────────────────────────────────────────────

def ecg_statistical_features(segment: np.ndarray) -> np.ndarray:
    """
    Basic statistical features per ECG channel.

    Args:
        segment: (samples, 2)

    Returns:
        features: (2 × 5,) = 10-dim
        [mean, std, rms, skewness, kurtosis] per channel
    """
    feats = []
    for ch in range(segment.shape[1]):
        sig = segment[:, ch]
        feats.extend([
            float(np.mean(sig)),
            float(np.std(sig)),
            float(np.sqrt(np.mean(sig ** 2))),
            float(skew(sig)),
            float(kurtosis(sig)),
        ])
    return np.array(feats, dtype=np.float32)


# ── Combined ECG feature vector ───────────────────────────────────────────────

def extract_ecg_features(segment: np.ndarray, fs: float = 256.0) -> np.ndarray:
    """
    Concatenate all ECG features into one vector.

    Composition:
        hrv_time_domain   :  8  (per segment, using ch0)
        hrv_freq_domain   :  4
        ecg_statistical   : 10  (2 ch × 5 stats)
        ──────────────────────
        TOTAL             : 22 features

    Args:
        segment: (samples, 2)
        fs     : ECG sampling rate

    Returns:
        features: (22,) float32
    """
    r_peaks = detect_r_peaks(segment[:, 0], fs)
    return np.concatenate([
        hrv_time_domain(r_peaks, fs),
        hrv_frequency_domain(r_peaks, fs),
        ecg_statistical_features(segment),
    ])


def feature_names() -> List[str]:
    """Return all ECG feature names."""
    names  = ["ecg_mean_rr", "ecg_sdnn", "ecg_rmssd", "ecg_nn50",
              "ecg_pnn50", "ecg_mean_hr", "ecg_std_hr", "ecg_cv_rr"]
    names += ["ecg_lf_power", "ecg_hf_power", "ecg_lf_hf", "ecg_total_power"]
    for ch in ["ch1", "ch2"]:
        for s in ["mean", "std", "rms", "skewness", "kurtosis"]:
            names.append(f"ecg_{ch}_{s}")
    return names