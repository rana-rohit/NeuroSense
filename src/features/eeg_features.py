import numpy as np
from scipy.signal import welch
from scipy.stats import entropy


def bandpower(signal, fs, band):
    f, Pxx = welch(signal, fs=fs, nperseg=128)
    idx = (f >= band[0]) & (f <= band[1])
    return np.trapz(Pxx[idx], f[idx]) if idx.any() else 0.0


def extract_eeg_features(eeg_segment, fs=128.0):
    """
    Improved EEG features
    Input: (samples, 14)
    Output: feature vector
    """

    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
    }

    features = []

    # ── 1. Bandpower per channel (linear + log) ──
    linear_band_feats = []
    log_band_feats = []
    for ch in range(eeg_segment.shape[1]):
        sig = eeg_segment[:, ch]

        for band in bands.values():
            bp = bandpower(sig, fs, band) + 1e-8   # linear
            linear_band_feats.append(bp)
            log_band_feats.append(np.log(bp))       # log for features

    features.extend(log_band_feats)

    # ── 2. Relative features (LINEAR bandpowers for ratios) ──
    rel_feats = []
    for ch in range(eeg_segment.shape[1]):
        delta = linear_band_feats[ch*4 + 0]
        theta = linear_band_feats[ch*4 + 1]
        alpha = linear_band_feats[ch*4 + 2]
        beta  = linear_band_feats[ch*4 + 3]
        total = delta + theta + alpha + beta + 1e-8

        rel_feats.extend([
            alpha / total,
            beta / total,
            np.log(alpha / beta),
            np.log(theta / alpha),
            np.log((alpha + theta) / beta),
        ])

    features.extend(rel_feats)

    # ── 3. Statistical features ──
    for ch in range(eeg_segment.shape[1]):
        sig = eeg_segment[:, ch]
        features.extend([
            np.mean(sig),
            np.std(sig) + 1e-8,
        ])

    # ── 4. Entropy features (AFTER band features → more meaningful) ──
    for ch in range(eeg_segment.shape[1]):
        sig = eeg_segment[:, ch]

        hist, _ = np.histogram(sig, bins=10, density=True)
        hist = hist / (np.sum(hist) + 1e-8)
        ent = entropy(hist)
        features.append(np.log(ent + 1e-8))

    # ── 5. Asymmetry features ──
    pairs = [(0, 13), (1, 12), (2, 11), (3, 10), (4, 9), (5, 8)]
    pairs.append((6, 7))   # O1–O2

    asym_feats = []
    for (l, r) in pairs:
        for i in range(len(bands)):
            left  = log_band_feats[l*4 + i]
            right = log_band_feats[r*4 + i]
            asym_feats.append(left - right)

    features.extend(asym_feats)

    features = np.array(features, dtype=np.float32)

    # ── Safety: remove NaN / Inf ──
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # ── Stability: clamp extreme values ──
    features = np.clip(features, -10, 10)

    return features