import numpy as np
from scipy.signal import welch
from scipy.stats import entropy


def bandpower(signal, fs, band):
    f, Pxx = welch(signal, fs=fs, nperseg=min(len(signal), 256))
    idx = (f >= band[0]) & (f <= band[1])
    return np.trapz(Pxx[idx], f[idx]) if idx.any() else 0.0


def extract_eeg_features(eeg_segment, fs=128.0):
    """
    Improved EEG features
    Input: (samples, 14)
    Output: (~240 features)
    """

    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
    }

    features = []

    # ── 1. Bandpower per channel ──
    band_feats = []
    for ch in range(eeg_segment.shape[1]):
        sig = eeg_segment[:, ch]

        for band in bands.values():
            bp = bandpower(sig, fs, band)
            band_feats.append(np.log(bp + 1e-8))  # log stability

    features.extend(band_feats)

    # ── 2. Relative features (VERY IMPORTANT for generalization) ──
    rel_feats = []
    for ch in range(eeg_segment.shape[1]):
        delta = band_feats[ch*4 + 0]
        theta = band_feats[ch*4 + 1]
        alpha = band_feats[ch*4 + 2]
        beta  = band_feats[ch*4 + 3]

        rel_feats.extend([
            alpha / (beta + 1e-6),
            theta / (alpha + 1e-6),
            (alpha + theta) / (beta + 1e-6),
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

        hist, _ = np.histogram(sig, bins=20, density=True)
        hist = hist / (np.sum(hist) + 1e-8)
        ent = entropy(hist)
        features.append(np.log(ent + 1e-8))

    # ── 5. Asymmetry features ──
    pairs = [(0, 13), (1, 12), (2, 11), (3, 10), (4, 9), (5, 8)]

    asym_feats = []
    for (l, r) in pairs:
        for i in range(len(bands)):
            left  = band_feats[l*4 + i]
            right = band_feats[r*4 + i]
            asym_feats.append(left - right)

    features.extend(asym_feats)

    features = np.array(features, dtype=np.float32)

    # ── Safety: remove NaN / Inf ──
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # ── Stability: clamp extreme values ──
    features = np.clip(features, -10, 10)

    return features