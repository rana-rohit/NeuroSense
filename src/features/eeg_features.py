import numpy as np
from scipy.signal import welch

def bandpower(signal, fs, band):
    f, Pxx = welch(signal, fs=fs, nperseg=min(len(signal), 256))
    idx = (f >= band[0]) & (f <= band[1])
    return np.trapz(Pxx[idx], f[idx]) if idx.any() else 0.0


def extract_eeg_features(eeg_segment, fs=128.0):
    """
    Improved EEG features
    Input: (samples, 14)
    Output: (~200 features)
    """

    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta" : (13, 30),
    }

    features = []

    # ── 1. Bandpower per channel ──
    band_feats = []
    for ch in range(eeg_segment.shape[1]):
        sig = eeg_segment[:, ch]

        for band in bands.values():
            bp = bandpower(sig, fs, band)
            band_feats.append(np.log(bp + 1e-8))  # log for stability

    features.extend(band_feats)

    # ── 2. Statistical features (keep but reduced importance) ──
    for ch in range(eeg_segment.shape[1]):
        sig = eeg_segment[:, ch]
        features.extend([
            np.mean(sig),
            np.std(sig),
        ])

    # ── 3. Asymmetry (left vs right hemisphere) ──
    # simple pairs (approx)
    pairs = [(0, 13), (1, 12), (2, 11), (3, 10), (4, 9), (5, 8)]

    asym_feats = []
    for (l, r) in pairs:
        for i in range(len(bands)):
            left  = band_feats[l*4 + i]
            right = band_feats[r*4 + i]
            asym_feats.append(left - right)

    features.extend(asym_feats)

    return np.array(features, dtype=np.float32)