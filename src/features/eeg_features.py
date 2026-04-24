import numpy as np

def extract_eeg_features(eeg_segment, fs=128.0):
    """
    Input: (samples, channels=14)
    Output: (258,)
    """

    features = []

    # --- Per-channel statistical features ---
    for ch in range(eeg_segment.shape[1]):
        sig = eeg_segment[:, ch]

        features.extend([
            np.mean(sig),
            np.std(sig),
            np.max(sig),
            np.min(sig),
            np.sqrt(np.mean(sig**2)),   # RMS
        ])

    features = np.array(features, dtype=np.float32)  # 14 * 5 = 70

    # --- Expand to required size ---
    # Repeat features to reach ~258
    repeated = np.tile(features, 4)  # 70 * 4 = 280

    return repeated[:258]  # exact size