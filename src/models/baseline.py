"""
baseline.py
Responsible for: Classical ML baseline using hand-crafted EEG + ECG features.
Models: Logistic Regression, SVM, Random Forest, XGBoost (optional)
"""

import numpy as np
import os
import joblib
from typing import Optional, Tuple, Dict

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report
)

from src.utils.logger import get_logger

logger = get_logger("baseline")


# ── Model registry ────────────────────────────────────────────────────────────

def build_baseline(model_type: str = "svm",
                   random_state: int = 42) -> Pipeline:
    """
    Build a scikit-learn Pipeline: StandardScaler → Classifier.

    Args:
        model_type: 'logreg' | 'svm' | 'rf' | 'gbm'
        random_state: seed

    Returns:
        sklearn Pipeline
    """
    classifiers = {
        "logreg": LogisticRegression(
            max_iter=1000, C=1.0,
            class_weight="balanced",
            random_state=random_state
        ),
        "svm": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=random_state
        ),
        "rf": RandomForestClassifier(
            n_estimators=200, max_depth=None,
            class_weight="balanced",
            random_state=random_state, n_jobs=-1
        ),
        "gbm": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=random_state
        ),
    }

    if model_type not in classifiers:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose from: {list(classifiers.keys())}"
        )

    logger.info(f"Building baseline pipeline: StandardScaler + {model_type.upper()}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    classifiers[model_type]),
    ])


# ── Feature matrix builder ────────────────────────────────────────────────────

def build_feature_matrix(
    eeg_segments: np.ndarray,
    ecg_segments: np.ndarray,
    eeg_fs: float = 128.0,
    ecg_fs: float = 256.0,
) -> np.ndarray:
    """
    Extract features from EEG + ECG segments and concatenate.

    Args:
        eeg_segments : (n_windows, eeg_samples, 14)
        ecg_segments : (n_windows, ecg_samples, 2)
        eeg_fs       : EEG sampling rate
        ecg_fs       : ECG sampling rate

    Returns:
        X : (n_windows, 280)  — 258 EEG + 22 ECG features
    """
    from src.features.eeg_features import extract_eeg_features
    from src.features.ecg_features import extract_ecg_features

    n = eeg_segments.shape[0]
    rows = []

    for i in range(n):
        eeg_feat = extract_eeg_features(eeg_segments[i], fs=eeg_fs)
        ecg_feat = extract_ecg_features(ecg_segments[i], fs=ecg_fs)
        rows.append(np.concatenate([eeg_feat, ecg_feat]))

    X = np.stack(rows, axis=0)
    logger.info(f"Feature matrix built: {X.shape}")
    return X


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate(model: Pipeline,
             X: np.ndarray,
             y: np.ndarray,
             split: str = "test") -> Dict[str, float]:
    """
    Run predictions and return metrics dict.

    Returns:
        dict with accuracy, f1, roc_auc
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "f1"      : round(f1_score(y, y_pred, average="binary"), 4),
        "roc_auc" : round(roc_auc_score(y, y_prob), 4),
    }

    logger.info(f"[{split.upper()}] Acc={metrics['accuracy']} "
                f"F1={metrics['f1']} AUC={metrics['roc_auc']}")
    print(f"\n── Classification Report ({split}) ──")
    print(classification_report(y, y_pred, target_names=["Low", "High"]))

    return metrics


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_model(model: Pipeline, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved → {path}")


def load_model(path: str) -> Pipeline:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at {path}")
    model = joblib.load(path)
    logger.info(f"Model loaded ← {path}")
    return model