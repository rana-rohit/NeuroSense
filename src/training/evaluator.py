"""
evaluator.py
Responsible for: Model evaluation, metrics, confusion matrix,
                 training curve plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from typing import Dict, Tuple, List

from src.utils.logger import get_logger

logger = get_logger("evaluator")


# ── Prediction collector ──────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(
    model  : nn.Module,
    loader : DataLoader,
    device : torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model on full DataLoader and collect predictions.

    Returns:
        y_true : (N,) int
        y_pred : (N,) int
        y_prob : (N,) float  — probability of class 1
    """
    model.eval()
    all_true, all_pred, all_prob = [], [], []

    for eeg, ecg, labels in loader:
        eeg, ecg = eeg.to(device), ecg.to(device)

        if hasattr(model, "ecg_branch"):
            logits = model(eeg, ecg)
        else:
            logits = model(eeg)

        probs  = torch.softmax(logits, dim=1)[:, 1]
        preds  = logits.argmax(dim=1)

        all_true.extend(labels.cpu().numpy())
        all_pred.extend(preds.cpu().numpy())
        all_prob.extend(probs.cpu().numpy())

    return (
        np.array(all_true),
        np.array(all_pred),
        np.array(all_prob),
    )


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray) -> Dict[str, float]:
    """
    Compute full classification metrics.

    Returns:
        dict: accuracy, f1, roc_auc, avg_precision
    """
    metrics = {
        "accuracy"      : round(float(accuracy_score(y_true, y_pred)), 4),
        "f1"            : round(float(f1_score(y_true, y_pred, average="binary")), 4),
        "roc_auc"       : round(float(roc_auc_score(y_true, y_prob)), 4),
        "avg_precision" : round(float(average_precision_score(y_true, y_prob)), 4),
    }
    logger.info(f"Metrics: {metrics}")
    return metrics


# ── Confusion matrix plot ─────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true    : np.ndarray,
    y_pred    : np.ndarray,
    target    : str,
    save_dir  : str = "outputs/results",
):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low", "High"],
                yticklabels=["Low", "High"],
                linewidths=0.5)
    plt.title(f"Confusion Matrix — {target.capitalize()}", fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"confusion_matrix_{target}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    logger.info(f"Confusion matrix saved → {path}")


# ── ROC + PR curve ────────────────────────────────────────────────────────────

def plot_roc_pr_curves(
    y_true   : np.ndarray,
    y_prob   : np.ndarray,
    target   : str,
    save_dir : str = "outputs/results",
):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score   = roc_auc_score(y_true, y_prob)

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap_score     = average_precision_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    axes[0].plot(fpr, tpr, color="steelblue", lw=2,
                 label=f"AUC = {auc_score:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve — {target.capitalize()}", fontweight="bold")
    axes[0].legend()

    # PR
    axes[1].plot(rec, prec, color="darkorange", lw=2,
                 label=f"AP = {ap_score:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR Curve — {target.capitalize()}", fontweight="bold")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(save_dir, f"roc_pr_{target}.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    logger.info(f"ROC/PR curves saved → {path}")


# ── Training curves ───────────────────────────────────────────────────────────

def plot_training_curves(
    history  : Dict[str, list],
    target   : str,
    save_dir : str = "outputs/results",
):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train", color="steelblue")
    axes[0].plot(epochs, history["val_loss"],   label="Val",   color="darkorange")
    axes[0].set_title(f"Loss — {target.capitalize()}", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train", color="steelblue")
    axes[1].plot(epochs, history["val_acc"],   label="Val",   color="darkorange")
    axes[1].set_title(f"Accuracy — {target.capitalize()}", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(save_dir, f"training_curves_{target}.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    logger.info(f"Training curves saved → {path}")


# ── Full evaluation pipeline ──────────────────────────────────────────────────

def evaluate_model(
    model     : nn.Module,
    loader    : DataLoader,
    device    : torch.device,
    target    : str,
    history   : Dict[str, list] = None,
    save_dir  : str = "outputs/results",
) -> Dict[str, float]:
    """
    Run full evaluation: metrics + all plots.

    Returns:
        metrics dict
    """
    logger.info(f"Evaluating model on {target}...")

    y_true, y_pred, y_prob = collect_predictions(model, loader, device)

    metrics = compute_metrics(y_true, y_pred, y_prob)

    print(f"\n{'='*50}")
    print(f"  Evaluation Results — {target.capitalize()}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred,
                                 target_names=["Low", "High"]))
    print(f"  ROC-AUC        : {metrics['roc_auc']}")
    print(f"  Avg Precision  : {metrics['avg_precision']}")
    print(f"{'='*50}\n")

    plot_confusion_matrix(y_true, y_pred, target, save_dir)
    plot_roc_pr_curves(y_true, y_prob, target, save_dir)

    if history:
        plot_training_curves(history, target, save_dir)

    return metrics