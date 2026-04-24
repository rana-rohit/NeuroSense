"""
interpretability.py
Responsible for: Model interpretability via SHAP (baseline) and
                 gradient-based saliency maps (deep models).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import torch.nn as nn
from typing import List, Optional

from src.utils.logger import get_logger

logger = get_logger("interpretability")

EEG_CHANNELS = [
    "AF3","F7","F3","FC5","T7","P7",
    "O1","O2","P8","T8","FC6","F4","F8","AF4"
]


# ══════════════════════════════════════════════════════════════════
# PART A — SHAP for Baseline Models
# ══════════════════════════════════════════════════════════════════

def shap_feature_importance(
    model,
    X_train   : np.ndarray,
    X_test    : np.ndarray,
    feature_names: List[str],
    target    : str,
    n_background: int  = 100,
    save_dir  : str    = "outputs/results",
):
    """
    Compute and plot SHAP values for a sklearn baseline model.

    Args:
        model        : fitted sklearn Pipeline
        X_train      : training feature matrix (n, 280)
        X_test       : test feature matrix     (m, 280)
        feature_names: list of 280 feature name strings
        target       : emotion target label
        n_background : background samples for KernelExplainer
        save_dir     : output directory
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Run: pip install shap")

    os.makedirs(save_dir, exist_ok=True)

    # Use TreeExplainer for RF/GBM, KernelExplainer otherwise
    clf = model.named_steps["clf"]
    scaler = model.named_steps["scaler"]
    X_tr_scaled = scaler.transform(X_train)
    X_te_scaled = scaler.transform(X_test)

    clf_type = type(clf).__name__
    logger.info(f"Computing SHAP values for {clf_type} | target={target}")

    if clf_type in ("RandomForestClassifier", "GradientBoostingClassifier"):
        explainer  = shap.TreeExplainer(clf)
        shap_vals  = explainer.shap_values(X_te_scaled)
        # For binary: shap_values returns list[2]; take class-1
        if isinstance(shap_vals, list):
            sv = shap_vals[1]
        else:
            sv = shap_vals
    else:
        bg = shap.kmeans(X_tr_scaled, min(n_background, len(X_tr_scaled)))
        def predict_proba_pos(x):
            return clf.predict_proba(x)[:, 1]
        explainer = shap.KernelExplainer(predict_proba_pos, bg)
        sv = explainer.shap_values(X_te_scaled[:50])

    # ── Global feature importance bar plot ────────────────────────
    mean_abs_shap = np.abs(sv).mean(axis=0)
    top_k         = 20
    top_idx       = np.argsort(mean_abs_shap)[::-1][:top_k]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = plt.cm.RdYlGn(
        np.linspace(0.2, 0.9, top_k)[::-1]
    )
    ax.barh(
        range(top_k),
        mean_abs_shap[top_idx],
        color=colors,
        edgecolor="white",
    )
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(
        [feature_names[i] for i in top_idx], fontsize=8
    )
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|", fontweight="bold")
    ax.set_title(
        f"Top {top_k} Feature Importances — {target.capitalize()}",
        fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(save_dir, f"shap_importance_{target}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    logger.info(f"SHAP bar plot saved → {path}")

    # ── Summary beeswarm plot ─────────────────────────────────────
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        sv[:, top_idx],
        X_te_scaled[:len(sv), :][:, top_idx],
        feature_names=[feature_names[i] for i in top_idx],
        show=False,
        plot_size=(10, 8),
    )
    path2 = os.path.join(save_dir, f"shap_beeswarm_{target}.png")
    plt.savefig(path2, bbox_inches="tight")
    plt.show()
    logger.info(f"SHAP beeswarm saved → {path2}")

    # Save raw SHAP values
    np.save(
        os.path.join(save_dir, f"shap_values_{target}.npy"), sv
    )

    return sv, mean_abs_shap


# ══════════════════════════════════════════════════════════════════
# PART B — Gradient Saliency for Deep Models
# ══════════════════════════════════════════════════════════════════

def compute_saliency(
    model  : nn.Module,
    eeg    : torch.Tensor,
    ecg    : torch.Tensor,
    target_class: int,
    device : torch.device,
) -> dict:
    """
    Vanilla gradient saliency: d(class_score) / d(input).

    Args:
        model       : trained FusionModel
        eeg         : (1, eeg_time, 14)
        ecg         : (1, ecg_time, 2)
        target_class: 0 or 1
        device      : torch device

    Returns:
        dict with eeg_saliency (eeg_time, 14) and ecg_saliency (ecg_time, 2)
    """
    model.eval()
    eeg_in = eeg.to(device).float().requires_grad_(True)
    ecg_in = ecg.to(device).float().requires_grad_(True)

    if hasattr(model, "ecg_branch"):
        logits = model(eeg_in, ecg_in)
    else:
        logits = model(eeg_in)

    score = logits[0, target_class]
    score.backward()

    eeg_sal = eeg_in.grad.abs().squeeze(0).cpu().numpy()  # (time, 14)
    ecg_sal = ecg_in.grad.abs().squeeze(0).cpu().numpy() \
              if ecg_in.grad is not None else None

    return {"eeg": eeg_sal, "ecg": ecg_sal}


def compute_smoothgrad(
    model       : nn.Module,
    eeg         : torch.Tensor,
    ecg         : torch.Tensor,
    target_class: int,
    device      : torch.device,
    n_samples   : int   = 30,
    noise_level : float = 0.1,
) -> dict:
    """
    SmoothGrad: average saliency over noisy input copies.
    Reduces noise artefacts in vanilla gradients.

    Returns:
        dict with eeg_saliency, ecg_saliency (same shapes as input)
    """
    eeg_grads, ecg_grads = [], []

    std_eeg = noise_level * (eeg.max() - eeg.min()).item()
    std_ecg = noise_level * (ecg.max() - ecg.min()).item()

    for _ in range(n_samples):
        eeg_noisy = (eeg + torch.randn_like(eeg) * std_eeg)
        ecg_noisy = (ecg + torch.randn_like(ecg) * std_ecg)

        sal = compute_saliency(
            model, eeg_noisy, ecg_noisy, target_class, device
        )
        eeg_grads.append(sal["eeg"])
        if sal["ecg"] is not None:
            ecg_grads.append(sal["ecg"])

    return {
        "eeg": np.mean(eeg_grads, axis=0),
        "ecg": np.mean(ecg_grads, axis=0) if ecg_grads else None,
    }


def plot_eeg_saliency(
    saliency : np.ndarray,
    eeg_fs   : float  = 128.0,
    target   : str    = "valence",
    method   : str    = "SmoothGrad",
    save_dir : str    = "outputs/results",
):
    """
    Plot EEG saliency as (1) channel heatmap and
    (2) time-aggregated bar chart per channel.

    Args:
        saliency : (eeg_time, 14) saliency array
    """
    os.makedirs(save_dir, exist_ok=True)
    time_ax = np.arange(saliency.shape[0]) / eeg_fs

    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                             hspace=0.4, wspace=0.35)

    # ── Heatmap: time × channel ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    sns.heatmap(
        saliency.T,
        ax=ax1,
        cmap="hot",
        xticklabels=False,
        yticklabels=EEG_CHANNELS,
        cbar_kws={"label": "|gradient|"},
    )
    ax1.set_title(
        f"EEG Saliency Map — {target.capitalize()} ({method})",
        fontweight="bold"
    )
    ax1.set_xlabel("Time →")
    ax1.set_ylabel("Channel")

    # ── Mean saliency per channel ──────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ch_sal = saliency.mean(axis=0)                  # (14,)
    colors = plt.cm.hot(ch_sal / ch_sal.max())
    ax2.bar(EEG_CHANNELS, ch_sal, color=colors, edgecolor="white")
    ax2.set_xticklabels(EEG_CHANNELS, rotation=45, ha="right", fontsize=8)
    ax2.set_title("Mean Saliency per Channel", fontweight="bold")
    ax2.set_ylabel("|gradient|")

    # ── Mean saliency over time ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    t_sal = saliency.mean(axis=1)                   # (time,)
    ax3.plot(time_ax, t_sal, color="crimson", linewidth=0.8)
    ax3.fill_between(time_ax, t_sal, alpha=0.3, color="crimson")
    ax3.set_title("Saliency Over Time (avg channels)", fontweight="bold")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("|gradient|")

    path = os.path.join(
        save_dir, f"eeg_saliency_{target}_{method.lower()}.png"
    )
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    logger.info(f"EEG saliency plot saved → {path}")


def plot_ecg_saliency(
    saliency : np.ndarray,
    ecg_fs   : float = 256.0,
    target   : str   = "valence",
    method   : str   = "SmoothGrad",
    save_dir : str   = "outputs/results",
):
    """
    Plot ECG saliency for both channels.

    Args:
        saliency: (ecg_time, 2)
    """
    os.makedirs(save_dir, exist_ok=True)
    time_ax = np.arange(saliency.shape[0]) / ecg_fs

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for i, (ax, ch_name) in enumerate(
        zip(axes, ["ECG Ch1", "ECG Ch2"])
    ):
        sal = saliency[:, i]
        ax.plot(time_ax, sal, color="darkorange", linewidth=0.8)
        ax.fill_between(time_ax, sal, alpha=0.3, color="darkorange")
        ax.set_title(
            f"{ch_name} Saliency — {target.capitalize()} ({method})",
            fontweight="bold"
        )
        ax.set_ylabel("|gradient|")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    path = os.path.join(
        save_dir, f"ecg_saliency_{target}_{method.lower()}.png"
    )
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    logger.info(f"ECG saliency plot saved → {path}")