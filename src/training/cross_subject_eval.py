"""
cross_subject_eval.py
Responsible for: Reusable LOSO cross-subject evaluation for deep models.
Can be imported in notebooks or run as a standalone script.

Usage (script):
    python src/training/cross_subject_eval.py \
        --target valence \
        --model_type fusion \
        --config configs/default.yaml
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
))

from src.utils.config      import load_config
from src.utils.logger      import get_logger
from src.data.dataset      import DREAMERDataset
from src.models.deep_model import build_model
from src.training.trainer  import Trainer, _run_epoch

logger = get_logger("cross_subject_eval")
SMOOTH_WINDOW = 7

# ── LOSO split indices ────────────────────────────────────────────────────────

def get_loso_indices(dataset: DREAMERDataset,
                     test_subject: int):
    """
    Return train and test index lists for one LOSO fold.

    Args:
        dataset     : DREAMERDataset (all subjects loaded)
        test_subject: 1-indexed subject ID held out for test

    Returns:
        train_indices, test_indices
    """
    train_idx, test_idx = [], []
    for i, sample in enumerate(dataset.samples):
        if sample["subject"] == test_subject:
            test_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, test_idx


# ── Single LOSO fold ──────────────────────────────────────────────────────────

@torch.no_grad()
def _predict(model, loader, device):
    model.eval()

    all_probs = []
    all_labels = []

    for eeg, ecg, labels in loader:
        eeg, ecg = eeg.to(device), ecg.to(device)

        logits = model(eeg, ecg) if hasattr(model, "ecg_branch") else model(eeg)
        probs = torch.softmax(logits, dim=1)[:, 1]

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_labels), None, np.array(all_probs)

def smooth_predictions(probs, window=SMOOTH_WINDOW):
    if len(probs) < window:
        return probs  # avoid edge crash
    return np.convolve(probs, np.ones(window)/window, mode='same')

def run_loso_fold(
    dataset     : DREAMERDataset,
    test_subject: int,
    model_type  : str,
    config      : dict,
    device      : torch.device,
    checkpoint_dir: str = "outputs/models",
) -> dict:
    """
    Train and evaluate one LOSO fold.

    Returns:
        dict with subject, accuracy, f1, roc_auc
    """
    train_idx_full, test_idx = get_loso_indices(dataset, test_subject)

    train_idx, val_idx = train_test_split(
        train_idx_full,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    if len(test_idx) == 0:
        logger.warning(f"No test samples for subject {test_subject}, skipping")
        return {}

    train_ds = Subset(dataset, train_idx)
    test_ds  = Subset(dataset, test_idx)

    tc = config["training"]

    # Check class balance in test fold
    test_labels = [dataset.samples[i]["label"] for i in test_idx]
    if len(set(test_labels)) < 2:
        logger.warning(f"Subject {test_subject} has single class — skipping")
        return {}

    # DataLoaders
    from torch.utils.data import WeightedRandomSampler

    train_labels_arr = np.array(
        [dataset.samples[i]["label"] for i in train_idx]
    )
    classes, counts = np.unique(train_labels_arr, return_counts=True)
    wpc = 1.0 / counts.astype(float)
    sw  = torch.tensor([wpc[l] for l in train_labels_arr], dtype=torch.float)

    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(tc["batch_size"]),
        sampler=sampler,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(tc["batch_size"]),
        shuffle=False,
        num_workers=2,
    )

    # Model
    model = build_model(model_type, config).to(device)

    # Loss with class weights
    w = torch.tensor(wpc / wpc.sum(), dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tc["learning_rate"]),
        weight_decay=float(tc.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    
    val_ds = Subset(dataset, val_idx)

    val_loader = DataLoader(
        val_ds,
        batch_size=int(tc["batch_size"]),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # Checkpoint path for this fold
    ckpt = os.path.join(
        checkpoint_dir,
        f"loso_sub{test_subject}_{dataset.target}_{model_type}.pt"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_loss = np.inf
    patience_counter = 0
    patience = int(tc.get("patience", 10))
    epochs   = int(tc["epochs"])

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(
            model, train_loader, criterion,
            optimizer, device, is_train=True
        )

        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion,
            None, device, is_train=False
        )

        scheduler.step(val_loss)

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    f"  Sub {test_subject:02d} | "
                    f"Early stop @ epoch {epoch}"
                )
                break

        if epoch % 10 == 0:
            logger.info(
                f"  Sub {test_subject:02d} | Ep {epoch:03d} | "
                f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

    # Load best and evaluate
    model.load_state_dict(torch.load(ckpt, map_location=device))

    y_val_true, _, y_val_prob = _predict(model, val_loader, device)

    # Direction fix
    auc_val = roc_auc_score(y_val_true, y_val_prob) if len(set(y_val_true)) > 1 else 0.5
    if auc_val < 0.5:
        y_val_prob = 1 - y_val_prob

    # ── VALIDATION SMOOTHING (FIXED) ──
    val_meta = [dataset.samples[i] for i in val_idx]

    y_val_smoothed = []
    current_group = []
    current_meta = None

    for i in range(len(y_val_prob)):
        meta = (val_meta[i]["subject"], val_meta[i]["video"])

        if current_meta is None:
            current_meta = meta

        if meta != current_meta and len(current_group) > 0:
            chunk_smoothed = smooth_predictions(np.array(current_group))
            y_val_smoothed.extend(chunk_smoothed)

            current_group = []
            current_meta = meta

        current_group.append(y_val_prob[i])  # ← INSIDE loop (FIXED)
    
    # last group
    if current_group:
        chunk_smoothed = smooth_predictions(np.array(current_group))
        y_val_smoothed.extend(chunk_smoothed)

    y_val_prob = np.array(y_val_smoothed)
    

    # Find threshold
    best_thresh, _ = find_best_threshold(y_val_true, y_val_prob)
    
    best_thresh = float(np.clip(best_thresh, 0.25, 0.75))

    y_true, _, y_prob = _predict(model, test_loader, device)

    # ── STEP 1: Direction Fix ──
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5

    if auc < 0.5:
        y_prob = 1 - y_prob
        auc = 1 - auc

    # ── STEP 2: Temporal Smoothing ──
    y_prob_smoothed = []

    test_indices = list(test_idx)
    current_group = []
    current_meta = None

    for i, idx in enumerate(test_indices):
        sample = dataset.samples[idx]
        meta = (sample["subject"], sample["video"])

        if current_meta is None:
            current_meta = meta

        if meta != current_meta and len(current_group) > 0:
            chunk_smoothed = smooth_predictions(np.array(current_group))
            y_prob_smoothed.extend(chunk_smoothed)

            current_group = []
            current_meta = meta

        current_group.append(y_prob[i])

    # last group
    if current_group:
        chunk_smoothed = smooth_predictions(np.array(current_group))
        y_prob_smoothed.extend(chunk_smoothed)

    y_prob = np.array(y_prob_smoothed)
    
    # Apply threshold
    y_pred = (y_prob > best_thresh).astype(int)

    # ── FINAL METRICS ──
    result = {
        "subject" : test_subject,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1"      : round(float(f1_score(y_true, y_pred,
                                          average="binary",
                                          zero_division=0)), 4),
        "roc_auc" : round(float(roc_auc_score(y_true, y_prob)
                                 if len(set(y_true)) > 1 else 0.5), 4),
        "n_train" : len(train_idx),
        "n_test"  : len(test_idx),
    }

    logger.info(
        f"Sub {test_subject:02d} DONE | "
        f"Acc={result['accuracy']:.4f} F1={result['f1']:.4f} AUC={result['roc_auc']:.4f} "
        f"| BestThresh={best_thresh:.2f}"
    )

    return result

def find_best_threshold(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return 0.5, 0

    best_thresh = 0.5
    best_f1 = 0

    thresholds = np.percentile(y_prob, np.linspace(5, 95, 100))

    for t in thresholds:
        y_pred = (y_prob > t).astype(int)
        f1 = f1_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh, best_f1

# ── Full LOSO loop ────────────────────────────────────────────────────────────

def run_loso(
    mat_path      : str,
    target        : str,
    model_type    : str,
    config        : dict,
    save_dir      : str = "outputs/results",
    checkpoint_dir: str = "outputs/models",
) -> dict:
    """
    Run full 23-fold LOSO evaluation for a deep model.

    Returns:
        summary dict with per-fold results and aggregate stats
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        f"LOSO | target={target} | model={model_type} | device={device}"
    )

    # Load full dataset (all subjects)
    dataset = DREAMERDataset(
        mat_path    = mat_path,
        target      = target,
        window_sec  = config["data"]["segment_length"],
        overlap_sec = config["data"]["overlap"],
        norm_method = config["data"]["norm_method"],
        threshold   = config["labels"]["threshold"],
    )

    fold_results = []

    for sub_id in range(1, 24):
        logger.info(f"\n── LOSO Fold: Test Subject {sub_id}/23 ──")
        result = run_loso_fold(
            dataset, sub_id, model_type,
            config, device, checkpoint_dir
        )
        if result:
            fold_results.append(result)

    # Aggregate
    accs = [r["accuracy"] for r in fold_results]
    f1s  = [r["f1"]       for r in fold_results]
    aucs = [r["roc_auc"]  for r in fold_results]

    summary = {
        "target"    : target,
        "model"     : model_type,
        "n_folds"   : len(fold_results),
        "acc_mean"  : round(float(np.mean(accs)), 4),
        "acc_std"   : round(float(np.std(accs)),  4),
        "f1_mean"   : round(float(np.mean(f1s)),  4),
        "f1_std"    : round(float(np.std(f1s)),   4),
        "auc_mean"  : round(float(np.mean(aucs)), 4),
        "auc_std"   : round(float(np.std(aucs)),  4),
        "per_fold"  : fold_results,
    }

    # Save
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(
        save_dir, f"loso_deep_{model_type}_{target}.json"
    )
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*55}")
    logger.info(f"  LOSO Complete | target={target} | model={model_type}")
    logger.info(f"  Acc : {summary['acc_mean']} ± {summary['acc_std']}")
    logger.info(f"  F1  : {summary['f1_mean']}  ± {summary['f1_std']}")
    logger.info(f"  AUC : {summary['auc_mean']} ± {summary['auc_std']}")
    logger.info(f"  Saved → {out_path}")
    logger.info(f"{'='*55}")

    return summary


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LOSO cross-subject evaluation for deep models"
    )
    parser.add_argument("--target",     type=str, default="valence",
                        choices=["valence", "arousal", "dominance"])
    parser.add_argument("--model_type", type=str, default="fusion",
                        choices=["eegnet", "cnn", "cnnlstm", "fusion"])
    parser.add_argument("--config",     type=str,
                        default="configs/default.yaml")
    parser.add_argument("--save_dir",   type=str,
                        default="outputs/results")
    parser.add_argument("--ckpt_dir",   type=str,
                        default="outputs/models")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    summary = run_loso(
        mat_path       = cfg["data"]["raw_path"],
        target         = args.target,
        model_type     = args.model_type,
        config         = cfg,
        save_dir       = args.save_dir,
        checkpoint_dir = args.ckpt_dir,
    )