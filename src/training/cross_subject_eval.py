"""
cross_subject_eval.py
Responsible for: Reusable LOSO cross-subject evaluation for deep models.
Can be imported in notebooks or run as a standalone script.

Usage (script):
    python src/training/cross_subject_eval.py \
        --target valence \
        --model_type fusion \
        --modality fusion \
        --config configs/default.yaml
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
))

from src.utils.config      import load_config
from src.utils.logger      import get_logger
from src.data.dataset      import DREAMERDataset
from src.models.deep_model import build_model
from src.training.trainer  import Trainer, _run_epoch
from src.training.contrastive_loss import SubjectContrastiveLoss

logger = get_logger("cross_subject_eval")

def _run_epoch_loso(model, loader, criterion, optimizer, device, is_train, epoch=1, contrastive_fn=None):
    model.train() if is_train else model.eval()
    total_loss, total_ce, total_cl = 0.0, 0.0, 0.0
    correct, total = 0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    
    avg_eeg_attn, avg_ecg_attn = 0.0, 0.0

    with ctx:
        for batch in loader:
            eeg, ecg, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            subjects = batch[3].to(device) if len(batch) > 3 else None

            if is_train: optimizer.zero_grad()

            logits = model(eeg, ecg) if hasattr(model, "ecg_branch") else model(eeg)
            ce_loss = criterion(logits, labels)
            loss = ce_loss
            cl_loss_val = 0.0

            # Contrastive Loss Warmup Strategy (Epochs 6+)
            if is_train and contrastive_fn and hasattr(model, "extract_embedding") and epoch >= 6:
                embeddings = model.extract_embedding(eeg, ecg)
                cl_loss = contrastive_fn(embeddings, labels, subjects)
                loss = ce_loss + 0.1 * cl_loss
                cl_loss_val = cl_loss.item()
                
            # Log attention weights if available
            if hasattr(model, "attention"):
                with torch.no_grad():
                    fused = model.extract_embedding(eeg, ecg)
                    attn = model.attention(fused).mean(dim=0)
                    avg_eeg_attn += attn[0].item() * labels.size(0)
                    avg_ecg_attn += attn[1].item() * labels.size(0)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_ce += ce_loss.item() * labels.size(0)
            total_cl += cl_loss_val * labels.size(0)
            
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    metrics = {
        "loss": total_loss / total,
        "acc": correct / total,
        "ce_loss": total_ce / total,
        "cl_loss": total_cl / total,
        "attn_eeg": avg_eeg_attn / total if hasattr(model, "attention") else 0.0,
        "attn_ecg": avg_ecg_attn / total if hasattr(model, "attention") else 0.0,
    }

    return metrics

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

    for batch in loader:
        eeg, ecg, labels = batch[0].to(device), batch[1].to(device), batch[2]

        logits = model(eeg, ecg) if hasattr(model, "ecg_branch") else model(eeg)
        probs = torch.softmax(logits, dim=1)[:, 1]

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_labels), None, np.array(all_probs)


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


def _aggregate_by_video(y_true, y_prob, meta_list):
    """
    Group predictions by (subject, video) and average probabilities.

    Args:
        y_true    : array of true labels (per window)
        y_prob    : array of predicted probabilities (per window)
        meta_list : list of sample dicts with 'subject' and 'video' keys

    Returns:
        y_true_agg, y_prob_agg as numpy arrays (per video)
    """
    group_probs = defaultdict(list)
    group_labels = {}

    for i in range(len(y_true)):
        key = (meta_list[i]["subject"], meta_list[i]["video"])
        group_probs[key].append(y_prob[i])
        group_labels[key] = y_true[i]

    y_true_agg = []
    y_prob_agg = []

    for key in group_probs:
        y_true_agg.append(group_labels[key])
        y_prob_agg.append(np.mean(group_probs[key]))

    return np.array(y_true_agg), np.array(y_prob_agg)


def run_loso_fold(
    dataset     : DREAMERDataset,
    test_subject: int,
    model_type  : str,
    config      : dict,
    device      : torch.device,
    checkpoint_dir: str = "outputs/models",
    modality    : str = "fusion",
) -> dict:
    """
    Train and evaluate one LOSO fold.

    Returns:
        dict with subject, accuracy, f1, roc_auc
    """
    train_idx_full, test_idx = get_loso_indices(dataset, test_subject)

    # ── Subject-level validation split (deterministic per fold) ──
    train_subjects_all = sorted(set(
        dataset.samples[i]["subject"] for i in train_idx_full
    ))
    rng = np.random.RandomState(test_subject)
    val_subjects = set(rng.choice(train_subjects_all, size=3, replace=False))

    train_idx = [i for i in train_idx_full
                 if dataset.samples[i]["subject"] not in val_subjects]
    val_idx   = [i for i in train_idx_full
                 if dataset.samples[i]["subject"] in val_subjects]

    if len(test_idx) == 0:
        logger.warning(f"No test samples for subject {test_subject}, skipping")
        return {}

    # Check class balance in test fold
    test_labels = [dataset.samples[i]["label"] for i in test_idx]
    if len(set(test_labels)) < 2:
        logger.warning(f"Subject {test_subject} has single class — skipping")
        return {}

    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)
    test_ds  = Subset(dataset, test_idx)

    tc = config["training"]

    # ── Class weights for loss ONLY (no WeightedRandomSampler) ──
    train_labels_arr = np.array(
        [dataset.samples[i]["label"] for i in train_idx]
    )
    classes, counts = np.unique(train_labels_arr, return_counts=True)
    wpc = 1.0 / counts.astype(float)

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=int(tc["batch_size"]),
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(tc["batch_size"]),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(tc["batch_size"]),
        shuffle=False,
        num_workers=2,
    )

    # Model (with modality ablation)
    model = build_model(model_type, config, modality=modality).to(device)

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

    contrastive_fn = SubjectContrastiveLoss(temperature=0.1)

    for epoch in range(1, epochs + 1):
        tr_metrics = _run_epoch_loso(
            model, train_loader, criterion,
            optimizer, device, is_train=True,
            epoch=epoch, contrastive_fn=contrastive_fn
        )

        val_metrics = _run_epoch_loso(
            model, val_loader, criterion,
            None, device, is_train=False
        )

        val_loss = val_metrics["loss"]
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
                f"tr_loss={tr_metrics['loss']:.4f} tr_acc={tr_metrics['acc']:.4f} cl_loss={tr_metrics['cl_loss']:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_metrics['acc']:.4f} | "
                f"Attn(EEG/ECG)={tr_metrics['attn_eeg']:.2f}/{tr_metrics['attn_ecg']:.2f}"
            )

    # ── Load best checkpoint and evaluate ──
    model.load_state_dict(torch.load(ckpt, map_location=device))

    # ── Get raw validation predictions ──
    y_val_true_raw, _, y_val_prob_raw = _predict(model, val_loader, device)

    # ── Aggregate validation predictions to video-level (matches test) ──
    val_meta = [dataset.samples[i] for i in val_idx]
    y_val_true, y_val_prob = _aggregate_by_video(
        y_val_true_raw, y_val_prob_raw, val_meta
    )

    # ── Direction fix using VALIDATION ONLY (no test leakage) ──
    auc_val = (roc_auc_score(y_val_true, y_val_prob)
               if len(set(y_val_true)) > 1 else 0.5)
    flip_direction = auc_val < 0.5

    if flip_direction:
        y_val_prob = 1 - y_val_prob

    # ── Threshold on aggregated validation predictions ──
    best_thresh, _ = find_best_threshold(y_val_true, y_val_prob)
    best_thresh = float(np.clip(best_thresh, 0.25, 0.75))

    # ── Predict on test (apply SAME flip decision) ──
    y_true, _, y_prob = _predict(model, test_loader, device)

    if flip_direction:
        y_prob = 1 - y_prob

    # ── FIX 3: Video-level aggregation ──
    test_meta = [dataset.samples[i] for i in test_idx]
    y_true_agg, y_prob_agg = _aggregate_by_video(y_true, y_prob, test_meta)

    # Apply threshold on aggregated predictions
    y_pred_agg = (y_prob_agg > best_thresh).astype(int)

    # ── FINAL METRICS (computed on video-level aggregated predictions) ──
    result = {
        "subject" : test_subject,
        "accuracy": round(float(accuracy_score(y_true_agg, y_pred_agg)), 4),
        "f1"      : round(float(f1_score(y_true_agg, y_pred_agg,
                                          average="binary",
                                          zero_division=0)), 4),
        "roc_auc" : round(float(roc_auc_score(y_true_agg, y_prob_agg)
                                 if len(set(y_true_agg)) > 1 else 0.5), 4),
        "n_train" : len(train_idx),
        "n_test"  : len(test_idx),
    }

    logger.info(
        f"Sub {test_subject:02d} DONE | "
        f"Acc={result['accuracy']:.4f} F1={result['f1']:.4f} AUC={result['roc_auc']:.4f} "
        f"| BestThresh={best_thresh:.2f}"
    )

    return result


# ── Full LOSO loop ────────────────────────────────────────────────────────────

def run_loso(
    mat_path      : str,
    target        : str,
    model_type    : str,
    config        : dict,
    save_dir      : str = "outputs/results",
    checkpoint_dir: str = "outputs/models",
    modality      : str = "fusion",
) -> dict:
    """
    Run full 23-fold LOSO evaluation for a deep model.

    Returns:
        summary dict with per-fold results and aggregate stats
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        f"LOSO | target={target} | model={model_type} | "
        f"modality={modality} | device={device}"
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

    # FIX 7: Full 23-subject LOSO
    for sub_id in range(1, 24):
        logger.info(f"\n── LOSO Fold: Test Subject {sub_id}/23 ──")
        result = run_loso_fold(
            dataset, sub_id, model_type,
            config, device, checkpoint_dir,
            modality=modality,
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
        "modality"  : modality,
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
    parser.add_argument("--modality",   type=str, default="fusion",
                        choices=["eeg", "ecg", "fusion"])
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
        modality       = args.modality,
    )