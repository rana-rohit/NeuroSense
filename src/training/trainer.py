"""
trainer.py
Responsible for: Training loop, checkpointing, early stopping,
                 LR scheduling for deep learning models.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from typing import Dict, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger("trainer")


# ── Early Stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stop training when val_loss stops improving.

    Args:
        patience : epochs to wait after last improvement
        delta    : minimum improvement to count
        path     : path to save best model checkpoint
    """

    def __init__(self, patience: int = 10, delta: float = 1e-4,
                 path: str = "outputs/models/best_model.pt"):
        self.patience   = patience
        self.delta      = delta
        self.path       = path
        self.counter    = 0
        self.best_loss  = np.inf
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter   = 0
            self._save(model)
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def _save(self, model: nn.Module):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
        logger.info(f"✅ Best model saved → {self.path}")


# ── Weighted sampler builder ──────────────────────────────────────────────────

def make_weighted_sampler(dataset) -> WeightedRandomSampler:
    """Balance classes using WeightedRandomSampler."""
    labels  = np.array([s["label"] for s in dataset.samples])
    classes, counts = np.unique(labels, return_counts=True)
    weight_per_class = 1.0 / counts.astype(float)
    sample_weights   = np.array([weight_per_class[l] for l in labels])
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


# ── One epoch ─────────────────────────────────────────────────────────────────

def _run_epoch(
    model      : nn.Module,
    loader     : DataLoader,
    criterion  : nn.Module,
    optimizer  : Optional[torch.optim.Optimizer],
    device     : torch.device,
    is_train   : bool,
) -> Tuple[float, float]:
    """
    Run one pass over a DataLoader.

    Returns:
        avg_loss, accuracy
    """
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for eeg, ecg, labels in loader:
            eeg, ecg, labels = (
                eeg.to(device),
                ecg.to(device),
                labels.to(device),
            )

            if is_train:
                optimizer.zero_grad()

            # FusionModel takes (eeg, ecg); EEGNet takes only eeg
            if hasattr(model, "ecg_branch"):
                logits = model(eeg, ecg)
            else:
                logits = model(eeg)

            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    return total_loss / total, correct / total


# ── Main Trainer ──────────────────────────────────────────────────────────────

class Trainer:
    """
    Encapsulates the full training workflow.

    Usage:
        trainer = Trainer(model, dataset, config)
        history = trainer.fit()
    """

    def __init__(
        self,
        model      : nn.Module,
        dataset,
        config     : dict,
        target     : str = "valence",
        checkpoint_dir: str = "outputs/models",
    ):
        self.config    = config
        self.target    = target
        self.device    = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Device: {self.device}")

        tc = config["training"]
        self.epochs      = int(tc["epochs"])
        self.batch_size  = int(tc["batch_size"])
        self.lr          = float(tc["learning_rate"])
        self.patience    = int(tc.get("patience", 10))

        # ── Train / Val split ──────────────────────────────────────────────
        val_ratio  = float(tc.get("val_ratio", 0.2))
        n_val      = int(len(dataset) * val_ratio)
        n_train    = len(dataset) - n_val

        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(
                int(tc.get("seed", 42))
            )
        )

        sampler = make_weighted_sampler(dataset)

        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=2,
            pin_memory=self.device.type == "cuda",
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=self.device.type == "cuda",
        )

        # ── Model ──────────────────────────────────────────────────────────
        self.model = model.to(self.device)

        # ── Loss ───────────────────────────────────────────────────────────
        class_weights = dataset.class_weights().to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # ── Optimiser + Scheduler ──────────────────────────────────────────
        opt_name = str(tc.get("optimizer", "adam")).lower()
        if opt_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr,
                weight_decay=float(tc.get("weight_decay", 1e-4))
            )
        elif opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.lr,
                weight_decay=float(tc.get("weight_decay", 1e-4))
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr, momentum=0.9
            )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5,
            patience=5, verbose=True
        )

        # ── Early stopping ─────────────────────────────────────────────────
        ckpt_path = os.path.join(
            checkpoint_dir, f"best_{target}_{type(model).__name__}.pt"
        )
        self.early_stopping = EarlyStopping(
            patience=self.patience, path=ckpt_path
        )

        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_acc" : [], "val_acc" : [],
        }

    def fit(self) -> Dict[str, list]:
        """
        Run the full training loop.

        Returns:
            history dict with train/val loss and accuracy per epoch
        """
        logger.info(
            f"Training started | target={self.target} | "
            f"epochs={self.epochs} | batch={self.batch_size} | lr={self.lr}"
        )

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            tr_loss, tr_acc = _run_epoch(
                self.model, self.train_loader, self.criterion,
                self.optimizer, self.device, is_train=True
            )
            va_loss, va_acc = _run_epoch(
                self.model, self.val_loader, self.criterion,
                None, self.device, is_train=False
            )

            self.scheduler.step(va_loss)
            self.early_stopping(va_loss, self.model)

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_acc"].append(va_acc)

            elapsed = time.time() - t0
            logger.info(
                f"Epoch [{epoch:03d}/{self.epochs}] "
                f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
                f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} | "
                f"{elapsed:.1f}s"
            )

            if self.early_stopping.early_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        logger.info("Training complete ✅")
        return self.history

    def load_best(self):
        """Load the best checkpoint back into model."""
        self.model.load_state_dict(
            torch.load(self.early_stopping.path,
                       map_location=self.device)
        )
        logger.info("Best checkpoint loaded ✅")