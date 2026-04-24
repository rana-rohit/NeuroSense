"""
tuner.py
Responsible for: Hyperparameter tuning for both baseline (GridSearch)
                 and deep models (Optuna).

Usage:
    # Baseline grid search
    python src/training/tuner.py --mode baseline --target valence

    # Deep model Optuna search
    python src/training/tuner.py --mode deep --target valence --n_trials 30
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
))

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("tuner")


# ══════════════════════════════════════════════════════════════════
# PART A — Baseline Grid Search
# ══════════════════════════════════════════════════════════════════

def tune_baseline(
    X          : np.ndarray,
    y          : np.ndarray,
    model_type : str  = "rf",
    target     : str  = "valence",
    cv_folds   : int  = 5,
    save_dir   : str  = "outputs/results",
) -> dict:
    """
    Grid search over baseline model hyperparameters.

    Args:
        X          : feature matrix (n_windows, 280)
        y          : binary labels  (n_windows,)
        model_type : 'logreg' | 'svm' | 'rf' | 'gbm'
        target     : emotion target name (for saving)
        cv_folds   : stratified K-fold count
        save_dir   : where to save best params JSON

    Returns:
        dict with best_params, best_score, cv_results
    """
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # ── Parameter grids ───────────────────────────────────────────
    grids = {
        "logreg": {
            "clf__C"        : [0.01, 0.1, 1.0, 10.0, 100.0],
            "clf__penalty"  : ["l2"],
            "clf__solver"   : ["lbfgs"],
        },
        "svm": {
            "clf__C"        : [0.1, 1.0, 10.0, 100.0],
            "clf__gamma"    : ["scale", "auto"],
            "clf__kernel"   : ["rbf", "linear"],
        },
        "rf": {
            "clf__n_estimators" : [100, 200, 300],
            "clf__max_depth"    : [None, 10, 20],
            "clf__min_samples_split": [2, 5],
            "clf__max_features" : ["sqrt", "log2"],
        },
        "gbm": {
            "clf__n_estimators"  : [100, 200],
            "clf__learning_rate" : [0.01, 0.05, 0.1],
            "clf__max_depth"     : [3, 4, 6],
            "clf__subsample"     : [0.8, 1.0],
        },
    }

    if model_type not in grids:
        raise ValueError(f"Unknown model_type: {model_type}")

    from src.models.baseline import build_baseline
    pipe = build_baseline(model_type)
    cv   = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    logger.info(
        f"Grid search | model={model_type} target={target} cv={cv_folds}"
    )

    gs = GridSearchCV(
        pipe,
        param_grid=grids[model_type],
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    gs.fit(X, y)

    result = {
        "model_type"  : model_type,
        "target"      : target,
        "best_params" : gs.best_params_,
        "best_auc"    : round(float(gs.best_score_), 4),
        "cv_folds"    : cv_folds,
    }

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"gridsearch_{model_type}_{target}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(
        f"Best AUC={result['best_auc']} | "
        f"params={result['best_params']}"
    )
    logger.info(f"Saved → {out}")
    return result


# ══════════════════════════════════════════════════════════════════
# PART B — Deep Model Optuna Search
# ══════════════════════════════════════════════════════════════════

def tune_deep(
    mat_path  : str,
    target    : str,
    config    : dict,
    n_trials  : int = 30,
    save_dir  : str = "outputs/results",
) -> dict:
    """
    Optuna hyperparameter search for FusionModel.

    Search space:
        lr           : log-uniform [1e-4, 1e-2]
        batch_size   : categorical [16, 32, 64]
        branch_dim   : categorical [64, 128, 256]
        dropout      : uniform [0.2, 0.6]
        weight_decay : log-uniform [1e-5, 1e-3]
        branch_type  : categorical ['cnn', 'cnnlstm']
        window_sec   : categorical [2, 4]

    Returns:
        dict with best_params, best_val_auc
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError(
            "Optuna not installed. Run: pip install optuna"
        )

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split
    from sklearn.metrics import roc_auc_score

    from src.data.dataset      import DREAMERDataset
    from src.models.deep_model import FusionModel
    from src.training.trainer  import make_weighted_sampler, _run_epoch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        f"Optuna search | target={target} n_trials={n_trials} "
        f"device={device}"
    )

    def objective(trial):
        # ── Sample hyperparameters ─────────────────────────────
        lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size  = trial.suggest_categorical("batch_size", [16, 32, 64])
        branch_dim  = trial.suggest_categorical("branch_dim", [64, 128, 256])
        dropout     = trial.suggest_float("dropout", 0.2, 0.6)
        weight_decay= trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        branch_type = trial.suggest_categorical("branch_type",
                                                 ["cnn", "cnnlstm"])
        window_sec  = trial.suggest_categorical("window_sec", [2, 4])

        # Build dataset with sampled window
        try:
            dataset = DREAMERDataset(
                mat_path    = mat_path,
                target      = target,
                window_sec  = float(window_sec),
                overlap_sec = float(window_sec) / 2,
                norm_method = config["data"]["norm_method"],
                threshold   = float(config["labels"]["threshold"]),
            )
        except Exception as e:
            logger.warning(f"Dataset build failed: {e}")
            return 0.0

        n_val   = max(1, int(len(dataset) * 0.2))
        n_train = len(dataset) - n_val

        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

        sampler = make_weighted_sampler(dataset)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            sampler=sampler, num_workers=2,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size,
            shuffle=False, num_workers=2,
        )

        # Model
        model = FusionModel(
            n_eeg_channels = 14,
            n_ecg_channels = 2,
            branch_type    = branch_type,
            branch_dim     = branch_dim,
            n_classes      = 2,
            dropout        = dropout,
        ).to(device)

        # Loss with class weights
        cw        = dataset.class_weights().to(device)
        criterion = nn.CrossEntropyLoss(weight=cw)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=15
        )

        best_auc = 0.0
        patience_counter = 0

        for epoch in range(20):
            _run_epoch(model, train_loader, criterion,
                       optimizer, device, is_train=True)
            scheduler.step()

            # Collect val predictions
            model.eval()
            y_true_v, y_prob_v = [], []
            with torch.no_grad():
                for eeg, ecg, labels in val_loader:
                    eeg, ecg = eeg.to(device), ecg.to(device)
                    logits   = model(eeg, ecg)
                    probs    = torch.softmax(logits, dim=1)[:, 1]
                    y_true_v.extend(labels.numpy())
                    y_prob_v.extend(probs.cpu().numpy())

            y_true_v = np.array(y_true_v)
            y_prob_v = np.array(y_prob_v)

            if len(np.unique(y_true_v)) < 2:
                return 0.0

            auc = float(roc_auc_score(y_true_v, y_prob_v))
            trial.report(auc, epoch)

            if auc > best_auc:
                best_auc = auc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    break

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_auc

    # ── Run study ─────────────────────────────────────────────────
    sampler_optuna = optuna.samplers.TPESampler(seed=42)
    pruner         = optuna.pruners.MedianPruner(n_warmup_steps=5)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler_optuna,
        pruner=pruner,
        study_name=f"fusion_{target}",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    result = {
        "target"      : target,
        "best_auc"    : round(best.value, 4),
        "best_params" : best.params,
        "n_trials"    : n_trials,
        "n_completed" : len(study.trials),
    }

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"optuna_fusion_{target}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(
        f"Best AUC={result['best_auc']} | params={result['best_params']}"
    )
    logger.info(f"Saved → {out}")
    return result


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning")
    parser.add_argument("--mode",      choices=["baseline", "deep"],
                        required=True)
    parser.add_argument("--target",    choices=["valence","arousal","dominance"],
                        default="valence")
    parser.add_argument("--model_type",choices=["logreg","svm","rf","gbm"],
                        default="rf",
                        help="Baseline model (ignored for deep mode)")
    parser.add_argument("--n_trials",  type=int, default=30,
                        help="Optuna trials (deep mode only)")
    parser.add_argument("--config",    type=str,
                        default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.mode == "baseline":
        # Build feature matrix from cached data
        import pandas as pd
        from src.data.save_processed import eeg_path, ecg_path
        from src.features.eeg_features import extract_eeg_features
        from src.features.ecg_features import extract_ecg_features

        processed_dir = cfg["data"]["processed_path"]
        df_labels     = pd.read_csv(
            os.path.join(processed_dir, "labels.csv")
        )
        label_col = f"{args.target}_bin"
        rows, ys  = [], []

        for _, row in df_labels.iterrows():
            sub, vid = int(row["subject"]), int(row["video"])
            ef = eeg_path(processed_dir, sub, vid)
            cf = ecg_path(processed_dir, sub, vid)
            if not os.path.exists(ef):
                continue
            eeg_segs = np.load(ef)
            ecg_segs = np.load(cf)
            n = min(eeg_segs.shape[0], ecg_segs.shape[0])
            for w in range(n):
                feat = np.concatenate([
                    extract_eeg_features(eeg_segs[w],
                                         fs=cfg["data"]["sampling_rate_eeg"]),
                    extract_ecg_features(ecg_segs[w],
                                         fs=cfg["data"]["sampling_rate_ecg"]),
                ])
                rows.append(feat)
                ys.append(int(row[label_col]))

        X = np.stack(rows)
        y = np.array(ys)
        tune_baseline(X, y, model_type=args.model_type, target=args.target)

    else:
        tune_deep(
            mat_path = cfg["data"]["raw_path"],
            target   = args.target,
            config   = cfg,
            n_trials = args.n_trials,
        )