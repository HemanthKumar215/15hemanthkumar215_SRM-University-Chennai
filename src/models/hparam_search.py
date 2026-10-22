"""
hparam_search.py – Hyperparameter Optimization via Optuna
===========================================================
Performs automated hyperparameter tuning (learning rate, GRU hidden size,
dropout, batch size, and loss class-weight) with early stopping and tracking.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config import get_project_root, load_config
from src.utils.logging_setup import setup_logging
from src.features.lazy_dataset import LazySpikeDataset
from src.models.fusion_model import FusionModel
from src.models.train_pipeline import evaluate_dataset, train_one_epoch

logger = logging.getLogger(__name__)

# Suppress verbose Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial: optuna.Trial, config: Dict[str, Any], root: Path, device: torch.device) -> float:
    """Optuna objective function: returns validation AUROC."""
    # Hyperparameter search space
    lr = trial.suggest_float("learning_rate", 2e-4, 3e-3, log=True)
    gru_hidden = trial.suggest_categorical("gru_hidden_dim", [32, 64])
    gru_dropout = trial.suggest_float("gru_dropout", 0.1, 0.4, step=0.1)
    fusion_dropout = trial.suggest_float("fusion_dropout", 0.1, 0.4, step=0.1)
    batch_size = trial.suggest_categorical("batch_size", [64, 128])

    ehr_df = pd.read_csv(root / config["data"]["output_dir"] / "ehr_profiles.csv")
    cgm_path = root / config["data"]["output_dir"] / "cgm_wearable.parquet"

    # Patient split
    pids = ehr_df["patient_id"].unique()
    rng = np.random.default_rng(config["project"]["seed"])
    shuffled = rng.permutation(pids)
    train_pids = shuffled[60:].tolist()
    val_pids = shuffled[:60].tolist()

    train_ds = LazySpikeDataset(cgm_path, ehr_df, train_pids, config, max_samples_per_patient=20)
    val_ds = LazySpikeDataset(cgm_path, ehr_df, val_pids, config, max_samples_per_patient=10)
    val_ds.set_normalization(train_ds.ts_mean, train_ds.ts_std, train_ds.static_mean, train_ds.static_std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = FusionModel(
        gru_input_dim=5,
        gru_hidden_dim=gru_hidden,
        gru_num_layers=2,
        gru_dropout=gru_dropout,
        static_input_dim=train_ds.static_dim,
        static_hidden_dim=64,
        static_embed_dim=32,
        fusion_hidden_dim=64,
        fusion_dropout=fusion_dropout,
    ).to(device)

    pos_w = torch.tensor([train_ds.pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_auroc = 0.0
    for epoch in range(1, 4):  # 3 fast epochs per trial
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        v = evaluate_dataset(model, val_loader, device)
        if v["auroc"] > best_val_auroc:
            best_val_auroc = v["auroc"]

        trial.report(best_val_auroc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_auroc


def run_hparam_search(n_trials: int = 5) -> Dict[str, Any]:
    """Execute Optuna hyperparameter study and record results."""
    setup_logging(log_file="optuna_search.log")
    config = load_config()
    root = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Starting Optuna search (%d trials) on device: %s", n_trials, device)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, config, root, device), n_trials=n_trials)

    best_trial = study.best_trial
    logger.info("Optuna search completed. Best Trial #%d -> Val AUROC: %.4f",
                best_trial.number, best_trial.value)
    logger.info("Best parameters: %s", best_trial.params)

    out_file = root / "reports" / "optuna_best_params.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "best_trial_number": best_trial.number,
            "best_val_auroc": best_trial.value,
            "best_params": best_trial.params,
        }, f, indent=2)

    print(f"\n[OK] Optuna study finished. Best Val AUROC: {best_trial.value:.4f}")
    print(f"     Saved best configuration to: {out_file}\n")
    return best_trial.params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna Hyperparameter Optimization.")
    parser.add_argument("--n_trials", type=int, default=5, help="Number of search trials.")
    args = parser.parse_args()
    run_hparam_search(n_trials=args.n_trials)
