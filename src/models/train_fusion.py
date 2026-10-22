"""
train_fusion.py – Train the PyTorch fusion model + ablation study
==================================================================
Run: ``python -m src.models.train_fusion``

Trains three model variants:
1. **Fusion** (GRU + MLP) – main model
2. **Sensor-only** (GRU only) – ablation
3. **Static-only** (MLP only) – ablation

Uses patient-level train/val/test split and handles class imbalance
via weighted BCE loss.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score,
)
from sklearn.calibration import calibration_curve

from src.utils.config import get_project_root, load_config
from src.utils.logging_setup import setup_logging
from src.features.dataset import SpikeDataset
from src.models.fusion_model import build_model_from_config

logger = logging.getLogger(__name__)


def _patient_split(
    patient_ids: Any,
    test_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """Split patient IDs into train/val/test."""
    rng = np.random.default_rng(seed)
    pids = list(patient_ids)
    rng.shuffle(pids)
    n = len(pids)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    return (
        pids[n_test + n_val:],
        pids[n_test: n_test + n_val],
        pids[:n_test],
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for ts, static, labels in loader:
        ts = ts.to(device)
        static = static.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        logits = model(ts, static).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on a data loader, return metrics + predictions."""
    model.eval()
    all_labels: List[int] = []
    all_probs: List[float] = []

    for ts, static, labels in loader:
        ts = ts.to(device)
        static = static.to(device)
        logits = model(ts, static).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    if len(np.unique(y_true)) < 2:
        return {"auroc": 0.5, "auprc": 0.0, "f1": 0.0,
                "precision": 0.0, "recall": 0.0,
                "y_true": y_true, "y_prob": y_prob}

    return {
        "auroc": round(roc_auc_score(y_true, y_prob), 4),
        "auprc": round(average_precision_score(y_true, y_prob), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "y_true": y_true,
        "y_prob": y_prob,
    }


def train_model(
    model_type: str,
    train_ds: SpikeDataset,
    val_ds: SpikeDataset,
    test_ds: SpikeDataset,
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Full training loop for one model variant."""
    model_cfg = config["model"]
    epochs = min(model_cfg.get("epochs", 30), 10)
    batch_size = 256
    lr = model_cfg["learning_rate"]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0)

    # Update static_input_dim based on actual data
    actual_static_dim = train_ds.static_dim
    cfg_override = dict(model_cfg)
    cfg_override["static_input_dim"] = actual_static_dim

    model = build_model_from_config(cfg_override, model_type).to(device)
    logger.info("Model: %s  params: %d", model_type,
                sum(p.numel() for p in model.parameters()))

    pos_w = torch.tensor([train_ds.pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=model_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_auroc = 0.0
    best_state = None
    patience_counter = 0

    logger.info("Training %s for %d epochs ...", model_type, epochs)
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        val_auroc = val_metrics["auroc"]
        scheduler.step(val_auroc)

        if epoch % 5 == 0 or epoch == 1:
            logger.info("  Epoch %2d/%d  loss=%.4f  val_auroc=%.4f  val_f1=%.4f",
                        epoch, epochs, loss, val_auroc, val_metrics["f1"])

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                logger.info("  Early stopping at epoch %d", epoch)
                break

    # Load best
    if best_state:
        model.load_state_dict(best_state)
    model.to(device)

    # Test evaluation
    test_metrics = evaluate(model, test_loader, device)
    logger.info("%s test metrics: AUROC=%.4f  AUPRC=%.4f  F1=%.4f",
                model_type, test_metrics["auroc"], test_metrics["auprc"],
                test_metrics["f1"])

    # Store normalization stats alongside model
    norm_stats = {
        "ts_mean": train_ds.ts_mean.tolist(),
        "ts_std": train_ds.ts_std.tolist(),
        "static_mean": train_ds.static_mean.tolist(),
        "static_std": train_ds.static_std.tolist(),
        "static_dim": actual_static_dim,
    }

    return model, {
        "val_auroc": best_auroc,
        "test": {k: v for k, v in test_metrics.items()
                 if k not in ("y_true", "y_prob")},
        "y_true": test_metrics.get("y_true"),
        "y_prob": test_metrics.get("y_prob"),
        "norm_stats": norm_stats,
    }


def main() -> None:
    setup_logging(log_file="train_fusion.log")
    config = load_config()
    root = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    models_dir = root / config["paths"]["models_dir"]
    reports_dir = root / config["paths"]["reports_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    ehr_df = pd.read_csv(root / config["data"]["output_dir"] / "ehr_profiles.csv")
    cgm_df = pd.read_parquet(root / config["data"]["output_dir"] / "cgm_wearable.parquet")

    patient_ids = ehr_df["patient_id"].unique()
    train_pids, val_pids, test_pids = _patient_split(
        patient_ids, config["model"]["test_fraction"],
        config["model"]["val_fraction"], config["project"]["seed"],
    )
    logger.info("Patient split: train=%d, val=%d, test=%d",
                len(train_pids), len(val_pids), len(test_pids))

    logger.info("Building datasets once for all ablation variants ...")
    train_ds = SpikeDataset(cgm_df, ehr_df, train_pids, config,
                            max_samples_per_patient=40)
    val_ds = SpikeDataset(cgm_df, ehr_df, val_pids, config,
                          max_samples_per_patient=20)
    test_ds = SpikeDataset(cgm_df, ehr_df, test_pids, config,
                           max_samples_per_patient=20)

    val_ds.set_normalization(train_ds.ts_mean, train_ds.ts_std,
                             train_ds.static_mean, train_ds.static_std)
    test_ds.set_normalization(train_ds.ts_mean, train_ds.ts_std,
                              train_ds.static_mean, train_ds.static_std)

    t0 = time.perf_counter()

    # Train all three model variants
    results = {}
    for model_type in ["fusion", "sensor_only", "static_only"]:
        logger.info("=" * 60)
        logger.info("Training: %s", model_type)
        logger.info("=" * 60)
        model, metrics = train_model(
            model_type, train_ds, val_ds, test_ds,
            config, device,
        )

        # Save model
        save_path = models_dir / f"{model_type}_model.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config["model"],
            "model_type": model_type,
            "norm_stats": metrics["norm_stats"],
        }, save_path)
        logger.info("Saved -> %s", save_path)

        results[model_type] = {
            "val_auroc": metrics["val_auroc"],
            "test": metrics["test"],
            "y_true": metrics["y_true"],
            "y_prob": metrics["y_prob"],
        }

    # ── Comparison table ──────────────────────────────────────────────────
    print("\n── Ablation Study Results ──")
    print(f"{'Model':<15} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
    print("-" * 58)
    for mt in ["static_only", "sensor_only", "fusion"]:
        t = results[mt]["test"]
        print(f"{mt:<15} {t['auroc']:>8.4f} {t['auprc']:>8.4f} "
              f"{t['f1']:>8.4f} {t['precision']:>8.4f} {t['recall']:>8.4f}")

    # Save metrics JSON
    json_results = {}
    for mt in results:
        json_results[mt] = {
            "val_auroc": results[mt]["val_auroc"],
            "test": results[mt]["test"],
        }
    with open(reports_dir / "fusion_metrics.json", "w") as f:
        json.dump(json_results, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────────
    # Calibration curves
    fig, ax = plt.subplots(figsize=(7, 7))
    for mt in ["fusion", "sensor_only", "static_only"]:
        y_true = results[mt]["y_true"]
        y_prob = results[mt]["y_prob"]
        if y_true is not None and len(np.unique(y_true)) > 1:
            frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10,
                                                     strategy="uniform")
            ax.plot(mean_pred, frac_pos, "o-", label=mt)
    ax.plot([0, 1], [0, 1], "k--", label="Ideal")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curves – Model Comparison")
    ax.legend()
    plt.tight_layout()
    fig.savefig(reports_dir / "fusion_calibration.png", dpi=150)
    plt.close(fig)

    # Ablation bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    model_names = ["Static Only", "Sensor Only", "Fusion"]
    model_keys = ["static_only", "sensor_only", "fusion"]
    aurocs = [results[k]["test"]["auroc"] for k in model_keys]
    auprcs = [results[k]["test"]["auprc"] for k in model_keys]

    x = np.arange(len(model_names))
    w = 0.35
    ax.bar(x - w/2, aurocs, w, label="AUROC", color="#4C78A8")
    ax.bar(x + w/2, auprcs, w, label="AUPRC", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study – Value of Data Fusion")
    ax.legend()
    for i, (a, p) in enumerate(zip(aurocs, auprcs)):
        ax.text(i - w/2, a + 0.02, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + w/2, p + 0.02, f"{p:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(reports_dir / "ablation_comparison.png", dpi=150)
    plt.close(fig)

    elapsed = time.perf_counter() - t0
    logger.info("All training complete in %.1f s", elapsed)
    print(f"\n  Total time: {elapsed:.1f} s")
    print(f"  Models saved to: {models_dir}")
    print(f"  Reports saved to: {reports_dir}")


if __name__ == "__main__":
    main()
