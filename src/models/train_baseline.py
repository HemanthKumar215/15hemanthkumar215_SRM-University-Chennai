"""
train_baseline.py – Train and evaluate the LightGBM baseline
==============================================================
Run: ``python -m src.models.train_baseline``

Patient-level train/val/test split to avoid data leakage.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from src.utils.config import get_project_root, load_config
from src.utils.logging_setup import setup_logging
from src.features.feature_engineering import build_feature_matrix
from src.models.baseline import train_lightgbm

logger = logging.getLogger(__name__)


def _patient_split(
    features: pd.DataFrame,
    labels: pd.Series,
    test_frac: float = 0.20,
    val_frac: float = 0.10,
    seed: int = 42,
):
    """Split by patient_id (never by row) for train / val / test."""
    rng = np.random.default_rng(seed)
    pids = features["patient_id"].unique()
    rng.shuffle(pids)

    n = len(pids)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)

    test_pids = set(pids[:n_test])
    val_pids = set(pids[n_test: n_test + n_val])
    train_pids = set(pids[n_test + n_val:])

    train_mask = features["patient_id"].isin(train_pids)
    val_mask = features["patient_id"].isin(val_pids)
    test_mask = features["patient_id"].isin(test_pids)

    return (
        features[train_mask], labels[train_mask],
        features[val_mask], labels[val_mask],
        features[test_mask], labels[test_mask],
    )


def main() -> None:
    setup_logging(log_file="train_baseline.log")
    config = load_config()
    root = get_project_root()

    models_dir = root / config["paths"]["models_dir"]
    reports_dir = root / config["paths"]["reports_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    ehr_path = root / config["data"]["output_dir"] / "ehr_profiles.csv"
    cgm_path = root / config["data"]["output_dir"] / "cgm_wearable.parquet"

    logger.info("Loading data ...")
    ehr_df = pd.read_csv(ehr_path)
    cgm_df = pd.read_parquet(cgm_path)

    t0 = time.perf_counter()

    # Build feature matrix
    features, labels = build_feature_matrix(cgm_df, ehr_df, config)

    # Patient-level split
    (X_train, y_train,
     X_val, y_val,
     X_test, y_test) = _patient_split(
        features, labels,
        test_frac=config["model"]["test_fraction"],
        val_frac=config["model"]["val_fraction"],
        seed=config["project"]["seed"],
    )

    logger.info("Split sizes: train=%d, val=%d, test=%d",
                len(X_train), len(X_val), len(X_test))

    # Train LightGBM
    model, val_metrics = train_lightgbm(X_train, y_train, X_val, y_val, config)

    # Evaluate on test set
    feat_cols = val_metrics.pop("feature_names")
    X_test_arr = np.nan_to_num(X_test[feat_cols].values, nan=0.0)
    y_test_prob = model.predict(X_test_arr)
    y_test_pred = (y_test_prob >= 0.5).astype(int)

    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                  f1_score, precision_score, recall_score)
    test_metrics = {
        "auroc": round(roc_auc_score(y_test.values, y_test_prob), 4),
        "auprc": round(average_precision_score(y_test.values, y_test_prob), 4),
        "f1": round(f1_score(y_test.values, y_test_pred), 4),
        "precision": round(precision_score(y_test.values, y_test_pred), 4),
        "recall": round(recall_score(y_test.values, y_test_pred), 4),
    }

    logger.info("Test metrics: %s", test_metrics)

    # Save model
    model.save_model(str(models_dir / "baseline_lgbm.txt"))
    logger.info("Model saved -> %s", models_dir / "baseline_lgbm.txt")

    # Save metrics
    all_metrics = {"val": val_metrics, "test": test_metrics}
    with open(reports_dir / "baseline_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Feature importance plot
    importance = model.feature_importance(importance_type="gain")
    feat_imp = pd.DataFrame({
        "feature": feat_cols,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = min(25, len(feat_imp))
    ax.barh(feat_imp["feature"].iloc[:top_n][::-1],
            feat_imp["importance"].iloc[:top_n][::-1])
    ax.set_xlabel("Gain")
    ax.set_title("LightGBM – Top Feature Importances")
    plt.tight_layout()
    fig.savefig(reports_dir / "baseline_feature_importance.png", dpi=150)
    plt.close(fig)

    # Calibration curve
    fig, ax = plt.subplots(figsize=(7, 7))
    fraction_pos, mean_predicted = calibration_curve(y_test.values, y_test_prob,
                                                      n_bins=10, strategy="uniform")
    ax.plot(mean_predicted, fraction_pos, "s-", label="LightGBM")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve – LightGBM Baseline")
    ax.legend()
    plt.tight_layout()
    fig.savefig(reports_dir / "baseline_calibration.png", dpi=150)
    plt.close(fig)

    elapsed = time.perf_counter() - t0
    logger.info("Baseline training complete in %.1f s", elapsed)

    print("\n── Baseline Results ──")
    print(f"  Val  AUROC: {val_metrics['auroc']}")
    print(f"  Test AUROC: {test_metrics['auroc']}")
    print(f"  Test AUPRC: {test_metrics['auprc']}")
    print(f"  Test F1   : {test_metrics['f1']}")
    print(f"  Elapsed   : {elapsed:.1f} s")


if __name__ == "__main__":
    main()
