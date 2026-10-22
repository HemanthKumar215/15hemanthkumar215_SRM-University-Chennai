"""
baseline.py – LightGBM / XGBoost baseline model
==================================================
Trains gradient-boosted trees on the engineered tabular features.
Reports AUROC, AUPRC, F1, precision, recall.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, classification_report,
)

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

logger = logging.getLogger(__name__)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature column names (exclude metadata and label)."""
    exclude = {"patient_id", "timestamp", "day", "label"}
    return [c for c in df.columns if c not in exclude]


def train_lightgbm(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    val_features: pd.DataFrame,
    val_labels: pd.Series,
    config: Dict[str, Any] | None = None,
) -> Tuple[Any, Dict[str, float]]:
    """Train a LightGBM classifier.

    Returns
    -------
    model : lgb.Booster
    metrics : dict
        Validation metrics.
    """
    if not _HAS_LGB:
        raise ImportError("lightgbm not installed – pip install lightgbm")

    feat_cols = _feature_cols(train_features)
    X_train = train_features[feat_cols].values
    X_val = val_features[feat_cols].values

    # Fill NaN
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)

    n_pos = int(train_labels.sum())
    n_neg = int((train_labels == 0).sum())
    scale_pos = max(1, n_neg) / max(1, n_pos)

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": scale_pos,
        "verbose": -1,
        "seed": 42,
        "n_jobs": -1,
    }

    dtrain = lgb.Dataset(X_train, label=train_labels.values, feature_name=feat_cols)
    dval = lgb.Dataset(X_val, label=val_labels.values, reference=dtrain, feature_name=feat_cols)

    logger.info("Training LightGBM (scale_pos_weight=%.2f) ...", scale_pos)

    callbacks = [lgb.log_evaluation(period=50)]
    model = lgb.train(
        params, dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=callbacks + [lgb.early_stopping(stopping_rounds=30)],
    )

    # Evaluate
    y_prob = model.predict(X_val)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = _compute_metrics(val_labels.values, y_prob, y_pred)
    metrics["feature_names"] = feat_cols

    logger.info("LightGBM val metrics: %s",
                {k: v for k, v in metrics.items() if k != "feature_names"})
    return model, metrics


def train_xgboost(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    val_features: pd.DataFrame,
    val_labels: pd.Series,
) -> Tuple[Any, Dict[str, float]]:
    """Train an XGBoost classifier (similar interface to LightGBM)."""
    if not _HAS_XGB:
        raise ImportError("xgboost not installed – pip install xgboost")

    feat_cols = _feature_cols(train_features)
    X_train = np.nan_to_num(train_features[feat_cols].values, nan=0.0)
    X_val = np.nan_to_num(val_features[feat_cols].values, nan=0.0)

    n_pos = max(1, int(train_labels.sum()))
    n_neg = max(1, int((train_labels == 0).sum()))

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=n_neg / n_pos,
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, train_labels.values,
              eval_set=[(X_val, val_labels.values)], verbose=False)

    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = _compute_metrics(val_labels.values, y_prob, y_pred)
    metrics["feature_names"] = feat_cols

    logger.info("XGBoost val metrics: %s",
                {k: v for k, v in metrics.items() if k != "feature_names"})
    return model, metrics


def _compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute standard binary-classification metrics."""
    return {
        "auroc": round(roc_auc_score(y_true, y_prob), 4),
        "auprc": round(average_precision_score(y_true, y_prob), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
    }
