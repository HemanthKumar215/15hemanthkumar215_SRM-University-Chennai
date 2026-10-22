"""
tracker.py – Experiment Tracking for Digital Twin ML Pipeline
=============================================================
Logs runs, hyperparameters, epochs, and metrics to structured CSV and JSON.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _sanitize(obj: Any) -> Any:
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), 4)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items() if k not in ("y_true", "y_prob")}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


class ExperimentTracker:
    """Lightweight experiment tracking logger (CSV + JSON)."""

    def __init__(self, log_dir: Path | str) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "experiment_runs.csv"
        self._init_csv()

    def _init_csv(self) -> None:
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "run_id", "stage", "model_type",
                    "epochs", "batch_size", "lr", "best_val_auroc",
                    "test_auroc", "test_auprc", "test_f1", "notes"
                ])

    def log_run(
        self,
        run_id: str,
        stage: str,
        model_type: str,
        hparams: Dict[str, Any],
        metrics: Dict[str, Any],
        notes: str = "",
    ) -> None:
        """Log a completed run to CSV and JSON."""
        ts = datetime.now().isoformat()
        clean_metrics = _sanitize(metrics)
        clean_hparams = _sanitize(hparams)

        row = [
            ts,
            run_id,
            stage,
            model_type,
            clean_hparams.get("epochs", ""),
            clean_hparams.get("batch_size", ""),
            clean_hparams.get("learning_rate", clean_hparams.get("lr", "")),
            clean_metrics.get("val_auroc", clean_metrics.get("best_val_auroc", "")),
            clean_metrics.get("test", {}).get("auroc", clean_metrics.get("test_auroc", "")),
            clean_metrics.get("test", {}).get("auprc", clean_metrics.get("test_auprc", "")),
            clean_metrics.get("test", {}).get("f1", clean_metrics.get("test_f1", "")),
            notes,
        ]
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # Also write run-specific json
        run_file = self.log_dir / f"run_{run_id}.json"
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": ts,
                "run_id": run_id,
                "stage": stage,
                "model_type": model_type,
                "hparams": clean_hparams,
                "metrics": clean_metrics,
                "notes": notes,
            }, f, indent=2)

        logger.info("Logged run %s to %s", run_id, self.csv_path)
