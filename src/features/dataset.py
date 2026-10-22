"""
dataset.py – PyTorch Dataset for the fusion model
===================================================
Provides ``SpikeDataset`` which yields (time_series_window, static_vector, label)
tuples suitable for the two-branch GRU+MLP fusion network.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.config import load_config
from src.features.feature_engineering import encode_static_features

logger = logging.getLogger(__name__)

# Sensor channels used by the GRU branch
_SENSOR_COLS = ["glucose_mg_dl", "heart_rate_bpm", "hrv_ms", "steps", "sleep_stage"]

# Static feature columns (from encode_static_features, minus patient_id)
_STATIC_FEATURE_COLS: Optional[List[str]] = None  # populated at runtime


class SpikeDataset(Dataset):
    """PyTorch dataset for hyperglycemic spike prediction.

    Each sample consists of:
    - ``time_series``: (lookback_readings, n_sensor_channels) float tensor
    - ``static``: (n_static_features,) float tensor
    - ``label``: scalar long tensor (0 or 1)
    """

    def __init__(
        self,
        cgm_df: pd.DataFrame,
        ehr_df: pd.DataFrame,
        patient_ids: List[str],
        config: Dict[str, Any] | None = None,
        max_samples_per_patient: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        cgm_df : pd.DataFrame
            Full CGM/wearable DataFrame (all patients).
        ehr_df : pd.DataFrame
            EHR profiles.
        patient_ids : list of str
            Subset of patient IDs to include (for train/val/test split).
        config : dict, optional
        max_samples_per_patient : int, optional
            Cap samples per patient for faster training.  ``None`` = no cap.
        """
        super().__init__()
        if config is None:
            config = load_config()

        model_cfg = config["model"]
        self.lookback = int(model_cfg["lookback_hours"] * 12)
        self.horizon = int(model_cfg["prediction_horizon_hours"] * 12)
        self.threshold = model_cfg["glucose_threshold"]

        # Encode static features
        static_df = encode_static_features(ehr_df)
        global _STATIC_FEATURE_COLS
        _STATIC_FEATURE_COLS = [c for c in static_df.columns if c != "patient_id"]
        self.static_dim = len(_STATIC_FEATURE_COLS)

        # Build per-patient index
        self.samples: List[Tuple[np.ndarray, np.ndarray, int]] = []

        static_lookup = {
            row["patient_id"]: row[_STATIC_FEATURE_COLS].values.astype(np.float32)
            for _, row in static_df.iterrows()
        }
        pid_set = set(patient_ids)
        filtered_cgm = cgm_df[cgm_df["patient_id"].isin(pid_set)]

        for pid, grp in filtered_cgm.groupby("patient_id"):
            grp = grp.sort_values("timestamp")
            if len(grp) < self.lookback + self.horizon:
                continue

            static_vec = static_lookup.get(pid)
            if static_vec is None:
                continue

            # Sensor matrix (forward-fill NaN, then zero remaining)
            sensor = grp[_SENSOR_COLS].ffill().fillna(0).values.astype(np.float32)

            # Glucose for labels
            glu = grp["glucose_mg_dl"].values

            # Generate samples: every 6th reading (30 min apart) to reduce redundancy
            indices = list(range(self.lookback, len(grp) - self.horizon, 6))
            if max_samples_per_patient and len(indices) > max_samples_per_patient:
                rng = np.random.default_rng(hash(pid) & 0x7FFFFFFF)
                indices = sorted(rng.choice(indices, max_samples_per_patient, replace=False))

            for i in indices:
                window = sensor[i - self.lookback: i]
                current_glu = glu[i - 1] if i > 0 else np.nan
                # Label: glucose crosses threshold in next horizon,
                # given it is currently below threshold (not already hyperglycemic)
                if not np.isnan(current_glu) and current_glu >= self.threshold:
                    label = 0   # already hyperglycemic – exclude from spike prediction
                else:
                    future = glu[i: i + self.horizon]
                    valid_future = future[~np.isnan(future)]
                    label = int(len(valid_future) > 0 and np.max(valid_future) > self.threshold)
                self.samples.append((window, static_vec, label))

        # Compute normalisation stats
        if self.samples:
            sub_n = min(len(self.samples), 5000)
            sub_indices = np.random.default_rng(42).choice(len(self.samples), sub_n, replace=False)
            sub_ts = np.stack([self.samples[i][0] for i in sub_indices])
            self.ts_mean = sub_ts.mean(axis=(0, 1))
            self.ts_std = sub_ts.std(axis=(0, 1)).clip(1e-6)
            sub_static = np.stack([self.samples[i][1] for i in sub_indices])
            self.static_mean = sub_static.mean(axis=0)
            self.static_std = sub_static.std(axis=0).clip(1e-6)
        else:
            self.ts_mean = np.zeros(len(_SENSOR_COLS))
            self.ts_std = np.ones(len(_SENSOR_COLS))
            self.static_mean = np.zeros(self.static_dim)
            self.static_std = np.ones(self.static_dim)

        labels = [s[2] for s in self.samples]
        n_pos = sum(labels)
        logger.info("SpikeDataset: %d samples, %d positive (%.1f%%)",
                    len(self.samples), n_pos,
                    100 * n_pos / max(1, len(self.samples)))

    def set_normalization(self, ts_mean: np.ndarray, ts_std: np.ndarray,
                          static_mean: np.ndarray, static_std: np.ndarray) -> None:
        """Override normalisation stats (e.g., use train set stats for val/test)."""
        self.ts_mean = ts_mean
        self.ts_std = ts_std
        self.static_mean = static_mean
        self.static_std = static_std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ts, static, label = self.samples[idx]

        # Normalise
        ts_norm = (ts - self.ts_mean) / self.ts_std
        static_norm = (static - self.static_mean) / self.static_std

        return (
            torch.tensor(ts_norm, dtype=torch.float32),
            torch.tensor(static_norm, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )

    @property
    def pos_weight(self) -> float:
        """Compute positive class weight for BCE loss."""
        labels = [s[2] for s in self.samples]
        n_pos = max(1, sum(labels))
        n_neg = max(1, len(labels) - n_pos)
        return n_neg / n_pos
