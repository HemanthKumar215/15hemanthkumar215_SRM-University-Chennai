"""
lazy_dataset.py – Memory-Efficient Lazy PyTorch Dataset for Large Cohorts
==========================================================================
Enables training on 10,000+ patient cohorts (tens of millions of CGM rows)
without running out of laptop RAM by loading slices on demand or in patient chunks.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.config import load_config
from src.features.feature_engineering import encode_static_features

logger = logging.getLogger(__name__)

_SENSOR_COLS = ["glucose_mg_dl", "heart_rate_bpm", "hrv_ms", "steps", "sleep_stage"]


class PatientCache:
    """Simple LRU cache for recently accessed patient time-series DataFrames."""

    def __init__(self, max_size: int = 16) -> None:
        self.max_size = max_size
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()

    def get(self, patient_id: str) -> Optional[pd.DataFrame]:
        if patient_id in self._cache:
            self._cache.move_to_end(patient_id)
            return self._cache[patient_id]
        return None

    def put(self, patient_id: str, df: pd.DataFrame) -> None:
        if patient_id in self._cache:
            self._cache.move_to_end(patient_id)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[patient_id] = df


class LazySpikeDataset(Dataset):
    """Memory-efficient PyTorch Dataset for large synthetic or real cohorts.

    Instead of pre-allocating hundreds of thousands of windows in RAM,
    it indexes sample positions (patient_id, start_idx, label) and loads
    the required sensor windows lazily via an LRU cache or slice lookup.
    """

    def __init__(
        self,
        cgm_source: Union[pd.DataFrame, Path, str],
        ehr_df: pd.DataFrame,
        patient_ids: List[str],
        config: Dict[str, Any] | None = None,
        max_samples_per_patient: int | None = 40,
        patient_chunk_dir: Optional[Path] = None,
    ) -> None:
        """
        Parameters
        ----------
        cgm_source : pd.DataFrame, Path, or str
            Full CGM DataFrame OR path to a single Parquet / partition directory.
        ehr_df : pd.DataFrame
            EHR profiles DataFrame.
        patient_ids : list of str
            Target patient IDs for this dataset partition.
        config : dict, optional
        max_samples_per_patient : int, optional
            Number of sampled lookback windows per patient.
        patient_chunk_dir : Path, optional
            Optional directory containing per-patient or chunked Parquets.
        """
        super().__init__()
        if config is None:
            config = load_config()

        model_cfg = config["model"]
        self.lookback = int(model_cfg["lookback_hours"] * 12)  # e.g., 288 for 24h
        self.horizon = int(model_cfg["prediction_horizon_hours"] * 12)  # e.g., 24 for 2h
        self.threshold = float(model_cfg["glucose_threshold"])

        # Encode static clinical features
        static_df = encode_static_features(ehr_df)
        self.static_cols = [c for c in static_df.columns if c != "patient_id"]
        self.static_dim = len(self.static_cols)

        # Lookup map for static vectors: pid -> np.ndarray
        self.static_lookup: Dict[str, np.ndarray] = {
            row["patient_id"]: row[self.static_cols].values.astype(np.float32)
            for _, row in static_df.iterrows()
        }

        self.patient_ids = set(patient_ids)
        self.patient_chunk_dir = Path(patient_chunk_dir) if patient_chunk_dir else None
        self._cgm_df: Optional[pd.DataFrame] = None
        self._parquet_path: Optional[Path] = None
        self._cache = PatientCache(max_size=32)

        if isinstance(cgm_source, pd.DataFrame):
            self._cgm_df = cgm_source[cgm_source["patient_id"].isin(self.patient_ids)].copy()
        elif isinstance(cgm_source, (str, Path)):
            p = Path(cgm_source)
            if p.is_file():
                self._parquet_path = p
            elif p.is_dir():
                self.patient_chunk_dir = p

        # Build lazy index: List of (patient_id, start_idx, label)
        self.index: List[Tuple[str, int, int]] = []
        self._build_index(max_samples_per_patient)

        # Normalization defaults
        self.ts_mean = np.zeros(len(_SENSOR_COLS), dtype=np.float32)
        self.ts_std = np.ones(len(_SENSOR_COLS), dtype=np.float32)
        self.static_mean = np.zeros(self.static_dim, dtype=np.float32)
        self.static_std = np.ones(self.static_dim, dtype=np.float32)

        # Compute sample stats
        self._compute_initial_stats()

    def _get_patient_series(self, pid: str) -> Optional[pd.DataFrame]:
        """Fetch a single patient's time series using memory caching."""
        cached = self._cache.get(pid)
        if cached is not None:
            return cached

        if self._cgm_df is not None:
            sub = self._cgm_df[self._cgm_df["patient_id"] == pid].sort_values("timestamp")
            if not sub.empty:
                self._cache.put(pid, sub)
                return sub

        if self.patient_chunk_dir is not None:
            # Check for per-patient file
            patient_file = self.patient_chunk_dir / f"{pid}.parquet"
            if patient_file.exists():
                df = pd.read_parquet(patient_file).sort_values("timestamp")
                self._cache.put(pid, df)
                return df

        if self._parquet_path is not None and self._parquet_path.exists():
            # Query directly from Parquet using pyarrow filters to load only this patient
            try:
                import pyarrow.dataset as ds
                dataset = ds.dataset(str(self._parquet_path), format="parquet")
                table = dataset.to_table(filter=ds.field("patient_id") == pid)
                df = table.to_pandas().sort_values("timestamp")
                if not df.empty:
                    self._cache.put(pid, df)
                    return df
            except Exception as e:
                logger.debug("PyArrow pushdown filter fallback: %s", e)
                if self._cgm_df is None:
                    full = pd.read_parquet(self._parquet_path)
                    self._cgm_df = full[full["patient_id"].isin(self.patient_ids)]
                    return self._get_patient_series(pid)

        return None

    def _build_index(self, max_samples_per_patient: Optional[int]) -> None:
        """Scan available patients to construct the index of valid prediction times."""
        # For indexing, if cgm_df is loaded, use it; otherwise inspect patient files
        target_pids = sorted(list(self.patient_ids))
        for pid in target_pids:
            grp = self._get_patient_series(pid)
            if grp is None or len(grp) < self.lookback + self.horizon:
                continue

            n_readings = len(grp)
            glu = grp["glucose_mg_dl"].values

            # Sample stride = 6 readings (30 min apart)
            step_indices = list(range(self.lookback, n_readings - self.horizon, 6))
            if max_samples_per_patient and len(step_indices) > max_samples_per_patient:
                rng = np.random.default_rng(hash(pid) & 0x7FFFFFFF)
                step_indices = sorted(rng.choice(step_indices, max_samples_per_patient, replace=False))

            for idx in step_indices:
                current_glu = glu[idx - 1] if idx > 0 else np.nan
                if not np.isnan(current_glu) and current_glu >= self.threshold:
                    label = 0  # Already hyperglycemic
                else:
                    future = glu[idx: idx + self.horizon]
                    valid = future[~np.isnan(future)]
                    label = int(len(valid) > 0 and np.max(valid) > self.threshold)
                self.index.append((pid, idx, label))

        labels = [item[2] for item in self.index]
        pos = sum(labels)
        total = len(self.index)
        pct = 100 * pos / max(1, total)
        logger.info("LazySpikeDataset: indexed %d samples across %d patients (%.1f%% positive)",
                    total, len(target_pids), pct)

    def _compute_initial_stats(self) -> None:
        """Estimate normalization stats from a sample of up to 1000 items."""
        if not self.index:
            self.pos_weight = 1.0
            return

        labels = [item[2] for item in self.index]
        pos = sum(labels)
        neg = len(labels) - pos
        self.pos_weight = float(np.clip(neg / max(1, pos), 1.0, 10.0))

        # Sample up to 500 items for sensor mean/std
        sample_indices = np.random.default_rng(42).choice(
            len(self.index), min(500, len(self.index)), replace=False
        )
        sample_windows = []
        for i in sample_indices:
            pid, idx, _ = self.index[i]
            grp = self._get_patient_series(pid)
            if grp is not None:
                sensor = grp[_SENSOR_COLS].ffill().fillna(0).values.astype(np.float32)
                window = sensor[idx - self.lookback: idx]
                if len(window) == self.lookback:
                    sample_windows.append(window)

        if sample_windows:
            arr = np.stack(sample_windows)  # (N, lookback, 5)
            self.ts_mean = arr.mean(axis=(0, 1)).astype(np.float32)
            self.ts_std = np.clip(arr.std(axis=(0, 1)), 1e-6, None).astype(np.float32)

        # Static stats from lookup
        if self.static_lookup:
            all_static = np.stack(list(self.static_lookup.values()))
            self.static_mean = all_static.mean(axis=0).astype(np.float32)
            self.static_std = np.clip(all_static.std(axis=0), 1e-6, None).astype(np.float32)

    def set_normalization(
        self,
        ts_mean: np.ndarray,
        ts_std: np.ndarray,
        static_mean: np.ndarray,
        static_std: np.ndarray,
    ) -> None:
        """Set normalisation statistics from training set."""
        self.ts_mean = np.asarray(ts_mean, dtype=np.float32)
        self.ts_std = np.asarray(ts_std, dtype=np.float32)
        self.static_mean = np.asarray(static_mean, dtype=np.float32)
        self.static_std = np.asarray(static_std, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (ts_window, static_vector, label)."""
        pid, start_idx, label = self.index[idx]
        grp = self._get_patient_series(pid)
        if grp is None:
            # Fallback zero window
            ts = np.zeros((self.lookback, len(_SENSOR_COLS)), dtype=np.float32)
        else:
            sensor = grp[_SENSOR_COLS].ffill().fillna(0).values.astype(np.float32)
            ts = sensor[start_idx - self.lookback: start_idx]
            if len(ts) < self.lookback:
                # Pad if slightly short
                pad = np.zeros((self.lookback - len(ts), len(_SENSOR_COLS)), dtype=np.float32)
                ts = np.vstack([pad, ts])

        # Normalize sensor window
        ts_norm = ((ts - self.ts_mean) / self.ts_std).astype(np.float32)

        # Static vector
        static = self.static_lookup.get(pid, np.zeros(self.static_dim, dtype=np.float32))
        static_norm = ((static - self.static_mean) / self.static_std).astype(np.float32)

        return (
            torch.from_numpy(ts_norm),
            torch.from_numpy(static_norm),
            torch.tensor(label, dtype=torch.long),
        )
