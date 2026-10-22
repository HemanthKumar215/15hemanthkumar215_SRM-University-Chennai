"""
feature_engineering.py – Extract features for glucose spike prediction
=======================================================================
Builds a tabular feature matrix from CGM/wearable time-series and static
EHR profiles.  Each sample represents a *prediction point*: given the
previous ``lookback_hours`` of sensor data and the patient's static profile,
predict whether glucose will exceed ``threshold`` mg/dL in the next
``horizon_hours``.

Feature categories
------------------
**Sensor-derived (rolling windows)**
- Glucose: mean, std, slope, min, max, range (1 h / 2 h / 4 h / 8 h)
- Time-in-range (70-180), time-above-range (>180), time-below-range (<70)
- Heart rate: mean, std
- HRV: mean, std
- Step count: sum in last 1 h / 4 h
- Sleep: hours asleep in last 8 h, deep-sleep fraction

**Static (from EHR)**
- age, sex (encoded), BMI, HbA1c, fasting glucose, lipids, BP,
  diagnoses, medications, family history, genetic risk score, etc.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress

from src.utils.config import load_config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Rolling sensor features
# ──────────────────────────────────────────────────────────────────────────────

def _rolling_glucose_features(glucose: pd.Series, windows: List[int]) -> pd.DataFrame:
    """Compute rolling mean, std, min, max, range, slope for glucose.

    Parameters
    ----------
    glucose : pd.Series
        Glucose readings (may contain NaN).
    windows : list of int
        Window sizes in number of readings (e.g., 12 = 1 hour at 5-min intervals).

    Returns
    -------
    pd.DataFrame
        Feature columns named ``gluc_{stat}_{w}r``.
    """
    feats: Dict[str, np.ndarray] = {}
    glu = glucose.ffill().bfill()  # fill NaN for rolling calcs
    glu_arr = glu.values.astype(float)
    n_pts = len(glu_arr)

    for w in windows:
        rolled = glu.rolling(window=w, min_periods=max(1, w // 2))
        feats[f"gluc_mean_{w}r"] = rolled.mean().values
        feats[f"gluc_std_{w}r"] = rolled.std().values
        feats[f"gluc_min_{w}r"] = rolled.min().values
        feats[f"gluc_max_{w}r"] = rolled.max().values
        feats[f"gluc_range_{w}r"] = (rolled.max() - rolled.min()).values

        # Slope: analytical linear regression over window of fixed length w via convolution
        slopes = np.full(n_pts, np.nan)
        if n_pts >= w and not np.isnan(glu_arr).all():
            x = np.arange(w)
            dx = x - (w - 1) / 2.0
            denom = np.sum(dx ** 2)
            if denom > 0:
                weights = dx / denom
                # weights[::-1] because np.convolve flips the kernel
                slopes[w - 1:] = np.convolve(glu_arr, weights[::-1], mode="valid")
        feats[f"gluc_slope_{w}r"] = slopes

    return pd.DataFrame(feats)


def _time_in_range_features(glucose: pd.Series, window: int) -> pd.DataFrame:
    """Compute time-in-range metrics over a rolling window.

    Ranges: 70-180 (in-range), >180 (above), <70 (below).
    """
    glu = glucose.ffill()
    feats: Dict[str, np.ndarray] = {}

    in_range = ((glu >= 70) & (glu <= 180)).astype(float)
    above = (glu > 180).astype(float)
    below = (glu < 70).astype(float)

    feats[f"tir_{window}r"] = in_range.rolling(window, min_periods=1).mean().values
    feats[f"tar_{window}r"] = above.rolling(window, min_periods=1).mean().values
    feats[f"tbr_{window}r"] = below.rolling(window, min_periods=1).mean().values

    return pd.DataFrame(feats)


def _hr_hrv_features(hr: pd.Series, hrv: pd.Series,
                     windows: List[int]) -> pd.DataFrame:
    """Rolling heart rate and HRV statistics."""
    feats: Dict[str, np.ndarray] = {}
    for w in windows:
        feats[f"hr_mean_{w}r"] = hr.rolling(w, min_periods=1).mean().values
        feats[f"hr_std_{w}r"] = hr.rolling(w, min_periods=1).std().values
        feats[f"hrv_mean_{w}r"] = hrv.rolling(w, min_periods=1).mean().values
        feats[f"hrv_std_{w}r"] = hrv.rolling(w, min_periods=1).std().values
    return pd.DataFrame(feats)


def _activity_sleep_features(steps: pd.Series, sleep: pd.Series,
                             windows: List[int]) -> pd.DataFrame:
    """Step count totals and sleep duration features."""
    feats: Dict[str, np.ndarray] = {}
    for w in windows:
        feats[f"steps_sum_{w}r"] = steps.rolling(w, min_periods=1).sum().values
    # Sleep: hours asleep in last 96 readings (8 h)
    asleep = (sleep > 0).astype(float)
    feats["sleep_hours_8h"] = (asleep.rolling(96, min_periods=1).sum() * 5 / 60).values
    # Deep sleep fraction
    deep = (sleep == 2).astype(float)
    deep_total = deep.rolling(96, min_periods=1).sum()
    total_sleep = asleep.rolling(96, min_periods=1).sum().replace(0, np.nan)
    feats["deep_sleep_frac_8h"] = (deep_total / total_sleep).values
    return pd.DataFrame(feats)


# ──────────────────────────────────────────────────────────────────────────────
# Static features
# ──────────────────────────────────────────────────────────────────────────────

_STATIC_FEATURE_COLS = [
    "age", "bmi", "hba1c", "fasting_glucose",
    "total_cholesterol", "ldl", "hdl", "triglycerides",
    "systolic_bp", "diastolic_bp",
    "diagnosed_t2d", "diagnosed_hypertension", "diagnosed_dyslipidemia",
    "medication_metformin", "medication_sulfonylurea", "medication_insulin",
    "medication_dpp4i", "medication_sglt2i",
    "medication_antihypertensive", "medication_statin",
    "family_history_diabetes", "family_history_cvd",
    "genetic_risk_score",
]


def encode_static_features(ehr_df: pd.DataFrame) -> pd.DataFrame:
    """Encode static EHR features into a numeric matrix.

    Handles real-world EHR variations, missing laboratory or medication channels,
    and column synonyms (e.g., sbp -> systolic_bp).
    """
    df = ehr_df.copy()

    # Column synonyms
    synonym_map = {
        "sbp": "systolic_bp",
        "dbp": "diastolic_bp",
        "hypertension": "diagnosed_hypertension",
        "dyslipidemia": "diagnosed_dyslipidemia",
    }
    for old, new in synonym_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    if "cohort" in df.columns and "diagnosed_t2d" not in df.columns:
        df["diagnosed_t2d"] = (
            df["cohort"].astype(str).str.upper().str.contains("T2D|DIABET")
        ).astype(int)

    # Defaults for all required static feature columns
    defaults = {
        "total_cholesterol": 190.0, "ldl": 110.0, "hdl": 45.0, "triglycerides": 150.0,
        "systolic_bp": 125.0, "diastolic_bp": 80.0,
        "diagnosed_t2d": 0, "diagnosed_hypertension": 0, "diagnosed_dyslipidemia": 0,
        "medication_metformin": 0, "medication_sulfonylurea": 0, "medication_insulin": 0,
        "medication_dpp4i": 0, "medication_sglt2i": 0,
        "medication_antihypertensive": 0, "medication_statin": 0,
        "family_history_diabetes": 0, "family_history_cvd": 0,
        "genetic_risk_score": 0.5,
    }
    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val

    df["sex_male"] = (df["sex"].astype(str).str.upper() == "M").astype(int)

    if "smoking_status" in df.columns:
        df["smoking_current"] = (df["smoking_status"] == "current").astype(int)
        df["smoking_former"] = (df["smoking_status"] == "former").astype(int)
    elif "smoker" in df.columns:
        df["smoking_current"] = df["smoker"].astype(int)
        df["smoking_former"] = 0
    else:
        df["smoking_current"] = 0
        df["smoking_former"] = 0

    activity_map = {"sedentary": 0, "light": 1, "moderate": 2, "active": 3}
    if "physical_activity_level" in df.columns:
        df["activity_encoded"] = df["physical_activity_level"].map(activity_map).fillna(1)
    else:
        df["activity_encoded"] = 1

    cols = _STATIC_FEATURE_COLS + ["sex_male", "smoking_current",
                                    "smoking_former", "activity_encoded"]
    result = df[["patient_id"] + cols].copy()
    result[cols] = result[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Label generation
# ──────────────────────────────────────────────────────────────────────────────

def create_labels(
    cgm_df: pd.DataFrame,
    horizon_readings: int = 24,   # 2 h at 5-min intervals
    threshold: float = 180.0,
) -> pd.Series:
    """Create binary labels: 1 if glucose exceeds ``threshold`` within
    the next ``horizon_readings`` readings.

    Parameters
    ----------
    cgm_df : pd.DataFrame
        Must contain ``glucose_mg_dl`` column, sorted by patient+time.
    horizon_readings : int
        Number of future readings to scan (default 24 = 2 h).
    threshold : float
        Glucose threshold in mg/dL.

    Returns
    -------
    pd.Series
        Binary labels aligned with cgm_df index.
    """
    glucose = cgm_df["glucose_mg_dl"].values
    n = len(glucose)
    labels = np.zeros(n, dtype=int)

    # For each row, check if any of the next `horizon_readings` values > threshold
    # Use rolling max on the reverse-shifted series for efficiency
    glu_series = pd.Series(glucose)
    future_max = glu_series.shift(-1).rolling(window=horizon_readings, min_periods=1).max().shift(-horizon_readings + 1)

    # Alternative: direct forward-looking approach
    for i in range(n - 1):
        end = min(i + 1 + horizon_readings, n)
        future_vals = glucose[i + 1: end]
        valid = future_vals[~np.isnan(future_vals)]
        if len(valid) > 0 and np.max(valid) > threshold:
            labels[i] = 1

    return pd.Series(labels, index=cgm_df.index, name="label")


def create_labels_vectorized(
    cgm_df: pd.DataFrame,
    horizon_readings: int = 24,
    threshold: float = 180.0,
) -> pd.Series:
    """Vectorized label creation – much faster for large datasets.

    For each reading, label=1 if glucose exceeds threshold at any point
    in the next ``horizon_readings`` readings (within the same patient).
    """
    labels = np.zeros(len(cgm_df), dtype=int)

    for pid, grp in cgm_df.groupby("patient_id"):
        idx = grp.index
        glu = grp["glucose_mg_dl"].values

        # Reverse cumulative max (forward-looking max)
        # For each position i, find max of glu[i+1 : i+1+horizon]
        n = len(glu)
        fwd_max = np.full(n, np.nan)
        for h in range(1, horizon_readings + 1):
            shifted = np.full(n, np.nan)
            if h < n:
                shifted[:n - h] = glu[h:]
            fwd_max = np.fmax(fwd_max, shifted)  # nan-aware max

        patient_labels = (fwd_max > threshold).astype(int)
        # Last `horizon_readings` get label 0 (not enough future data)
        patient_labels[-(horizon_readings):] = 0
        labels[idx] = patient_labels

    return pd.Series(labels, index=cgm_df.index, name="label")


# ──────────────────────────────────────────────────────────────────────────────
# Master feature builder
# ──────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    cgm_df: pd.DataFrame,
    ehr_df: pd.DataFrame,
    config: Dict[str, Any] | None = None,
    subsample_negative: float = 0.3,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build the full feature matrix + labels for the baseline model.

    Parameters
    ----------
    cgm_df : pd.DataFrame
        CGM/wearable time-series.
    ehr_df : pd.DataFrame
        EHR profiles.
    config : dict, optional
    subsample_negative : float
        Fraction of negative samples to keep (reduces dataset size for training).

    Returns
    -------
    features : pd.DataFrame
        Feature matrix (one row per prediction point).
    labels : pd.Series
        Binary labels.
    """
    if config is None:
        config = load_config()

    model_cfg = config["model"]
    lookback_readings = int(model_cfg["lookback_hours"] * 12)  # 12 readings/h
    horizon_readings = int(model_cfg["prediction_horizon_hours"] * 12)
    threshold = model_cfg["glucose_threshold"]

    # Windows in readings (12 readings = 1 h)
    gluc_windows = [12, 24, 48, 96]   # 1h, 2h, 4h, 8h
    hr_windows = [12, 48]             # 1h, 4h
    step_windows = [12, 48]           # 1h, 4h
    tir_window = 48                    # 4h

    logger.info("Building feature matrix ...")
    logger.info("  Lookback   : %d readings (%d h)", lookback_readings,
                model_cfg["lookback_hours"])
    logger.info("  Horizon    : %d readings (%d h)", horizon_readings,
                model_cfg["prediction_horizon_hours"])
    logger.info("  Threshold  : %.0f mg/dL", threshold)

    # Encode static features
    static_df = encode_static_features(ehr_df)
    static_cols = [c for c in static_df.columns if c != "patient_id"]

    all_features: List[pd.DataFrame] = []
    all_labels: List[pd.Series] = []
    all_meta: List[pd.DataFrame] = []

    for pid, grp in cgm_df.groupby("patient_id"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        n = len(grp)

        # Skip patients with too little data
        if n < lookback_readings + horizon_readings:
            continue

        # Sensor features
        gf = _rolling_glucose_features(grp["glucose_mg_dl"], gluc_windows)
        tir = _time_in_range_features(grp["glucose_mg_dl"], tir_window)
        hrf = _hr_hrv_features(grp["heart_rate_bpm"], grp["hrv_ms"], hr_windows)
        act = _activity_sleep_features(grp["steps"], grp["sleep_stage"], step_windows)

        # Current glucose and instantaneous features
        inst = pd.DataFrame({
            "glucose_current": grp["glucose_mg_dl"].ffill().values,
            "hour_sin": np.sin(2 * np.pi * grp["hour"].values / 24),
            "hour_cos": np.cos(2 * np.pi * grp["hour"].values / 24),
            "meal_recent_1h": grp["meal_event"].rolling(12, min_periods=1).sum().values,
            "meal_carbs_recent_1h": grp["meal_carbs"].rolling(12, min_periods=1).sum().values,
            "exercise_recent_2h": grp["exercise_event"].rolling(24, min_periods=1).sum().values,
        })

        sensor_feats = pd.concat([gf, tir, hrf, act, inst], axis=1)

        # ── Labels: "will glucose first CROSS 180 in next horizon, given
        # glucose is currently < 180?" ──────────────────────────────────
        # A sample is positive only if:
        #   (a) current glucose < threshold (not already hyper), AND
        #   (b) at least one future reading within horizon exceeds threshold
        glu = grp["glucose_mg_dl"].values
        patient_labels = np.zeros(n, dtype=int)
        for h in range(1, horizon_readings + 1):
            shifted = np.full(n, np.nan)
            if h < n:
                shifted[:n - h] = glu[h:]
            # Positive only if future value > threshold AND current < threshold
            currently_below = glu < threshold
            patient_labels |= (
                (shifted > threshold).astype(int) & currently_below.astype(int)
            )
        patient_labels[-(horizon_readings):] = 0
        labels = pd.Series(patient_labels, name="label")

        # Only keep rows with enough lookback
        valid = np.arange(lookback_readings, n - horizon_readings)
        sensor_feats = sensor_feats.iloc[valid].reset_index(drop=True)
        labels = labels.iloc[valid].reset_index(drop=True)

        # Add static features
        static_row = static_df[static_df["patient_id"] == pid][static_cols]
        if static_row.empty:
            continue
        static_values = static_row.iloc[0].values
        for j, col in enumerate(static_cols):
            sensor_feats[col] = static_values[j]

        meta = pd.DataFrame({
            "patient_id": pid,
            "timestamp": grp["timestamp"].iloc[valid].values,
            "day": grp["day"].iloc[valid].values,
        })

        all_features.append(sensor_feats)
        all_labels.append(labels)
        all_meta.append(meta)

    features = pd.concat(all_features, ignore_index=True)
    labels = pd.concat(all_labels, ignore_index=True)
    meta = pd.concat(all_meta, ignore_index=True)

    # Add metadata columns
    features["patient_id"] = meta["patient_id"].values
    features["timestamp"] = meta["timestamp"].values
    features["day"] = meta["day"].values

    # Subsample negatives to reduce class imbalance
    if subsample_negative < 1.0:
        pos_mask = labels == 1
        neg_mask = labels == 0
        n_pos = pos_mask.sum()
        n_neg_keep = int(neg_mask.sum() * subsample_negative)
        rng = np.random.default_rng(42)
        neg_indices = rng.choice(np.where(neg_mask)[0], size=n_neg_keep, replace=False)
        keep_indices = np.sort(np.concatenate([np.where(pos_mask)[0], neg_indices]))
        features = features.iloc[keep_indices].reset_index(drop=True)
        labels = labels.iloc[keep_indices].reset_index(drop=True)

    logger.info("Feature matrix shape: %s", features.shape)
    logger.info("Label distribution: 0=%d, 1=%d (%.1f%% positive)",
                (labels == 0).sum(), (labels == 1).sum(),
                labels.mean() * 100)

    return features, labels
