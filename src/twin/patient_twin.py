"""
patient_twin.py – Digital Twin engine
=======================================
The ``PatientTwin`` class holds a patient's clinical profile and
time-series state, ingests new sensor readings, runs the fusion model
for rolling predictions, and supports **what-if** simulations.

What-if examples
----------------
* *"What if the patient walks 30 min after dinner?"* → inject an exercise
  event, re-predict, and compare the spike probability.
* *"What if the patient skips the late-night snack?"* → remove a meal event,
  re-predict.

All data is synthetic; this class is for educational / hackathon purposes.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.utils.config import load_config
from src.models.fusion_model import build_model_from_config

logger = logging.getLogger(__name__)

_SENSOR_COLS = ["glucose_mg_dl", "heart_rate_bpm", "hrv_ms", "steps", "sleep_stage"]

_SENSOR_LABELS = {
    "glucose_mg_dl": "CGM glucose trace (24 h)",
    "heart_rate_bpm": "Heart rate (24 h)",
    "hrv_ms": "Heart rate variability (24 h)",
    "steps": "Step count (24 h)",
    "sleep_stage": "Sleep stages (24 h)",
}

_STATIC_LABELS = {
    "hba1c": "HbA1c",
    "bmi": "BMI",
    "ldl": "LDL cholesterol",
    "hdl": "HDL cholesterol",
    "systolic_bp": "Systolic BP",
    "diastolic_bp": "Diastolic BP",
    "diagnosed_t2d": "T2D diagnosis",
    "medication_dpp4i": "DPP-4 inhibitor",
    "medication_sglt2i": "SGLT2 inhibitor",
    "sex_male": "Male sex",
    "smoking_current": "Current smoker",
    "smoking_former": "Former smoker",
    "activity_encoded": "Physical activity level",
}


def _static_label(name: str) -> str:
    """Human-readable label for an encoded static feature column."""
    if name in _STATIC_LABELS:
        return _STATIC_LABELS[name]
    for prefix in ("medication_", "diagnosed_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace("_", " ").capitalize()


@dataclass
class PredictionResult:
    """Container for a single prediction output."""
    timestamp: pd.Timestamp
    probability: float
    risk_level: str          # "low", "medium", "high"
    glucose_current: float
    top_factors: List[Dict[str, Any]] = field(default_factory=list)


class PatientTwin:
    """Digital twin for a single patient.

    Holds the patient's static profile, maintains a sliding window of
    sensor data, and runs the fusion model to produce spike predictions.
    """

    def __init__(
        self,
        patient_id: str,
        static_profile: pd.Series,
        model: nn.Module,
        norm_stats: Dict[str, Any],
        config: Dict[str, Any] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        patient_id : str
        static_profile : pd.Series
            One row from the EHR profiles DataFrame.
        model : nn.Module
            Trained fusion model (or any model with the same interface).
        norm_stats : dict
            Normalisation stats from training (ts_mean, ts_std, static_mean, static_std).
        config : dict, optional
        """
        if config is None:
            config = load_config()

        self.patient_id = patient_id
        self.static_profile = static_profile
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

        model_cfg = config["model"]
        self.lookback = int(model_cfg["lookback_hours"] * 12)
        self.threshold = model_cfg["glucose_threshold"]

        # Risk thresholds
        risk_cfg = config.get("risk_thresholds", {})
        self.risk_low = risk_cfg.get("low", 0.25)
        self.risk_high = risk_cfg.get("medium", 0.55)

        # Normalisation stats
        self.ts_mean = np.array(norm_stats["ts_mean"], dtype=np.float32)
        self.ts_std = np.array(norm_stats["ts_std"], dtype=np.float32)
        self.static_mean = np.array(norm_stats["static_mean"], dtype=np.float32)
        self.static_std = np.array(norm_stats["static_std"], dtype=np.float32)
        # Guard against zero-variance features in older checkpoints
        self.ts_std = np.clip(self.ts_std, 1e-6, None)
        self.static_std = np.clip(self.static_std, 1e-6, None)

        # Encode static features
        self.static_vec = self._encode_static(static_profile)

        # Sensor buffer (sliding window)
        self.sensor_buffer: List[Dict[str, float]] = []
        self.prediction_history: List[PredictionResult] = []

    def _encode_static(self, row: pd.Series) -> np.ndarray:
        """Encode a patient's static profile into a numeric vector."""
        from src.features.feature_engineering import encode_static_features

        df = pd.DataFrame([row])
        encoded = encode_static_features(df)
        cols = [c for c in encoded.columns if c != "patient_id"]
        self.static_names = cols
        vec = encoded[cols].iloc[0].values.astype(np.float32)
        return vec

    def ingest_reading(self, reading: Dict[str, float]) -> None:
        """Add a new sensor reading to the buffer.

        Parameters
        ----------
        reading : dict
            Keys: glucose_mg_dl, heart_rate_bpm, hrv_ms, steps, sleep_stage.
            Optionally: timestamp, meal_event, meal_carbs, exercise_event.
        """
        self.sensor_buffer.append(reading)
        # Keep only `lookback` most recent readings
        if len(self.sensor_buffer) > self.lookback * 2:
            self.sensor_buffer = self.sensor_buffer[-self.lookback * 2:]

    def ingest_batch(self, df: pd.DataFrame) -> None:
        """Ingest a batch of readings from a DataFrame."""
        for _, row in df.iterrows():
            self.ingest_reading(row.to_dict())

    def predict(self) -> Optional[PredictionResult]:
        """Run the fusion model on the current sensor buffer.

        Returns
        -------
        PredictionResult or None
            None if insufficient data in the buffer.
        """
        result = self._predict_buffer(self.sensor_buffer)
        if result is not None:
            self.prediction_history.append(result)
        return result

    def _normalised_inputs(
        self, window_data: List[Dict[str, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the normalised (lookback, channels) sensor window and static vector."""
        ts_df = pd.DataFrame(
            [[r.get(c, np.nan) for c in _SENSOR_COLS] for r in window_data],
            columns=_SENSOR_COLS,
        ).apply(pd.to_numeric, errors="coerce")

        # Handle NaN exactly as the training datasets do (forward-fill, then 0)
        # so sparse uploads (e.g. CGM-only exports) match the training distribution.
        ts_array = ts_df.ffill().fillna(0).values.astype(np.float32)

        ts_norm = (ts_array - self.ts_mean) / self.ts_std
        static_norm = (self.static_vec - self.static_mean) / self.static_std
        return ts_norm.astype(np.float32), static_norm.astype(np.float32)

    def _probabilities(self, ts_norm: np.ndarray, static_norm: np.ndarray) -> np.ndarray:
        """Spike probabilities for a batch of normalised inputs."""
        ts_tensor = torch.tensor(ts_norm, dtype=torch.float32).to(self.device)
        static_tensor = torch.tensor(static_norm, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(ts_tensor, static_tensor).reshape(-1)
        return torch.sigmoid(logits).cpu().numpy()

    def explain(self, top_k_static: int = 6) -> List[Dict[str, Any]]:
        """Occlusion attribution for the current prediction.

        Each sensor channel (over the whole 24 h window) and each static EHR
        feature is replaced in turn by its training-set mean (normalised value
        0), and the model is re-run. ``impact`` is the drop in spike
        probability, in percentage points, caused by that replacement:
        positive means the patient's actual value *raises* risk relative to an
        average patient, negative means it is protective.

        Returns
        -------
        list of dict
            Keys ``feature``, ``group`` ("sensor" or "ehr") and ``impact``,
            sorted by absolute impact. All sensor channels are returned plus the
            ``top_k_static`` most influential EHR features. Empty if there is
            not enough data for a prediction.
        """
        if len(self.sensor_buffer) < self.lookback:
            return []

        ts_norm, static_norm = self._normalised_inputs(self.sensor_buffer[-self.lookback:])
        n_ts, n_static = ts_norm.shape[1], static_norm.shape[0]

        # Row 0 is the unmodified input; then one row per occluded input.
        ts_batch = np.repeat(ts_norm[None], 1 + n_ts + n_static, axis=0)
        static_batch = np.repeat(static_norm[None], 1 + n_ts + n_static, axis=0)
        for j in range(n_ts):
            ts_batch[1 + j, :, j] = 0.0
        for j in range(n_static):
            static_batch[1 + n_ts + j, j] = 0.0

        probs = self._probabilities(ts_batch, static_batch)
        deltas = (probs[0] - probs[1:]) * 100.0

        sensor = [
            {"feature": _SENSOR_LABELS.get(c, c), "group": "sensor", "impact": round(float(d), 2)}
            for c, d in zip(_SENSOR_COLS, deltas[:n_ts])
        ]
        ehr = [
            {"feature": _static_label(name), "group": "ehr", "impact": round(float(d), 2)}
            for name, d in zip(self.static_names, deltas[n_ts:])
        ]
        ehr = sorted(ehr, key=lambda f: abs(f["impact"]), reverse=True)[:top_k_static]
        return sorted(sensor + ehr, key=lambda f: abs(f["impact"]), reverse=True)

    def _predict_buffer(self, buffer: List[Dict[str, float]]) -> Optional[PredictionResult]:
        """Run the model on ``buffer`` without recording prediction history."""
        if len(buffer) < self.lookback:
            logger.debug("Not enough data for prediction (%d < %d)",
                         len(buffer), self.lookback)
            return None

        window_data = buffer[-self.lookback:]
        ts_norm, static_norm = self._normalised_inputs(window_data)
        probability = float(self._probabilities(ts_norm[None], static_norm[None])[0])

        # Risk level
        if probability < self.risk_low:
            risk_level = "low"
        elif probability < self.risk_high:
            risk_level = "medium"
        else:
            risk_level = "high"

        # Current glucose
        current_glucose = window_data[-1].get("glucose_mg_dl", 0.0)
        timestamp = window_data[-1].get("timestamp", pd.Timestamp.now())
        if isinstance(timestamp, str):
            timestamp = pd.Timestamp(timestamp)

        return PredictionResult(
            timestamp=timestamp,
            probability=round(float(probability), 4),
            risk_level=risk_level,
            glucose_current=current_glucose,
        )

    def what_if(self, scenario: str, **kwargs: Any) -> Dict[str, Any]:
        """Run a what-if simulation by modifying the sensor buffer.

        Supported scenarios
        -------------------
        ``"add_exercise"``
            Kwargs: ``duration_min`` (int), ``intensity`` (float 0-1).
            Simulates a walk/exercise after the last reading.

        ``"skip_meal"``
            Removes the last meal from the buffer.

        ``"add_meal"``
            Kwargs: ``carbs`` (int).
            Adds a meal event.

        ``"improve_sleep"``
            Improves sleep quality in the buffer.

        Returns
        -------
        dict
            Keys: ``original_prob``, ``modified_prob``, ``original_risk``,
            ``modified_risk``, ``delta_prob``, ``scenario``.
        """
        # Original prediction (simulations never touch prediction_history)
        original = self._predict_buffer(self.sensor_buffer)
        if original is None:
            return {"error": "Not enough data for prediction"}

        # Make a modified copy of the buffer
        modified_buffer = copy.deepcopy(self.sensor_buffer)

        if scenario == "add_exercise":
            duration = kwargs.get("duration_min", 30)
            intensity = kwargs.get("intensity", 0.5)
            readings_affected = min(duration // 5, len(modified_buffer))
            for i in range(-readings_affected, 0):
                # Exercise lowers glucose and raises HR
                glucose_drop = intensity * 25 / max(1, readings_affected)
                modified_buffer[i]["glucose_mg_dl"] = max(
                    70, modified_buffer[i].get("glucose_mg_dl", 120) - glucose_drop
                )
                modified_buffer[i]["heart_rate_bpm"] = min(
                    170, modified_buffer[i].get("heart_rate_bpm", 75) + 30 * intensity
                )
                modified_buffer[i]["steps"] = modified_buffer[i].get("steps", 0) + int(80 * intensity)

        elif scenario == "skip_meal":
            # Find and remove last meal effect (reduce glucose by ~30-50 mg/dL)
            for i in range(-24, 0):  # last 2 hours
                if abs(i) <= len(modified_buffer):
                    idx = len(modified_buffer) + i
                    modified_buffer[idx]["glucose_mg_dl"] = max(
                        80, modified_buffer[idx].get("glucose_mg_dl", 120) - 20
                    )
                    modified_buffer[idx]["meal_event"] = 0
                    modified_buffer[idx]["meal_carbs"] = 0

        elif scenario == "add_meal":
            carbs = kwargs.get("carbs", 50)
            # Simulate glucose rise from a meal
            for i in range(-12, 0):  # last hour
                if abs(i) <= len(modified_buffer):
                    idx = len(modified_buffer) + i
                    rise = carbs * 0.5 * max(0, 1 - abs(i + 6) / 6)
                    modified_buffer[idx]["glucose_mg_dl"] = (
                        modified_buffer[idx].get("glucose_mg_dl", 120) + rise
                    )

        elif scenario == "improve_sleep":
            # Improve sleep quality: lower glucose slightly, improve HRV
            for i in range(len(modified_buffer)):
                if modified_buffer[i].get("sleep_stage", 0) > 0:
                    modified_buffer[i]["glucose_mg_dl"] = max(
                        70, modified_buffer[i].get("glucose_mg_dl", 120) - 5
                    )
                    modified_buffer[i]["hrv_ms"] = (
                        modified_buffer[i].get("hrv_ms", 50) + 10
                    )

        else:
            return {"error": f"Unknown scenario: {scenario}"}

        modified_result = self._predict_buffer(modified_buffer)

        return {
            "scenario": scenario,
            "original_prob": original.probability,
            "modified_prob": modified_result.probability if modified_result else None,
            "original_risk": original.risk_level,
            "modified_risk": modified_result.risk_level if modified_result else None,
            "delta_prob": round(
                (modified_result.probability - original.probability), 4
            ) if modified_result else None,
            "kwargs": kwargs,
        }

    def get_recent_glucose(self, n: int = 288) -> List[float]:
        """Return the last ``n`` glucose readings."""
        readings = self.sensor_buffer[-n:]
        return [r.get("glucose_mg_dl", np.nan) for r in readings]

    def get_prediction_history(self) -> pd.DataFrame:
        """Return prediction history as a DataFrame."""
        if not self.prediction_history:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "timestamp": p.timestamp,
                "probability": p.probability,
                "risk_level": p.risk_level,
                "glucose_current": p.glucose_current,
            }
            for p in self.prediction_history
        ])


def load_twin(
    patient_id: str,
    ehr_df: pd.DataFrame,
    model_path: str,
    config: Dict[str, Any] | None = None,
) -> PatientTwin:
    """Convenience function to load a PatientTwin from saved model.

    Parameters
    ----------
    patient_id : str
        Patient ID (e.g., "P0042").
    ehr_df : pd.DataFrame
        EHR profiles.
    model_path : str
        Path to saved ``.pt`` model file.
    config : dict, optional
    """
    if config is None:
        config = load_config()

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model_type = checkpoint.get("model_type", "fusion")
    model_config = checkpoint["config"]
    norm_stats = checkpoint["norm_stats"]

    # Override static_input_dim from norm_stats
    model_config["static_input_dim"] = norm_stats.get(
        "static_dim", model_config.get("static_input_dim", 22)
    )

    model = build_model_from_config(model_config, model_type)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    patient_row = ehr_df[ehr_df["patient_id"] == patient_id].iloc[0]

    return PatientTwin(
        patient_id=patient_id,
        static_profile=patient_row,
        model=model,
        norm_stats=norm_stats,
        config=config,
    )
