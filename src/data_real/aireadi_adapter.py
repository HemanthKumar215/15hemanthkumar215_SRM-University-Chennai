"""
aireadi_adapter.py – Adapter for AI-READI Dataset (NIH Bridge2AI)
===================================================================
Ingests Dexcom G6 CGM traces, Garmin wearable signals, and clinical REDCap EHR.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.data_real.base import BaseRealAdapter, CANONICAL_CGM_COLS, CANONICAL_EHR_COLS

logger = logging.getLogger(__name__)


class AIREADIAdapter(BaseRealAdapter):
    """Adapter for AI-READI (NIH Bridge2AI) multimodal diabetes study."""

    @property
    def dataset_name(self) -> str:
        return "aireadi"

    @property
    def available_modalities(self) -> List[str]:
        return [
            "glucose", "heart_rate", "hrv", "steps",
            "sleep_stage", "meal_carbs", "ehr",
        ]

    def load_cgm(self) -> pd.DataFrame:
        """Load and harmonize AI-READI wearable & CGM data."""
        cgm_file = self.data_dir / "cgm_readings.csv"
        wearable_file = self.data_dir / "garmin_metrics.csv"

        if not cgm_file.exists():
            raise FileNotFoundError(f"AI-READI CGM file not found at: {cgm_file}")

        cgm_raw = pd.read_csv(cgm_file)

        # Standardize CGM columns
        # Expected raw: [study_id, datetime, cgm_value_mg_dl, ...]
        rename_map = {
            "study_id": "patient_id",
            "participant_id": "patient_id",
            "datetime": "timestamp",
            "cgm_value": "glucose_mg_dl",
            "glucose": "glucose_mg_dl",
            "cgm_value_mg_dl": "glucose_mg_dl",
        }
        for k, v in rename_map.items():
            if k in cgm_raw.columns:
                cgm_raw = cgm_raw.rename(columns={k: v})

        cgm_raw["timestamp"] = pd.to_datetime(cgm_raw["timestamp"])

        # Merge wearable signals if available
        if wearable_file.exists():
            wearable_raw = pd.read_csv(wearable_file)
            wearable_raw = wearable_raw.rename(columns={
                "study_id": "patient_id",
                "participant_id": "patient_id",
                "datetime": "timestamp",
                "hr": "heart_rate_bpm",
                "rmssd": "hrv_ms",
                "step_count": "steps",
                "sleep": "sleep_stage",
            })
            wearable_raw["timestamp"] = pd.to_datetime(wearable_raw["timestamp"])
            merged = pd.merge_asof(
                cgm_raw.sort_values("timestamp"),
                wearable_raw.sort_values("timestamp"),
                on="timestamp",
                by="patient_id",
                tolerance=pd.Timedelta("5min"),
                direction="nearest",
            )
        else:
            merged = cgm_raw

        # Fill any missing canonical columns with default NaNs or zeros
        for col in CANONICAL_CGM_COLS:
            if col not in merged.columns:
                if col in ("meal_carbs", "exercise_duration"):
                    merged[col] = 0
                elif col in ("meal_type",):
                    merged[col] = "none"
                elif col in ("exercise_intensity",):
                    merged[col] = 0.0
                else:
                    merged[col] = np.nan

        return merged[CANONICAL_CGM_COLS].copy()

    def load_ehr(self) -> pd.DataFrame:
        """Load and harmonize AI-READI participant EHR profiles."""
        ehr_file = self.data_dir / "participant_ehr.csv"
        if not ehr_file.exists():
            raise FileNotFoundError(f"AI-READI EHR file not found at: {ehr_file}")

        raw = pd.read_csv(ehr_file)
        rename_map = {
            "study_id": "patient_id",
            "participant_id": "patient_id",
            "glycated_hemoglobin": "hba1c",
            "fasting_plasma_glucose": "fasting_glucose",
            "systolic_bp": "sbp",
            "diastolic_bp": "dbp",
            "diabetes_status": "cohort",
        }
        for k, v in rename_map.items():
            if k in raw.columns:
                raw = raw.rename(columns={k: v})

        # Ensure standardized cohort values
        if "cohort" in raw.columns:
            cohort_map = {
                "Control": "Normal",
                "Normoglycemic": "Normal",
                "Pre-diabetes": "Prediabetic",
                "Prediabetes": "Prediabetic",
                "Type 2 Diabetes": "T2D",
                "T2DM": "T2D",
            }
            raw["cohort"] = raw["cohort"].map(lambda x: cohort_map.get(str(x), str(x)))

        for col in CANONICAL_EHR_COLS:
            if col not in raw.columns:
                if col in ("smoker", "hypertension", "dyslipidemia"):
                    raw[col] = 0
                else:
                    raw[col] = np.nan

        return raw[CANONICAL_EHR_COLS].copy()
