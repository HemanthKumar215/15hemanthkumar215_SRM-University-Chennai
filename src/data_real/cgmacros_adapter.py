"""
cgmacros_adapter.py – Adapter for CGMacros Benchmark Dataset
=============================================================
Ingests standardized benchmark CGM traces across diverse diabetic cohorts.
Modality availability: glucose time-series and static clinical covariates.
Missing modalities: PPG heart rate, HRV, steps, sleep staging (handled via masking).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.data_real.base import BaseRealAdapter, CANONICAL_CGM_COLS, CANONICAL_EHR_COLS

logger = logging.getLogger(__name__)


class CGMacrosAdapter(BaseRealAdapter):
    """Adapter for CGMacros benchmark repository (PhysioNet / open-source)."""

    @property
    def dataset_name(self) -> str:
        return "cgmacros"

    @property
    def available_modalities(self) -> List[str]:
        return ["glucose", "ehr"]

    def load_cgm(self) -> pd.DataFrame:
        """Load and harmonize CGMacros time-series readings."""
        cgm_file = self.data_dir / "cgmacros_cgm.csv"
        if not cgm_file.exists():
            raise FileNotFoundError(f"CGMacros CGM file not found at: {cgm_file}")

        raw = pd.read_csv(cgm_file)
        rename_map = {
            "subject_id": "patient_id",
            "id": "patient_id",
            "time": "timestamp",
            "datetime": "timestamp",
            "glucose": "glucose_mg_dl",
            "cgm": "glucose_mg_dl",
            "sensor_glucose": "glucose_mg_dl",
        }
        for k, v in rename_map.items():
            if k in raw.columns:
                raw = raw.rename(columns={k: v})

        raw["timestamp"] = pd.to_datetime(raw["timestamp"])

        # Populate unmeasured channels with NaNs / defaults
        for col in CANONICAL_CGM_COLS:
            if col not in raw.columns:
                if col in ("meal_carbs", "exercise_duration"):
                    raw[col] = 0
                elif col in ("meal_type",):
                    raw[col] = "none"
                elif col in ("exercise_intensity",):
                    raw[col] = 0.0
                else:
                    raw[col] = np.nan

        return raw[CANONICAL_CGM_COLS].copy()

    def load_ehr(self) -> pd.DataFrame:
        """Load and harmonize CGMacros subject demographics and clinical profiles."""
        ehr_file = self.data_dir / "cgmacros_participants.csv"
        if not ehr_file.exists():
            raise FileNotFoundError(f"CGMacros participants file not found at: {ehr_file}")

        raw = pd.read_csv(ehr_file)
        rename_map = {
            "subject_id": "patient_id",
            "id": "patient_id",
            "gender": "sex",
            "diabetes_type": "cohort",
            "diagnosis": "cohort",
        }
        for k, v in rename_map.items():
            if k in raw.columns:
                raw = raw.rename(columns={k: v})

        if "cohort" in raw.columns:
            cohort_map = {
                "T2D": "T2D",
                "Type 2": "T2D",
                "Control": "Normal",
                "Non-diabetic": "Normal",
                "Prediabetes": "Prediabetic",
            }
            raw["cohort"] = raw["cohort"].map(lambda x: cohort_map.get(str(x), str(x)))

        for col in CANONICAL_EHR_COLS:
            if col not in raw.columns:
                if col in ("smoker", "hypertension", "dyslipidemia"):
                    raw[col] = 0
                elif col == "fasting_glucose" and "hba1c" in raw.columns:
                    # Estimate fasting glucose from HbA1c if not measured
                    raw[col] = (28.7 * raw["hba1c"] - 46.7).round(1)
                else:
                    raw[col] = np.nan

        return raw[CANONICAL_EHR_COLS].copy()
