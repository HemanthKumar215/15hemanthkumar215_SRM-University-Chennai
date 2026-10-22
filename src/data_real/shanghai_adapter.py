"""
shanghai_adapter.py – Adapter for Shanghai T1DM/T2DM Dataset
=============================================================
Ingests Abbott Freestyle Libre CGM traces (15-min resampled to 5-min grid),
dietary carbohydrate logs, and inpatient clinical profiles.
Modality availability: glucose, meal_carbs, and clinical covariates.
Missing modalities: heart rate, HRV, steps, sleep stages (handled via masking).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.data_real.base import BaseRealAdapter, CANONICAL_CGM_COLS, CANONICAL_EHR_COLS

logger = logging.getLogger(__name__)


class ShanghaiAdapter(BaseRealAdapter):
    """Adapter for Shanghai T1DM/T2DM Clinical Cohort (Nature Scientific Data)."""

    @property
    def dataset_name(self) -> str:
        return "shanghai"

    @property
    def available_modalities(self) -> List[str]:
        return ["glucose", "meal_carbs", "ehr"]

    def load_cgm(self) -> pd.DataFrame:
        """Load and harmonize Shanghai CGM traces and meal annotations."""
        cgm_file = self.data_dir / "shanghai_cgm.csv"
        meals_file = self.data_dir / "shanghai_meals.csv"

        if not cgm_file.exists():
            raise FileNotFoundError(f"Shanghai CGM file not found at: {cgm_file}")

        raw = pd.read_csv(cgm_file)
        rename_map = {
            "patient_id": "patient_id",
            "subject_id": "patient_id",
            "datetime": "timestamp",
            "date_time": "timestamp",
            "glucose": "glucose_mg_dl",
            "glucose_value": "glucose_mg_dl",
        }
        for k, v in rename_map.items():
            if k in raw.columns:
                raw = raw.rename(columns={k: v})

        raw["timestamp"] = pd.to_datetime(raw["timestamp"])

        # Shanghai CGM data is natively recorded at 15-minute intervals.
        # Resample each patient's series to the unified 5-minute grid using linear interpolation.
        resampled_patients: List[pd.DataFrame] = []
        for pid, grp in raw.groupby("patient_id"):
            grp = grp.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
            grp = grp.set_index("timestamp")
            # Resample to 5-minute frequency
            resampled = grp.resample("5min").asfreq()
            # Interpolate glucose linearly up to 3 missing steps (15 mins)
            resampled["glucose_mg_dl"] = resampled["glucose_mg_dl"].interpolate(
                method="linear", limit=3
            )
            resampled["patient_id"] = pid
            resampled_patients.append(resampled.reset_index())

        if resampled_patients:
            cgm_aligned = pd.concat(resampled_patients, ignore_index=True)
        else:
            cgm_aligned = raw

        # Merge meal logs if present
        if meals_file.exists():
            meals_raw = pd.read_csv(meals_file)
            meals_raw = meals_raw.rename(columns={
                "subject_id": "patient_id",
                "datetime": "timestamp",
                "carbs_g": "meal_carbs",
                "carbohydrate": "meal_carbs",
            })
            meals_raw["timestamp"] = pd.to_datetime(meals_raw["timestamp"])
            cgm_aligned = pd.merge_asof(
                cgm_aligned.sort_values("timestamp"),
                meals_raw.sort_values("timestamp"),
                on="timestamp",
                by="patient_id",
                tolerance=pd.Timedelta("10min"),
                direction="nearest",
            )
        else:
            cgm_aligned["meal_carbs"] = 0

        # Fill remaining canonical columns
        for col in CANONICAL_CGM_COLS:
            if col not in cgm_aligned.columns:
                if col in ("meal_carbs", "exercise_duration"):
                    cgm_aligned[col] = 0
                elif col in ("meal_type",):
                    cgm_aligned[col] = "none"
                elif col in ("exercise_intensity",):
                    cgm_aligned[col] = 0.0
                else:
                    cgm_aligned[col] = np.nan

        return cgm_aligned[CANONICAL_CGM_COLS].copy()

    def load_ehr(self) -> pd.DataFrame:
        """Load and harmonize Shanghai clinical profiles."""
        ehr_file = self.data_dir / "shanghai_patients.csv"
        if not ehr_file.exists():
            raise FileNotFoundError(f"Shanghai patients file not found at: {ehr_file}")

        raw = pd.read_csv(ehr_file)
        rename_map = {
            "subject_id": "patient_id",
            "gender": "sex",
            "diabetes_type": "cohort",
        }
        for k, v in rename_map.items():
            if k in raw.columns:
                raw = raw.rename(columns={k: v})

        if "cohort" in raw.columns:
            cohort_map = {
                "Type 2 Diabetes": "T2D",
                "T2D": "T2D",
                "Type 1 Diabetes": "T1D",
                "T1D": "T1D",
            }
            raw["cohort"] = raw["cohort"].map(lambda x: cohort_map.get(str(x), str(x)))

        for col in CANONICAL_EHR_COLS:
            if col not in raw.columns:
                if col in ("smoker", "hypertension", "dyslipidemia"):
                    raw[col] = 0
                elif col == "fasting_glucose" and "hba1c" in raw.columns:
                    raw[col] = (28.7 * raw["hba1c"] - 46.7).round(1)
                else:
                    raw[col] = np.nan

        return raw[CANONICAL_EHR_COLS].copy()
