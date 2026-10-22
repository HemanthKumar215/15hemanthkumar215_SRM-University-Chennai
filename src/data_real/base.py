"""
base.py – Abstract Base Adapter for Real Diabetes Datasets
===========================================================
Defines the standardized contract for converting external datasets
(AI-READI, CGMacros, Shanghai T1DM/T2DM) to the canonical schema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Canonical columns expected for dynamic CGM / wearable time-series
CANONICAL_CGM_COLS = [
    "patient_id",
    "timestamp",
    "glucose_mg_dl",
    "heart_rate_bpm",
    "hrv_ms",
    "steps",
    "sleep_stage",
    "meal_carbs",
    "meal_type",
    "exercise_intensity",
    "exercise_duration",
]

# Canonical columns expected for static EHR profiles
CANONICAL_EHR_COLS = [
    "patient_id",
    "age",
    "sex",
    "bmi",
    "hba1c",
    "fasting_glucose",
    "sbp",
    "dbp",
    "smoker",
    "hypertension",
    "dyslipidemia",
    "cohort",
]


class BaseRealAdapter(ABC):
    """Abstract base class for all real-world data loaders."""

    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Name of the dataset (e.g., 'aireadi', 'cgmacros', 'shanghai')."""
        ...

    @property
    @abstractmethod
    def available_modalities(self) -> List[str]:
        """List of modalities natively present in this dataset.

        Possible values: 'glucose', 'heart_rate', 'hrv', 'steps',
                         'sleep_stage', 'meal_carbs', 'ehr'
        """
        ...

    @abstractmethod
    def load_cgm(self) -> pd.DataFrame:
        """Load and harmonize the dynamic CGM/wearable stream into canonical schema.

        Must return a DataFrame containing at least:
        ['patient_id', 'timestamp', 'glucose_mg_dl', ...]
        with missing sensor channels filled with NaN.
        """
        ...

    @abstractmethod
    def load_ehr(self) -> pd.DataFrame:
        """Load and harmonize the static EHR profiles into canonical schema.

        Must return a DataFrame containing at least:
        ['patient_id', 'age', 'sex', 'bmi', 'hba1c', 'fasting_glucose', 'cohort']
        """
        ...

    def get_modality_mask(self) -> Dict[str, bool]:
        """Return a dictionary indicating whether each modality is present."""
        all_mods = [
            "glucose", "heart_rate", "hrv", "steps",
            "sleep_stage", "meal_carbs", "ehr",
        ]
        present = set(self.available_modalities)
        return {mod: (mod in present) for mod in all_mods}

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both CGM time-series and EHR profiles and validate columns."""
        cgm_df = self.load_cgm()
        ehr_df = self.load_ehr()

        # Enforce canonical columns
        for col in ["patient_id", "timestamp", "glucose_mg_dl"]:
            if col not in cgm_df.columns:
                raise ValueError(f"{self.dataset_name} CGM missing required column: {col}")

        for col in ["patient_id", "age", "sex", "bmi", "hba1c", "cohort"]:
            if col not in ehr_df.columns:
                raise ValueError(f"{self.dataset_name} EHR missing required column: {col}")

        # Ensure timestamp is datetime
        cgm_df["timestamp"] = pd.to_datetime(cgm_df["timestamp"])
        cgm_df = cgm_df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

        return cgm_df, ehr_df
