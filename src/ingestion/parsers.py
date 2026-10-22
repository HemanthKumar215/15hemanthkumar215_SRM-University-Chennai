"""
parsers.py – Vendor CGM & Wearable Export Parsers
==================================================
Provides parsers for FreeStyle Libre, Dexcom, and Apple Health exports
behind a common interface with automatic format detection.
"""

from __future__ import annotations

import io
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseCGMParser(ABC):
    """Abstract base class for vendor export parsers."""

    @property
    @abstractmethod
    def format_name(self) -> str:
        ...

    @abstractmethod
    def can_parse(self, df: pd.DataFrame) -> bool:
        """Inspect column headers / metadata to determine if this parser applies."""
        ...

    @abstractmethod
    def parse(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert vendor DataFrame into canonical schema:
        ['timestamp', 'glucose_mg_dl', 'heart_rate_bpm', 'hrv_ms', 'steps', 'sleep_stage', 'meal_carbs']
        """
        ...


class CanonicalParser(BaseCGMParser):
    """Parser for native Digital Twin format."""

    @property
    def format_name(self) -> str:
        return "Digital Twin Canonical"

    def can_parse(self, df: pd.DataFrame) -> bool:
        cols_lower = [c.lower().strip() for c in df.columns]
        has_time = any(c in cols_lower for c in ["timestamp", "datetime", "time"])
        has_glu = any(c in cols_lower for c in ["glucose_mg_dl", "glucose", "cgm"])
        return has_time and has_glu

    def parse(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        # Find timestamp
        for c in df.columns:
            if c.lower().strip() in ["timestamp", "datetime", "time"]:
                out["timestamp"] = pd.to_datetime(df[c], errors="coerce")
                break

        # Find glucose
        for c in df.columns:
            if c.lower().strip() in ["glucose_mg_dl", "glucose", "cgm", "sensor_glucose"]:
                out["glucose_mg_dl"] = pd.to_numeric(df[c], errors="coerce")
                break

        # Optional channels
        channel_map = {
            "heart_rate_bpm": ["heart_rate_bpm", "hr", "heart_rate", "pulse"],
            "hrv_ms": ["hrv_ms", "hrv", "rmssd"],
            "steps": ["steps", "step_count"],
            "sleep_stage": ["sleep_stage", "sleep"],
            "meal_carbs": ["meal_carbs", "carbs", "carbohydrates", "carbs_g"],
        }
        for target, aliases in channel_map.items():
            found = False
            for c in df.columns:
                if c.lower().strip() in aliases:
                    out[target] = pd.to_numeric(df[c], errors="coerce")
                    found = True
                    break
            if not found:
                out[target] = np.nan if target not in ("meal_carbs", "steps") else 0

        return out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


class FreeStyleLibreParser(BaseCGMParser):
    """Parser for Abbott FreeStyle Libre desktop and LibreLink CSV exports."""

    @property
    def format_name(self) -> str:
        return "Abbott FreeStyle Libre"

    def can_parse(self, df: pd.DataFrame) -> bool:
        cols_lower = [str(c).lower().strip() for c in df.columns]
        has_libre_time = any("meter timestamp" in c or "device timestamp" in c for c in cols_lower)
        has_libre_glu = any("historic glucose" in c or "scan glucose" in c for c in cols_lower)
        return has_libre_time or has_libre_glu

    def parse(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        cols_lower = {str(c).lower().strip(): c for c in df.columns}

        # Find timestamp
        time_col = None
        for k in ["device timestamp", "meter timestamp", "timestamp"]:
            if k in cols_lower:
                time_col = cols_lower[k]
                break

        if time_col is None:
            raise ValueError("FreeStyle Libre export missing timestamp column ('Device Timestamp').")
        out["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")

        # Find glucose (Historic Glucose preferred over Scan Glucose)
        glu_col = None
        for k, orig in cols_lower.items():
            if "historic glucose" in k:
                glu_col = orig
                break
        if glu_col is None:
            for k, orig in cols_lower.items():
                if "scan glucose" in k or "glucose" in k:
                    glu_col = orig
                    break

        if glu_col is None:
            raise ValueError("FreeStyle Libre export missing glucose column ('Historic Glucose').")
        out["glucose_mg_dl"] = pd.to_numeric(df[glu_col], errors="coerce")

        # FreeStyle Libre does not measure continuous HR, HRV, steps, sleep
        out["heart_rate_bpm"] = np.nan
        out["hrv_ms"] = np.nan
        out["steps"] = 0
        out["sleep_stage"] = np.nan

        # Carbohydrates if logged
        carbs_col = None
        for k, orig in cols_lower.items():
            if "carbohydrates" in k or "carbs" in k:
                carbs_col = orig
                break
        out["meal_carbs"] = pd.to_numeric(df[carbs_col], errors="coerce").fillna(0) if carbs_col else 0

        return out.dropna(subset=["timestamp", "glucose_mg_dl"]).sort_values("timestamp").reset_index(drop=True)


class DexcomParser(BaseCGMParser):
    """Parser for Dexcom Clarity / Studio CSV exports."""

    @property
    def format_name(self) -> str:
        return "Dexcom Clarity"

    def can_parse(self, df: pd.DataFrame) -> bool:
        cols_lower = [str(c).lower().strip() for c in df.columns]
        has_dex_time = any("timestamp (yyyy-mm-dd" in c or "event datetime" in c or "transmitter time" in c for c in cols_lower)
        has_dex_glu = any("glucose value" in c or "egv" in c for c in cols_lower)
        return has_dex_time or has_dex_glu

    def parse(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        cols_lower = {str(c).lower().strip(): c for c in df.columns}

        time_col = None
        for k, orig in cols_lower.items():
            if "timestamp" in k or "event datetime" in k:
                time_col = orig
                break

        if time_col is None:
            raise ValueError("Dexcom export missing timestamp column ('Timestamp (YYYY-MM-DDThh:mm:ss)').")
        out["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")

        glu_col = None
        for k, orig in cols_lower.items():
            if "glucose value" in k or "egv" in k:
                glu_col = orig
                break

        if glu_col is None:
            raise ValueError("Dexcom export missing glucose column ('Glucose Value (mg/dL)').")
        out["glucose_mg_dl"] = pd.to_numeric(df[glu_col], errors="coerce")

        out["heart_rate_bpm"] = np.nan
        out["hrv_ms"] = np.nan
        out["steps"] = 0
        out["sleep_stage"] = np.nan

        # Carbs if logged (Dexcom event carbs)
        carbs_col = None
        for k, orig in cols_lower.items():
            if "carb" in k:
                carbs_col = orig
                break
        out["meal_carbs"] = pd.to_numeric(df[carbs_col], errors="coerce").fillna(0) if carbs_col else 0

        return out.dropna(subset=["timestamp", "glucose_mg_dl"]).sort_values("timestamp").reset_index(drop=True)


class AppleHealthParser(BaseCGMParser):
    """Parser for Apple Health CSV exports (e.g. Health Export app / Auto Export)."""

    @property
    def format_name(self) -> str:
        return "Apple Health"

    def can_parse(self, df: pd.DataFrame) -> bool:
        cols_lower = [str(c).lower().strip() for c in df.columns]
        # Either tall format (type, startDate, value) or wide format with HK identifiers
        has_hk_type = any("hkquantity" in str(val).lower() for val in df.get("type", pd.Series()).dropna()[:20]) if "type" in cols_lower else False
        has_apple_cols = "startdate" in cols_lower and "value" in cols_lower
        has_blood_glu_header = any("bloodglucose" in c or "blood_glucose" in c for c in cols_lower)
        return has_hk_type or (has_apple_cols and "type" in cols_lower) or has_blood_glu_header

    def parse(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_lower = {str(c).lower().strip(): c for c in df.columns}

        # Check if tall format (type, startDate, value)
        if "type" in cols_lower and "value" in cols_lower:
            time_col = cols_lower.get("startdate", cols_lower.get("date", "startDate"))
            type_col = cols_lower["type"]
            val_col = cols_lower["value"]

            # Filter for Blood Glucose
            glu_mask = df[type_col].astype(str).str.contains("BloodGlucose|blood_glucose", case=False)
            glu_df = df[glu_mask].copy()
            if glu_df.empty:
                raise ValueError("Apple Health export contains no BloodGlucose records.")

            out = pd.DataFrame()
            out["timestamp"] = pd.to_datetime(glu_df[time_col], errors="coerce")
            out["glucose_mg_dl"] = pd.to_numeric(glu_df[val_col], errors="coerce")

            # Check if HeartRate records exist in the same export
            hr_mask = df[type_col].astype(str).str.contains("HeartRate|heart_rate", case=False)
            if hr_mask.any():
                hr_df = df[hr_mask].copy()
                hr_df["timestamp"] = pd.to_datetime(hr_df[time_col], errors="coerce")
                hr_df["heart_rate_bpm"] = pd.to_numeric(hr_df[val_col], errors="coerce")
                out = pd.merge_asof(
                    out.sort_values("timestamp"),
                    hr_df[["timestamp", "heart_rate_bpm"]].sort_values("timestamp"),
                    on="timestamp",
                    direction="nearest",
                    tolerance=pd.Timedelta("5min"),
                )
            else:
                out["heart_rate_bpm"] = np.nan

            out["hrv_ms"] = np.nan
            out["steps"] = 0
            out["sleep_stage"] = np.nan
            out["meal_carbs"] = 0

            return out.dropna(subset=["timestamp", "glucose_mg_dl"]).sort_values("timestamp").reset_index(drop=True)

        # Otherwise wide format
        out = pd.DataFrame()
        time_col = cols_lower.get("startdate", cols_lower.get("timestamp", "date"))
        out["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")

        for k, orig in cols_lower.items():
            if "glucose" in k:
                out["glucose_mg_dl"] = pd.to_numeric(df[orig], errors="coerce")
                break

        for k, orig in cols_lower.items():
            if "heartrate" in k or "heart_rate" in k:
                out["heart_rate_bpm"] = pd.to_numeric(df[orig], errors="coerce")
                break
        if "heart_rate_bpm" not in out.columns:
            out["heart_rate_bpm"] = np.nan

        out["hrv_ms"] = np.nan
        out["steps"] = 0
        out["sleep_stage"] = np.nan
        out["meal_carbs"] = 0

        return out.dropna(subset=["timestamp", "glucose_mg_dl"]).sort_values("timestamp").reset_index(drop=True)


ALL_PARSERS: List[BaseCGMParser] = [
    FreeStyleLibreParser(),
    DexcomParser(),
    AppleHealthParser(),
    CanonicalParser(),  # Fallback canonical
]


def detect_and_parse_cgm(
    file_or_df: Union[pd.DataFrame, io.StringIO, io.BytesIO, str, Path],
) -> Tuple[pd.DataFrame, str]:
    """Auto-detect export format, parse, and return (canonical_df, detected_format_name)."""
    if isinstance(file_or_df, pd.DataFrame):
        df = file_or_df.copy()
    else:
        # If it's a file path string or Path object that exists, open or read it
        if isinstance(file_or_df, (str, Path)) and os.path.exists(str(file_or_df)):
            try:
                df = pd.read_csv(file_or_df, comment="#")
            except Exception:
                df = pd.read_csv(file_or_df, skiprows=1)
        else:
            # Handle buffer / in-memory string
            def try_read(seek_pos: int = 0, **kwargs) -> pd.DataFrame:
                if hasattr(file_or_df, "seek"):
                    file_or_df.seek(seek_pos)
                return pd.read_csv(file_or_df, **kwargs)

            try:
                df = try_read(0, comment="#")
            except Exception:
                try:
                    df = try_read(0)
                except Exception:
                    df = try_read(0, skiprows=1)

    for parser in ALL_PARSERS:
        if parser.can_parse(df):
            parsed = parser.parse(df)
            logger.info("Parsed %d CGM readings using parser: %s", len(parsed), parser.format_name)
            return parsed, parser.format_name

    # If no vendor parser matched, attempt canonical parser
    canonical = CanonicalParser()
    parsed = canonical.parse(df)
    return parsed, canonical.format_name
