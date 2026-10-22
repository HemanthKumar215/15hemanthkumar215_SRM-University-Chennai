"""
validator.py – Ingestion Validation, Cleaning & Unit Harmonization
==================================================================
Validates uploaded EHR and CGM files:
- Checks required columns, types, and plausible physiological ranges.
- Auto-detects and converts glucose units (mmol/L -> mg/dL).
- Auto-detects vendor formats (FreeStyle Libre, Dexcom, Apple Health).
- Resamples to a uniform 5-minute grid, interpolates gaps <= 30 mins, and flags larger gaps.
- Verifies at least 24 hours of lookback history.
- Imputes missing optional EHR fields with cohort medians, tracking imputed values.
- In-memory only: zero disk persistence for user privacy.
"""

from __future__ import annotations

import io
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.ingestion.parsers import detect_and_parse_cgm

logger = logging.getLogger(__name__)

# Standard cohort medians for EHR field imputation
COHORT_MEDIANS: Dict[str, Any] = {
    "systolic_bp": 125,
    "diastolic_bp": 80,
    "total_cholesterol": 190.0,
    "ldl": 110.0,
    "hdl": 45.0,
    "triglycerides": 150.0,
    "smoker": 0,
    "physical_activity_level": "moderate",
    "diagnosed_hypertension": 0,
    "diagnosed_dyslipidemia": 0,
    "medication_metformin": 0,
    "medication_sulfonylurea": 0,
    "medication_insulin": 0,
    "medication_dpp4i": 0,
    "medication_sglt2i": 0,
    "medication_antihypertensive": 0,
    "medication_statin": 0,
    "family_history_diabetes": 0,
    "family_history_cvd": 0,
    "genetic_risk_score": 0.50,
}


@dataclass
class ValidationResult:
    """Structured report returned by validate_patient_data."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    cleaned_cgm: Optional[pd.DataFrame] = None
    cleaned_ehr: Optional[pd.Series] = None
    imputed_fields: Dict[str, Any] = field(default_factory=dict)
    format_detected: str = "Unknown"
    duration_hours: float = 0.0
    units_converted: bool = False


def validate_cgm_data(
    file_or_df: Union[pd.DataFrame, io.StringIO, io.BytesIO, str],
) -> Tuple[bool, List[str], List[str], Optional[pd.DataFrame], str, float, bool]:
    """Validate, clean, resample, and harmonize CGM time-series data."""
    errors: List[str] = []
    warnings: List[str] = []
    units_converted = False

    try:
        raw_df, format_name = detect_and_parse_cgm(file_or_df)
    except Exception as e:
        return False, [f"Failed to parse CGM file: {str(e)}"], [], None, "Unknown", 0.0, False

    if raw_df.empty or "timestamp" not in raw_df.columns or "glucose_mg_dl" not in raw_df.columns:
        return False, ["CGM data must contain valid 'timestamp' and glucose readings."], [], None, format_name, 0.0, False

    df = raw_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if len(df) < 20:
        return False, ["CGM file contains too few readings to establish a physiological timeline."], [], None, format_name, 0.0, False

    # ── Unit detection: mmol/L vs mg/dL ────────────────────────────────────
    valid_glu = df["glucose_mg_dl"].dropna()
    if valid_glu.empty:
        return False, ["No valid numeric glucose readings found."], [], None, format_name, 0.0, False

    med_glu = float(valid_glu.median())
    if med_glu < 30.0:
        # Detected mmol/L -> convert to mg/dL
        df["glucose_mg_dl"] = (df["glucose_mg_dl"] * 18.0182).round(1)
        units_converted = True
        warnings.append(f"Auto-detected glucose in mmol/L (median: {med_glu:.1f}). Converted to mg/dL (median: {df['glucose_mg_dl'].median():.1f}).")
        med_glu = float(df["glucose_mg_dl"].dropna().median())

    # ── Plausible range checks ─────────────────────────────────────────────
    if med_glu < 30.0 or med_glu > 700.0:
        errors.append(f"Median glucose reading ({med_glu:.1f} mg/dL) is outside plausible physiological range (30–700 mg/dL).")
        return False, errors, warnings, None, format_name, 0.0, units_converted

    # Clip extreme physiologically impossible glucose values (< 20 or > 600)
    out_of_bounds = ((df["glucose_mg_dl"] < 20) | (df["glucose_mg_dl"] > 600)).sum()
    if out_of_bounds > 0:
        warnings.append(f"Found {out_of_bounds} readings outside physiological range (20–600 mg/dL). Clipped to bounds.")
        df["glucose_mg_dl"] = df["glucose_mg_dl"].clip(20.0, 600.0)

    # ── Duration check (minimum 24 hours required for prediction) ──────────
    t_start = df["timestamp"].min()
    t_end = df["timestamp"].max()
    duration_hours = (t_end - t_start).total_seconds() / 3600.0

    if duration_hours < 24.0:
        errors.append(
            f"Insufficient data duration: file contains {duration_hours:.1f} hours of data. "
            "The Digital Twin requires at least 24 hours (1 full day) of continuous lookback history."
        )
        return False, errors, warnings, None, format_name, duration_hours, units_converted

    # ── Resample to uniform 5-minute grid ──────────────────────────────────
    # Drop duplicate timestamps
    df = df.drop_duplicates(subset=["timestamp"]).set_index("timestamp")

    # Resample to 5 minutes
    resampled = df.resample("5min").mean()

    # Analyze gaps
    missing_mask = resampled["glucose_mg_dl"].isna()
    missing_pct = 100.0 * missing_mask.mean()

    if missing_pct > 20.0:
        warnings.append(f"High missing data rate ({missing_pct:.1f}% missing). Estimates may have lower confidence.")

    # Interpolate small gaps <= 30 mins (up to 6 consecutive readings)
    interpolated = resampled["glucose_mg_dl"].interpolate(method="linear", limit=6)
    
    # Flag remaining large gaps > 30 mins
    remaining_nans = interpolated.isna()
    if remaining_nans.any():
        n_unfilled = remaining_nans.sum()
        warnings.append(f"Detected {n_unfilled * 5 / 60:.1f} hours of sensor gaps (> 30 min duration). Forward-filled from latest available reading.")
        # Forward fill and backfill any remaining to ensure a complete tensor
        interpolated = interpolated.ffill().bfill()

    resampled["glucose_mg_dl"] = interpolated.round(1)

    # Fill optional wearable channels
    if "heart_rate_bpm" in resampled.columns:
        resampled["heart_rate_bpm"] = resampled["heart_rate_bpm"].interpolate(limit=6).ffill().bfill().fillna(72).round(0)
    else:
        resampled["heart_rate_bpm"] = 72.0

    if "hrv_ms" in resampled.columns:
        resampled["hrv_ms"] = resampled["hrv_ms"].interpolate(limit=6).ffill().bfill().fillna(35.0).round(1)
    else:
        resampled["hrv_ms"] = 35.0

    if "steps" in resampled.columns:
        resampled["steps"] = resampled["steps"].fillna(0).round(0)
    else:
        resampled["steps"] = 0

    if "sleep_stage" in resampled.columns:
        resampled["sleep_stage"] = resampled["sleep_stage"].fillna(0).astype(int)
    else:
        resampled["sleep_stage"] = 0

    if "meal_carbs" in resampled.columns:
        resampled["meal_carbs"] = resampled["meal_carbs"].fillna(0).round(0)
    else:
        resampled["meal_carbs"] = 0

    cleaned = resampled.reset_index()
    return True, errors, warnings, cleaned, format_name, duration_hours, units_converted


def validate_ehr_data(
    file_or_df_or_dict: Union[pd.DataFrame, io.StringIO, io.BytesIO, str, dict],
) -> Tuple[bool, List[str], List[str], Optional[pd.Series], Dict[str, Any]]:
    """Validate, standardize, and impute missing fields in an EHR record."""
    errors: List[str] = []
    warnings: List[str] = []
    imputed_fields: Dict[str, Any] = {}

    # Convert input to a single-row Series
    if isinstance(file_or_df_or_dict, dict):
        ehr_series = pd.Series(file_or_df_or_dict)
    elif isinstance(file_or_df_or_dict, pd.DataFrame):
        if file_or_df_or_dict.empty:
            return False, ["EHR table is empty."], [], None, {}
        ehr_series = file_or_df_or_dict.iloc[0].copy()
    else:
        # Read from buffer/str (supports CSV and JSON)
        content = file_or_df_or_dict
        if hasattr(content, "read"):
            text = content.read()
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="replace")
            content = text
        elif isinstance(content, (str, Path)) and os.path.exists(str(content)):
            with open(str(content), "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

        text_str = str(content).strip()
        if text_str.startswith("{"):
            # JSON format
            try:
                data = json.loads(text_str)
                ehr_series = pd.Series(data)
            except Exception as e:
                return False, [f"Invalid JSON EHR format: {str(e)}"], [], None, {}
        else:
            # CSV format
            try:
                # Read skipping comments (#)
                df = pd.read_csv(io.StringIO(text_str), comment="#")
                if df.empty:
                    return False, ["EHR CSV is empty."], [], None, {}
                ehr_series = df.iloc[0].copy()
            except Exception as e:
                return False, [f"Invalid CSV EHR format: {str(e)}"], [], None, {}

    # Normalize column names
    col_map = {str(k).lower().strip(): k for k in ehr_series.index}

    # Helper to retrieve value
    def get_val(keys: List[str]) -> Any:
        for k in keys:
            if k in col_map:
                v = ehr_series[col_map[k]]
                if pd.notna(v) and str(v).strip() != "":
                    return v
        return None

    # Check primary required demographic / clinical fields
    age = get_val(["age"])
    sex = get_val(["sex", "gender"])
    bmi = get_val(["bmi"])
    hba1c = get_val(["hba1c", "glycated_hemoglobin"])
    fasting_glu = get_val(["fasting_glucose", "fpg", "fasting_plasma_glucose"])
    cohort = get_val(["cohort", "diabetes_status", "diagnosis"])

    if age is None:
        errors.append("Missing required clinical field: 'age'.")
    else:
        try:
            age = float(age)
            if age < 0 or age > 130:
                errors.append(f"Implausible age ({age:.0f}). Must be between 0 and 130.")
            elif age < 18 or age > 115:
                warnings.append(f"Age {age:.0f} is outside standard adult cohort range (18–115).")
        except ValueError:
            errors.append("Field 'age' must be a numeric value.")

    if sex is None:
        warnings.append("Missing 'sex' field; defaulting to 'M'.")
        sex = "M"
        imputed_fields["sex"] = "M"
    else:
        sex = str(sex).strip().upper()[0]
        if sex not in ("M", "F"):
            sex = "M"

    if bmi is None:
        errors.append("Missing required clinical field: 'bmi'.")
    else:
        try:
            bmi = float(bmi)
            if bmi < 10.0 or bmi > 90.0:
                errors.append(f"Implausible BMI ({bmi:.1f}). Must be between 10 and 90.")
            elif bmi < 12.0 or bmi > 75.0:
                warnings.append(f"BMI {bmi:.1f} is outside standard range (12–75).")
        except ValueError:
            errors.append("Field 'bmi' must be a numeric value.")

    if hba1c is None:
        errors.append("Missing required clinical field: 'hba1c'.")
    else:
        try:
            hba1c = float(hba1c)
            if hba1c < 2.0 or hba1c > 25.0:
                errors.append(f"Implausible HbA1c ({hba1c:.1f}%). Must be between 2.0% and 25.0%.")
            elif hba1c < 3.5 or hba1c > 18.0:
                warnings.append(f"HbA1c {hba1c:.1f}% is outside expected clinical bounds (3.5–18.0%).")
        except ValueError:
            errors.append("Field 'hba1c' must be a numeric value.")

    # Fasting glucose: auto-impute from Nathan formula if absent
    if fasting_glu is None:
        if hba1c is not None and isinstance(hba1c, (int, float)):
            fasting_glu = round(float(28.7 * hba1c - 46.7), 1)
            imputed_fields["fasting_glucose"] = f"{fasting_glu:.1f} mg/dL (derived from HbA1c via Nathan formula)"
            warnings.append(f"Estimated missing fasting glucose from HbA1c ({fasting_glu:.1f} mg/dL).")
        else:
            fasting_glu = 120.0
            imputed_fields["fasting_glucose"] = "120.0 mg/dL (cohort median)"
    else:
        try:
            fasting_glu = float(fasting_glu)
            if fasting_glu < 25.0:  # In mmol/L
                fasting_glu = round(fasting_glu * 18.0182, 1)
                warnings.append("Converted fasting glucose from mmol/L to mg/dL.")
        except ValueError:
            fasting_glu = 120.0

    # Cohort: auto-infer from HbA1c if missing
    if cohort is None or str(cohort).strip() == "":
        if hba1c is not None and isinstance(hba1c, (int, float)):
            if hba1c < 5.7:
                cohort = "Normal"
            elif hba1c < 6.5:
                cohort = "Prediabetic"
            else:
                cohort = "T2D"
            imputed_fields["cohort"] = f"{cohort} (inferred from HbA1c: {hba1c}%)"
            warnings.append(f"Inferred clinical cohort as '{cohort}' based on HbA1c ({hba1c}%).")
        else:
            cohort = "T2D"
            imputed_fields["cohort"] = "T2D (default)"

    if errors:
        return False, errors, warnings, None, imputed_fields

    # Build standardized clean Series
    cleaned_dict: Dict[str, Any] = {
        "patient_id": str(get_val(["patient_id", "id", "study_id"]) or "UPLOADED_PATIENT"),
        "name": str(get_val(["name", "full_name"]) or "Uploaded Patient"),
        "age": int(age),
        "sex": str(sex),
        "bmi": float(bmi),
        "hba1c": float(hba1c),
        "fasting_glucose": float(fasting_glu),
        "cohort": str(cohort),
    }

    # Impute optional fields with cohort medians
    for field_name, default_val in COHORT_MEDIANS.items():
        val = get_val([field_name, field_name.replace("_", "")])
        if val is None or pd.isna(val) or str(val).strip() == "":
            cleaned_dict[field_name] = default_val
            imputed_fields[field_name] = default_val
        else:
            try:
                if isinstance(default_val, int):
                    cleaned_dict[field_name] = int(float(val))
                elif isinstance(default_val, float):
                    cleaned_dict[field_name] = float(val)
                else:
                    cleaned_dict[field_name] = val
            except (ValueError, TypeError):
                cleaned_dict[field_name] = default_val
                imputed_fields[field_name] = default_val

    # Ensure diagnoses flag
    cleaned_dict["diagnosed_t2d"] = 1 if "T2D" in str(cleaned_dict["cohort"]).upper() else 0

    return True, errors, warnings, pd.Series(cleaned_dict), imputed_fields


def validate_patient_data(
    cgm_file_or_df: Any,
    ehr_file_or_dict: Any,
) -> ValidationResult:
    """Validate both uploaded files in-memory without any disk writes."""
    cgm_ok, cgm_errs, cgm_warns, cleaned_cgm, fmt_name, dur_h, converted = validate_cgm_data(cgm_file_or_df)
    ehr_ok, ehr_errs, ehr_warns, cleaned_ehr, imputed = validate_ehr_data(ehr_file_or_dict)

    all_errors = cgm_errs + ehr_errs
    all_warnings = cgm_warns + ehr_warns

    is_valid = cgm_ok and ehr_ok and len(all_errors) == 0

    if is_valid and cleaned_cgm is not None and cleaned_ehr is not None:
        # Match patient_id
        pid = cleaned_ehr["patient_id"]
        cleaned_cgm["patient_id"] = pid

    return ValidationResult(
        is_valid=is_valid,
        errors=all_errors,
        warnings=all_warnings,
        cleaned_cgm=cleaned_cgm,
        cleaned_ehr=cleaned_ehr,
        imputed_fields=imputed,
        format_detected=fmt_name,
        duration_hours=dur_h,
        units_converted=converted,
    )
