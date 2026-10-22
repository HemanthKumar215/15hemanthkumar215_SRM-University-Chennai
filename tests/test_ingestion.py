"""
test_ingestion.py – Unit Tests for Upload Validation and Vendor Parsers
=======================================================================
Tests valid, malformed, and unit-mismatched CGM and EHR files across
FreeStyle Libre, Dexcom, Apple Health, and Canonical formats.
"""

from __future__ import annotations

import io
import json
import numpy as np
import pandas as pd
import pytest

from src.ingestion.parsers import (
    FreeStyleLibreParser,
    DexcomParser,
    AppleHealthParser,
    CanonicalParser,
    detect_and_parse_cgm,
)
from src.ingestion.validator import (
    validate_cgm_data,
    validate_ehr_data,
    validate_patient_data,
)


class TestVendorParsers:
    """Test format detection and parsing of vendor exports."""

    def test_canonical_parser(self) -> None:
        csv_data = io.StringIO(
            "timestamp,glucose_mg_dl,heart_rate_bpm\n"
            "2026-03-01 00:00:00,120.5,70\n"
            "2026-03-01 00:05:00,122.0,71\n"
        )
        parsed, fmt = detect_and_parse_cgm(csv_data)
        assert fmt == "Digital Twin Canonical"
        assert len(parsed) == 2
        assert "timestamp" in parsed.columns
        assert "glucose_mg_dl" in parsed.columns
        assert parsed["glucose_mg_dl"].iloc[0] == 120.5

    def test_freestyle_libre_parser(self) -> None:
        csv_data = io.StringIO(
            "Device Timestamp,Record Type,Historic Glucose mg/dL,Scan Glucose mg/dL\n"
            "01-03-2026 00:00,0,135,\n"
            "01-03-2026 00:15,0,140,\n"
            "01-03-2026 00:30,0,148,\n"
        )
        parsed, fmt = detect_and_parse_cgm(csv_data)
        assert fmt == "Abbott FreeStyle Libre"
        assert len(parsed) == 3
        assert parsed["glucose_mg_dl"].iloc[1] == 140.0

    def test_dexcom_parser(self) -> None:
        csv_data = io.StringIO(
            "Timestamp (YYYY-MM-DDThh:mm:ss),Event Type,Glucose Value (mg/dL)\n"
            "2026-03-01T00:00:00,EGV,115\n"
            "2026-03-01T00:05:00,EGV,118\n"
            "2026-03-01T00:10:00,EGV,125\n"
        )
        parsed, fmt = detect_and_parse_cgm(csv_data)
        assert fmt == "Dexcom Clarity"
        assert len(parsed) == 3
        assert parsed["glucose_mg_dl"].iloc[2] == 125.0

    def test_apple_health_parser(self) -> None:
        csv_data = io.StringIO(
            "type,startDate,value,unit\n"
            "HKQuantityTypeIdentifierBloodGlucose,2026-03-01 00:00:00,110.0,mg/dL\n"
            "HKQuantityTypeIdentifierBloodGlucose,2026-03-01 00:05:00,114.5,mg/dL\n"
            "HKQuantityTypeIdentifierHeartRate,2026-03-01 00:00:00,68.0,count/min\n"
        )
        parsed, fmt = detect_and_parse_cgm(csv_data)
        assert fmt == "Apple Health"
        assert len(parsed) == 2
        assert parsed["glucose_mg_dl"].iloc[0] == 110.0


class TestValidatorAndCleaning:
    """Test validator unit conversion, gap interpolation, range checks, and EHR handling."""

    def test_mmol_to_mgdl_conversion(self) -> None:
        # Generate 25 hours of data in mmol/L (typical values 5.5 to 8.0)
        times = pd.date_range("2026-03-01 00:00:00", periods=300, freq="5min")
        glu_mmol = np.full(300, 6.5)  # 6.5 mmol/L = ~117.1 mg/dL
        df = pd.DataFrame({"timestamp": times, "glucose_mg_dl": glu_mmol})

        ok, errs, warns, cleaned, fmt, dur, converted = validate_cgm_data(df)
        assert ok is True
        assert converted is True
        assert any("mmol/L" in w for w in warns)
        assert cleaned["glucose_mg_dl"].median() > 100.0  # Converted to ~117.1 mg/dL

    def test_insufficient_duration_rejected(self) -> None:
        # Only 6 hours of data
        times = pd.date_range("2026-03-01 00:00:00", periods=72, freq="5min")
        df = pd.DataFrame({"timestamp": times, "glucose_mg_dl": np.full(72, 120.0)})

        ok, errs, warns, cleaned, fmt, dur, converted = validate_cgm_data(df)
        assert ok is False
        assert any("at least 24 hours" in e for e in errs)

    def test_gap_interpolation_and_flagging(self) -> None:
        times = pd.date_range("2026-03-01 00:00:00", periods=300, freq="5min")
        glu = np.full(300, 120.0)
        # Small gap of 3 readings (15 mins) -> should be interpolated
        glu[50:53] = np.nan
        # Large gap of 12 readings (60 mins) -> should be flagged as warning
        glu[150:162] = np.nan
        df = pd.DataFrame({"timestamp": times, "glucose_mg_dl": glu})

        ok, errs, warns, cleaned, fmt, dur, converted = validate_cgm_data(df)
        assert ok is True
        assert any("sensor gaps" in w for w in warns)
        # Verify no remaining NaNs in cleaned data
        assert cleaned["glucose_mg_dl"].isna().sum() == 0

    def test_ehr_csv_validation_and_imputation(self) -> None:
        # EHR with minimal required fields, missing SBP/DBP/cholesterol
        ehr_csv = io.StringIO(
            "name,age,sex,bmi,hba1c\n"
            "Jane Doe,48,F,29.5,7.4\n"
        )
        ok, errs, warns, cleaned_ehr, imputed = validate_ehr_data(ehr_csv)
        assert ok is True
        assert cleaned_ehr["age"] == 48
        assert cleaned_ehr["sex"] == "F"
        assert cleaned_ehr["bmi"] == 29.5
        assert cleaned_ehr["hba1c"] == 7.4
        # Cohort inferred from HbA1c 7.4 -> T2D
        assert cleaned_ehr["cohort"] == "T2D"
        # Check imputed optional fields
        assert "systolic_bp" in imputed
        assert cleaned_ehr["systolic_bp"] == 125

    def test_ehr_json_validation(self) -> None:
        ehr_dict = {
            "patient_id": "TEST_JSON",
            "age": 55,
            "sex": "M",
            "bmi": 26.2,
            "hba1c": 6.1,
            "fasting_glucose": 112.0,
            "cohort": "Prediabetic",
        }
        ok, errs, warns, cleaned_ehr, imputed = validate_ehr_data(json.dumps(ehr_dict))
        assert ok is True
        assert cleaned_ehr["cohort"] == "Prediabetic"
        assert cleaned_ehr["patient_id"] == "TEST_JSON"

    def test_full_pipeline_validation(self) -> None:
        times = pd.date_range("2026-03-01 00:00:00", periods=300, freq="5min")
        cgm_df = pd.DataFrame({"timestamp": times, "glucose_mg_dl": np.full(300, 130.0)})
        ehr_dict = {"age": 60, "sex": "M", "bmi": 27.0, "hba1c": 7.0}

        res = validate_patient_data(cgm_df, ehr_dict)
        assert res.is_valid is True
        assert res.cleaned_cgm is not None
        assert res.cleaned_ehr is not None
        assert len(res.cleaned_cgm) == 300
        assert res.duration_hours >= 24.0

    def test_malformed_cgm_files_rejected(self) -> None:
        # 1. Missing timestamp and glucose
        bad_df = pd.DataFrame({"notes": ["random text", "more text"]})
        ok, errs, warns, cleaned, fmt, dur, conv = validate_cgm_data(bad_df)
        assert ok is False
        assert len(errs) > 0

        # 2. Corrupt non-numeric glucose values
        times = pd.date_range("2026-03-01 00:00:00", periods=300, freq="5min")
        corrupt_df = pd.DataFrame({"timestamp": times, "glucose_mg_dl": ["N/A"] * 300})
        ok, errs, warns, cleaned, fmt, dur, conv = validate_cgm_data(corrupt_df)
        assert ok is False
        assert any("no valid numeric glucose" in e.lower() for e in errs)

    def test_malformed_ehr_files_rejected(self) -> None:
        # 1. Missing required field 'age'
        bad_ehr1 = {"sex": "M", "bmi": 28.0, "hba1c": 6.5}
        ok, errs, warns, cleaned, imp = validate_ehr_data(bad_ehr1)
        assert ok is False
        assert any("missing required clinical field: 'age'" in e.lower() for e in errs)

        # 2. Implausible physiological values (e.g. BMI = 5.0, Age = 150)
        bad_ehr2 = {"age": 150, "sex": "M", "bmi": 5.0, "hba1c": 6.5}
        ok, errs, warns, cleaned, imp = validate_ehr_data(bad_ehr2)
        assert ok is False
        assert any("age (150" in e.lower() for e in errs)
        assert any("bmi (5.0" in e.lower() for e in errs)

    def test_unphysiological_glucose_rejected(self) -> None:
        # All glucose values 1000 mg/dL (unphysiological)
        times = pd.date_range("2026-03-01 00:00:00", periods=300, freq="5min")
        df_unphys = pd.DataFrame({"timestamp": times, "glucose_mg_dl": [1000.0] * 300})
        ok, errs, warns, cleaned, fmt, dur, conv = validate_cgm_data(df_unphys)
        assert ok is False
        assert any("plausible" in e.lower() for e in errs)

    def test_high_missing_rate_triggers_warning(self) -> None:
        times = pd.date_range("2026-03-01 00:00:00", periods=300, freq="5min")
        glu = np.full(300, 120.0)
        # 25% missing data
        glu[:75] = np.nan
        df = pd.DataFrame({"timestamp": times, "glucose_mg_dl": glu})
        ok, errs, warns, cleaned, fmt, dur, conv = validate_cgm_data(df)
        assert ok is True
        assert any("high missing data rate" in w.lower() for w in warns)

    def test_template_files_are_valid(self) -> None:
        from pathlib import Path
        tmpl_cgm = Path("data/templates/cgm_template.csv")
        tmpl_ehr = Path("data/templates/ehr_template.csv")
        assert tmpl_cgm.exists()
        assert tmpl_ehr.exists()

        res = validate_patient_data(tmpl_cgm, tmpl_ehr)
        assert res.is_valid is True
        assert res.duration_hours >= 24.0
        assert res.cleaned_cgm is not None
        assert res.cleaned_ehr is not None
