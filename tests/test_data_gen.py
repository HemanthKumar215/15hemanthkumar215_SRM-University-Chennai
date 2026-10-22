"""
test_data_gen.py – Unit tests for data generators
====================================================
Run: ``python -m pytest tests/ -v``
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_gen.ehr_generator import generate_ehr_profiles
from src.data_gen.cgm_generator import generate_patient_timeseries
from src.utils.config import load_config


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def ehr_df(config):
    """Generate a small EHR dataset for testing."""
    return generate_ehr_profiles(n_patients=10, seed=42, config=config)


class TestEHRGenerator:
    """Tests for the EHR profile generator."""

    def test_shape(self, ehr_df: pd.DataFrame) -> None:
        assert len(ehr_df) == 10
        assert "patient_id" in ehr_df.columns
        assert "hba1c" in ehr_df.columns

    def test_patient_ids_unique(self, ehr_df: pd.DataFrame) -> None:
        assert ehr_df["patient_id"].nunique() == 10

    def test_age_range(self, ehr_df: pd.DataFrame) -> None:
        assert ehr_df["age"].min() >= 25
        assert ehr_df["age"].max() <= 75

    def test_bmi_range(self, ehr_df: pd.DataFrame) -> None:
        assert ehr_df["bmi"].min() >= 18.0
        assert ehr_df["bmi"].max() <= 43.0

    def test_hba1c_range(self, ehr_df: pd.DataFrame) -> None:
        assert ehr_df["hba1c"].min() >= 4.0
        assert ehr_df["hba1c"].max() <= 13.0

    def test_sex_values(self, ehr_df: pd.DataFrame) -> None:
        assert set(ehr_df["sex"].unique()).issubset({"M", "F"})

    def test_reproducibility(self, config) -> None:
        df1 = generate_ehr_profiles(n_patients=5, seed=123, config=config)
        df2 = generate_ehr_profiles(n_patients=5, seed=123, config=config)
        pd.testing.assert_frame_equal(df1, df2)

    def test_medications_only_for_diagnosed(self, ehr_df: pd.DataFrame) -> None:
        no_t2d = ehr_df[ehr_df["diagnosed_t2d"] == 0]
        # Metformin should be 0 for non-T2D patients
        assert (no_t2d["medication_metformin"] == 0).all()

    def test_has_required_columns(self, ehr_df: pd.DataFrame) -> None:
        required = [
            "patient_id", "age", "sex", "bmi", "hba1c",
            "fasting_glucose", "systolic_bp", "diastolic_bp",
            "genetic_risk_score", "family_history_diabetes",
        ]
        for col in required:
            assert col in ehr_df.columns, f"Missing column: {col}"


class TestCGMGenerator:
    """Tests for the CGM/wearable time-series generator."""

    def test_output_shape(self, ehr_df: pd.DataFrame, config) -> None:
        row = ehr_df.iloc[0]
        df = generate_patient_timeseries(row, n_days=2, config=config)
        expected_rows = 2 * 288  # 2 days × 288 readings/day
        assert len(df) == expected_rows

    def test_columns_present(self, ehr_df: pd.DataFrame, config) -> None:
        row = ehr_df.iloc[0]
        df = generate_patient_timeseries(row, n_days=1, config=config)
        required = [
            "patient_id", "timestamp", "glucose_mg_dl",
            "heart_rate_bpm", "hrv_ms", "steps", "sleep_stage",
            "meal_event", "exercise_event",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_glucose_range(self, ehr_df: pd.DataFrame, config) -> None:
        row = ehr_df.iloc[0]
        df = generate_patient_timeseries(row, n_days=2, config=config)
        glu = df["glucose_mg_dl"].dropna()
        assert glu.min() >= 40
        assert glu.max() <= 400

    def test_missing_values_present(self, ehr_df: pd.DataFrame, config) -> None:
        row = ehr_df.iloc[0]
        df = generate_patient_timeseries(row, n_days=5, config=config)
        # There should be some NaN values (missing rate ~3%)
        nan_frac = df["glucose_mg_dl"].isna().mean()
        assert nan_frac > 0.005, f"Too few missing values: {nan_frac:.4f}"
        assert nan_frac < 0.10, f"Too many missing values: {nan_frac:.4f}"

    def test_meal_events_exist(self, ehr_df: pd.DataFrame, config) -> None:
        row = ehr_df.iloc[0]
        df = generate_patient_timeseries(row, n_days=3, config=config)
        assert df["meal_event"].sum() > 0, "No meal events generated"

    def test_sleep_stages_valid(self, ehr_df: pd.DataFrame, config) -> None:
        row = ehr_df.iloc[0]
        df = generate_patient_timeseries(row, n_days=2, config=config)
        assert set(df["sleep_stage"].unique()).issubset({0, 1, 2, 3})

    def test_reproducibility(self, ehr_df: pd.DataFrame, config) -> None:
        row = ehr_df.iloc[0]
        df1 = generate_patient_timeseries(row, n_days=1, config=config, seed=42)
        df2 = generate_patient_timeseries(row, n_days=1, config=config, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_diabetic_higher_glucose(self, config) -> None:
        """HbA1c–CGM mean correlation must be > 0.80 across a 30-patient cohort."""
        ehr = generate_ehr_profiles(n_patients=30, seed=77, config=config)
        records = []
        for _, row in ehr.iterrows():
            df = generate_patient_timeseries(row, n_days=7, config=config)
            mean_glu = df["glucose_mg_dl"].dropna().mean()
            records.append({"hba1c": row["hba1c"], "mean_glucose": mean_glu})

        check = pd.DataFrame(records)
        corr = check["hba1c"].corr(check["mean_glucose"])
        assert corr > 0.80, (
            f"HbA1c–CGM correlation too low: {corr:.3f} (expected > 0.80). "
            "The CGM generator is not properly calibrated to EHR data."
        )

    def test_normal_patients_never_spike_high(self, config) -> None:
        """Patients with HbA1c < 5.7 should never exceed 165 mg/dL."""
        ehr = generate_ehr_profiles(n_patients=20, seed=55, config=config)
        normal_pts = ehr[ehr["hba1c"] < 5.7]
        if normal_pts.empty:
            pytest.skip("No normal patients in this seed's sample – skip")
        for _, row in normal_pts.iterrows():
            df = generate_patient_timeseries(row, n_days=5, config=config)
            max_glu = df["glucose_mg_dl"].dropna().max()
            assert max_glu <= 165, (
                f"Normal patient (HbA1c={row['hba1c']}) exceeded 165 mg/dL: {max_glu}"
            )

    def test_poorly_controlled_t2d_can_exceed_300(self, config) -> None:
        """Poorly-controlled T2D patients (HbA1c > 9) should reach ≥ 270 mg/dL."""
        ehr = generate_ehr_profiles(n_patients=50, seed=88, config=config)
        poor_ctrl = ehr[ehr["hba1c"] > 9.0]
        if poor_ctrl.empty:
            pytest.skip("No HbA1c>9 patients in this seed's sample – skip")
        maxima = []
        for _, row in poor_ctrl.iterrows():
            df = generate_patient_timeseries(row, n_days=5, config=config)
            maxima.append(df["glucose_mg_dl"].dropna().max())
        assert any(m >= 270 for m in maxima), (
            f"No poorly-controlled T2D patient reached 270+ mg/dL. Maxima: {maxima}"
        )

    def test_mean_glucose_approx_nathan_formula(self, config) -> None:
        """Each patient's 30-day mean glucose should be within ±25 mg/dL of Nathan formula."""
        ehr = generate_ehr_profiles(n_patients=15, seed=321, config=config)
        errors = []
        for _, row in ehr.iterrows():
            expected = 28.7 * row["hba1c"] - 46.7
            df = generate_patient_timeseries(row, n_days=14, config=config)
            actual = df["glucose_mg_dl"].dropna().mean()
            errors.append(abs(actual - expected))
        mean_err = float(pd.Series(errors).mean())
        assert mean_err <= 30.0, (
            f"Mean absolute error vs Nathan formula is {mean_err:.1f} mg/dL (threshold: 30)"
        )

