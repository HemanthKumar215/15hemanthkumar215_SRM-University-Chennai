"""
test_data_real.py – Tests for Real-World Data Adapters
======================================================
Tests AI-READI, CGMacros, and Shanghai adapters for schema conformance,
modality availability, and missing value handling.
"""

from __future__ import annotations

import pytest
from pathlib import Path
import pandas as pd

from src.utils.config import get_project_root
from src.data_real import (
    AIREADIAdapter,
    CGMacrosAdapter,
    ShanghaiAdapter,
    CANONICAL_CGM_COLS,
    CANONICAL_EHR_COLS,
)


@pytest.fixture(scope="module")
def real_data_dir() -> Path:
    root = get_project_root()
    return root / "data" / "real"


class TestRealAdapters:
    """Test all real dataset adapters."""

    def test_aireadi_adapter(self, real_data_dir: Path) -> None:
        adapter = AIREADIAdapter(real_data_dir / "aireadi")
        assert adapter.dataset_name == "aireadi"
        assert "glucose" in adapter.available_modalities
        assert "heart_rate" in adapter.available_modalities

        cgm_df, ehr_df = adapter.load_all()
        assert len(cgm_df) > 0
        assert len(ehr_df) > 0
        for col in CANONICAL_CGM_COLS:
            assert col in cgm_df.columns
        for col in CANONICAL_EHR_COLS:
            assert col in ehr_df.columns

    def test_cgmacros_adapter(self, real_data_dir: Path) -> None:
        adapter = CGMacrosAdapter(real_data_dir / "cgmacros")
        assert adapter.dataset_name == "cgmacros"
        assert "glucose" in adapter.available_modalities
        assert "heart_rate" not in adapter.available_modalities

        cgm_df, ehr_df = adapter.load_all()
        assert len(cgm_df) > 0
        assert len(ehr_df) > 0
        for col in CANONICAL_CGM_COLS:
            assert col in cgm_df.columns
        for col in CANONICAL_EHR_COLS:
            assert col in ehr_df.columns

    def test_shanghai_adapter(self, real_data_dir: Path) -> None:
        adapter = ShanghaiAdapter(real_data_dir / "shanghai")
        assert adapter.dataset_name == "shanghai"
        assert "glucose" in adapter.available_modalities
        assert "meal_carbs" in adapter.available_modalities

        cgm_df, ehr_df = adapter.load_all()
        assert len(cgm_df) > 0
        assert len(ehr_df) > 0
        for col in CANONICAL_CGM_COLS:
            assert col in cgm_df.columns
        for col in CANONICAL_EHR_COLS:
            assert col in ehr_df.columns
        # Verify 5-minute sampling alignment
        time_diffs = cgm_df.groupby("patient_id")["timestamp"].diff().dropna()
        # Most diffs should be 5 minutes
        assert (time_diffs == pd.Timedelta("5min")).mean() > 0.85
