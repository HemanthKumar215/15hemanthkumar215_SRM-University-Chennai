"""
test_features.py – Unit tests for feature engineering
=======================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_gen.ehr_generator import generate_ehr_profiles
from src.data_gen.cgm_generator import generate_patient_timeseries
from src.features.feature_engineering import (
    encode_static_features,
    build_feature_matrix,
)
from src.utils.config import load_config


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def small_data(config):
    """Generate small dataset for feature tests."""
    ehr = generate_ehr_profiles(n_patients=5, seed=42, config=config)
    frames = []
    for _, row in ehr.iterrows():
        frames.append(generate_patient_timeseries(row, n_days=3, config=config))
    cgm = pd.concat(frames, ignore_index=True)
    return ehr, cgm


class TestStaticFeatures:
    """Tests for static EHR feature encoding."""

    def test_output_shape(self, small_data) -> None:
        ehr, _ = small_data
        encoded = encode_static_features(ehr)
        assert len(encoded) == len(ehr)
        assert "patient_id" in encoded.columns
        assert "sex_male" in encoded.columns

    def test_no_nan(self, small_data) -> None:
        ehr, _ = small_data
        encoded = encode_static_features(ehr)
        non_id_cols = [c for c in encoded.columns if c != "patient_id"]
        assert not encoded[non_id_cols].isna().any().any()

    def test_sex_encoding(self, small_data) -> None:
        ehr, _ = small_data
        encoded = encode_static_features(ehr)
        assert set(encoded["sex_male"].unique()).issubset({0, 1})


class TestFeatureMatrix:
    """Tests for the full feature matrix builder."""

    def test_build_runs(self, small_data, config) -> None:
        ehr, cgm = small_data
        features, labels = build_feature_matrix(
            cgm, ehr, config, subsample_negative=1.0,
        )
        assert len(features) > 0
        assert len(features) == len(labels)

    def test_labels_binary(self, small_data, config) -> None:
        ehr, cgm = small_data
        _, labels = build_feature_matrix(
            cgm, ehr, config, subsample_negative=1.0,
        )
        assert set(labels.unique()).issubset({0, 1})

    def test_has_sensor_and_static_features(self, small_data, config) -> None:
        ehr, cgm = small_data
        features, _ = build_feature_matrix(
            cgm, ehr, config, subsample_negative=1.0,
        )
        # Should have both rolling glucose features and static features
        assert any("gluc_mean" in c for c in features.columns)
        assert "hba1c" in features.columns or "bmi" in features.columns
