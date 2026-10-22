"""
test_twin.py – Unit tests for the PatientTwin engine and config loader
=======================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.features.feature_engineering import encode_static_features
from src.models.fusion_model import FusionModel
from src.twin.patient_twin import PatientTwin, _SENSOR_COLS
from src.utils.config import get_project_root, load_config


@pytest.fixture(scope="module")
def twin_parts():
    ehr = pd.read_csv(get_project_root() / "data" / "sample" / "ehr_profiles_sample.csv")
    row = ehr.iloc[0]
    static_dim = encode_static_features(pd.DataFrame([row])).shape[1] - 1
    torch.manual_seed(0)
    model = FusionModel(gru_input_dim=len(_SENSOR_COLS), gru_hidden_dim=8,
                        gru_num_layers=1, static_input_dim=static_dim,
                        static_hidden_dim=8, static_embed_dim=4,
                        fusion_hidden_dim=8)
    norm_stats = {
        "ts_mean": [120.0, 75.0, 50.0, 10.0, 0.5],
        "ts_std": [30.0, 10.0, 15.0, 20.0, 0.0],  # zero std must be guarded
        "static_mean": [0.0] * static_dim,
        "static_std": [1.0] * static_dim,
    }
    return row, model, norm_stats


def _make_twin(twin_parts) -> PatientTwin:
    row, model, norm_stats = twin_parts
    return PatientTwin("P0000", row, model, norm_stats, config=load_config())


def _reading(i: int, **overrides) -> dict:
    r = {"glucose_mg_dl": 120.0 + i % 10, "heart_rate_bpm": 75.0,
         "hrv_ms": 50.0, "steps": 0.0, "sleep_stage": 0.0}
    r.update(overrides)
    return r


class TestPatientTwin:

    def test_predict_is_finite_with_zero_std(self, twin_parts) -> None:
        twin = _make_twin(twin_parts)
        for i in range(twin.lookback):
            twin.ingest_reading(_reading(i))
        res = twin.predict()
        assert res is not None
        assert 0.0 <= res.probability <= 1.0
        assert len(twin.prediction_history) == 1

    def test_missing_vitals_are_forward_filled(self, twin_parts) -> None:
        """Gaps are forward-filled like training, not zero-filled."""
        full = _make_twin(twin_parts)
        gappy = _make_twin(twin_parts)
        for i in range(full.lookback):
            full.ingest_reading(_reading(i))
            if i == 0:
                gappy.ingest_reading(_reading(i))
            else:
                gappy.ingest_reading(_reading(i, heart_rate_bpm=np.nan, hrv_ms=np.nan))
        assert gappy.predict().probability == pytest.approx(full.predict().probability)

    def test_what_if_does_not_touch_history(self, twin_parts) -> None:
        twin = _make_twin(twin_parts)
        for i in range(twin.lookback):
            twin.ingest_reading(_reading(i))
        before = [dict(r) for r in twin.sensor_buffer]
        out = twin.what_if("add_exercise", duration_min=30, intensity=0.8)
        assert "error" not in out
        assert out["modified_prob"] is not None
        assert twin.prediction_history == []
        assert twin.sensor_buffer == before

    def test_explain_matches_occlusion(self, twin_parts) -> None:
        """Attributions are real model deltas, not fixed numbers."""
        twin = _make_twin(twin_parts)
        for i in range(twin.lookback):
            twin.ingest_reading(_reading(i, glucose_mg_dl=170.0 + i % 5))
        attrs = twin.explain(top_k_static=4)
        sensors = [a for a in attrs if a["group"] == "sensor"]
        ehr = [a for a in attrs if a["group"] == "ehr"]
        assert len(sensors) == len(_SENSOR_COLS)
        assert len(ehr) == 4
        assert [abs(a["impact"]) for a in attrs] == sorted((abs(a["impact"]) for a in attrs), reverse=True)

        # Recompute the glucose attribution by hand
        ts, st = twin._normalised_inputs(twin.sensor_buffer[-twin.lookback:])
        occluded = ts.copy()
        occluded[:, 0] = 0.0
        base, occ = twin._probabilities(np.stack([ts, occluded]), np.stack([st, st]))
        glucose = next(a for a in sensors if a["feature"].startswith("CGM glucose"))
        assert glucose["impact"] == pytest.approx((base - occ) * 100, abs=0.01)
        assert twin.prediction_history == []

    def test_explain_needs_full_window(self, twin_parts) -> None:
        twin = _make_twin(twin_parts)
        twin.ingest_reading(_reading(0))
        assert twin.explain() == []

    def test_what_if_unknown_scenario(self, twin_parts) -> None:
        twin = _make_twin(twin_parts)
        for i in range(twin.lookback):
            twin.ingest_reading(_reading(i))
        assert "error" in twin.what_if("teleport")


class TestLoadConfig:

    def test_explicit_path_is_honoured(self, tmp_path) -> None:
        default = load_config()
        custom = tmp_path / "custom.yaml"
        custom.write_text("model:\n  lookback_hours: 99\n", encoding="utf-8")
        assert load_config(custom)["model"]["lookback_hours"] == 99
        # Default cache is not replaced by the custom file
        assert load_config() is default
