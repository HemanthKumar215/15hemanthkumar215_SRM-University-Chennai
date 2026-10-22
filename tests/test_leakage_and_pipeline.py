"""
test_leakage_and_pipeline.py – Tests for Leakage Checks & Pipeline
==================================================================
Verifies patient disjointness assertions, temporal forward-only constraints,
LazySpikeDataset batching, and high AUROC sanity audit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.utils.sanity_checks import (
    assert_no_patient_overlap,
    assert_no_future_leakage,
    investigate_high_auroc,
)
from src.features.lazy_dataset import LazySpikeDataset
from src.utils.config import load_config, get_project_root


class TestSanityAndLeakage:
    """Test data leakage and sanity check utilities."""

    def test_patient_overlap_detection(self) -> None:
        train_p = ["P001", "P002", "P003"]
        val_p = ["P004", "P005"]
        test_p = ["P006", "P007"]
        # Disjoint should pass without error
        assert_no_patient_overlap(train_p, val_p, test_p)

        # Overlap should raise AssertionError
        leaky_val = ["P002", "P005"]
        with pytest.raises(AssertionError, match="Data leakage detected"):
            assert_no_patient_overlap(train_p, leaky_val, test_p)

    def test_future_leakage_detection(self) -> None:
        lookback = pd.date_range("2026-01-01 00:00:00", "2026-01-01 23:55:00", freq="5min")
        anchor = pd.Timestamp("2026-01-01 23:55:00")
        future = pd.date_range("2026-01-02 00:00:00", "2026-01-02 02:00:00", freq="5min")

        # Valid temporal sequence
        assert_no_future_leakage(lookback, anchor, future)

        # Leaky future window overlapping lookback
        leaky_anchor = pd.Timestamp("2026-01-01 20:00:00")
        with pytest.raises(AssertionError, match="Temporal leakage"):
            assert_no_future_leakage(lookback, leaky_anchor, future)

    def test_high_auroc_audit(self) -> None:
        # Realistic AUROC should not flag
        res_ok = investigate_high_auroc(0.895, "FusionGRU")
        assert not res_ok["flagged"]
        assert len(res_ok["analysis"]) == 0

        # Suspiciously high AUROC (> 0.97) should flag with detailed root cause
        res_flagged = investigate_high_auroc(0.985, "TabularLightGBM")
        assert res_flagged["flagged"]
        assert len(res_flagged["analysis"]) > 0

    def test_lazy_dataset_structure(self) -> None:
        root = get_project_root()
        config = load_config()
        ehr_path = root / "data" / "sample" / "ehr_profiles_sample.csv"
        cgm_path = root / "data" / "sample" / "cgm_wearable_sample.parquet"

        ehr_df = pd.read_csv(ehr_path)
        cgm_df = pd.read_parquet(cgm_path)
        pids = ehr_df["patient_id"].tolist()[:3]

        ds = LazySpikeDataset(cgm_df, ehr_df, pids, config, max_samples_per_patient=10)
        assert len(ds) > 0

        ts, static, label = ds[0]
        assert isinstance(ts, torch.Tensor)
        assert isinstance(static, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert ts.ndim == 2
        assert ts.shape[0] == 288  # 24h lookback @ 5min
        assert ts.shape[1] == 5    # 5 sensor channels
        assert label.item() in (0, 1)
