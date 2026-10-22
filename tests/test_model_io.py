"""
test_model_io.py – Unit tests for model I/O
=============================================
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

from src.models.fusion_model import (
    FusionModel,
    SensorOnlyModel,
    StaticOnlyModel,
    build_model_from_config,
)
from src.utils.config import load_config


@pytest.fixture(scope="module")
def config():
    return load_config()


class TestFusionModel:
    """Tests for the FusionModel architecture."""

    def test_forward_shape(self) -> None:
        model = FusionModel(gru_input_dim=5, gru_hidden_dim=32,
                            gru_num_layers=1, static_input_dim=10,
                            static_hidden_dim=16, static_embed_dim=8,
                            fusion_hidden_dim=16)
        ts = torch.randn(4, 288, 5)     # batch=4, seq=288, channels=5
        static = torch.randn(4, 10)     # batch=4, features=10
        out = model(ts, static)
        assert out.shape == (4, 1)

    def test_sensor_only_forward(self) -> None:
        model = SensorOnlyModel(gru_input_dim=5, gru_hidden_dim=32,
                                gru_num_layers=1, fusion_hidden_dim=16)
        ts = torch.randn(2, 288, 5)
        out = model(ts)
        assert out.shape == (2, 1)

    def test_static_only_forward(self) -> None:
        model = StaticOnlyModel(static_input_dim=10, static_hidden_dim=16,
                                static_embed_dim=8, fusion_hidden_dim=16)
        static = torch.randn(2, 10)
        out = model(static=static)
        assert out.shape == (2, 1)

    def test_build_from_config(self, config) -> None:
        for mt in ["fusion", "sensor_only", "static_only"]:
            model = build_model_from_config(config, mt)
            assert isinstance(model, torch.nn.Module)

    def test_save_load_roundtrip(self) -> None:
        model = FusionModel(gru_input_dim=5, gru_hidden_dim=16,
                            gru_num_layers=1, static_input_dim=10,
                            static_hidden_dim=8, static_embed_dim=4,
                            fusion_hidden_dim=8)
        model.eval()
        ts = torch.randn(1, 100, 5)
        static = torch.randn(1, 10)
        with torch.no_grad():
            original_out = model(ts, static).detach()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
            torch.save(model.state_dict(), path)

        model2 = FusionModel(gru_input_dim=5, gru_hidden_dim=16,
                             gru_num_layers=1, static_input_dim=10,
                             static_hidden_dim=8, static_embed_dim=4,
                             fusion_hidden_dim=8)
        model2.load_state_dict(torch.load(path, weights_only=True))
        model2.eval()
        with torch.no_grad():
            loaded_out = model2(ts, static).detach()

        assert torch.allclose(original_out, loaded_out, atol=1e-5)

        Path(path).unlink(missing_ok=True)

    def test_gradient_flow(self) -> None:
        model = FusionModel(gru_input_dim=5, gru_hidden_dim=16,
                            gru_num_layers=1, static_input_dim=10,
                            static_hidden_dim=8, static_embed_dim=4,
                            fusion_hidden_dim=8)
        ts = torch.randn(2, 50, 5)
        static = torch.randn(2, 10)
        out = model(ts, static)
        loss = out.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
