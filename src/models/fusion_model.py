"""
fusion_model.py – Two-branch GRU + MLP fusion network (PyTorch)
=================================================================
Architecture
------------
1. **GRU branch**: encodes the last 24 h of sensor time-series
   (glucose, HR, HRV, steps, sleep_stage) into a fixed-length embedding.
2. **MLP branch**: encodes static EHR features into a fixed-length embedding.
3. **Fusion head**: concatenates both embeddings → FC → sigmoid → P(spike).

The model outputs raw logits; use ``torch.sigmoid`` or ``BCEWithLogitsLoss``.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GRUBranch(nn.Module):
    """Encode sensor time-series via a multi-layer GRU."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_dim)

        Returns
        -------
        h : (batch, hidden_dim)
        """
        _, h_n = self.gru(x)          # h_n: (num_layers, batch, hidden)
        h_last = h_n[-1]              # last layer
        return self.layer_norm(h_last)


class MLPBranch(nn.Module):
    """Encode static EHR features via a small MLP."""

    def __init__(self, input_dim: int = 22, hidden_dim: int = 64,
                 embed_dim: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, input_dim)

        Returns
        -------
        e : (batch, embed_dim)
        """
        return self.net(x)


class FusionModel(nn.Module):
    """Two-branch fusion model for hyperglycemic spike prediction.

    Concatenates GRU time-series embedding and MLP static embedding,
    then feeds through a classification head.
    """

    def __init__(
        self,
        gru_input_dim: int = 5,
        gru_hidden_dim: int = 64,
        gru_num_layers: int = 2,
        gru_dropout: float = 0.3,
        static_input_dim: int = 22,
        static_hidden_dim: int = 64,
        static_embed_dim: int = 32,
        fusion_hidden_dim: int = 64,
        fusion_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.gru_branch = GRUBranch(gru_input_dim, gru_hidden_dim,
                                     gru_num_layers, gru_dropout)
        self.mlp_branch = MLPBranch(static_input_dim, static_hidden_dim,
                                     static_embed_dim, fusion_dropout)

        concat_dim = gru_hidden_dim + static_embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, 1),  # raw logit
        )

    def forward(self, time_series: torch.Tensor,
                static: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        time_series : (batch, seq_len, gru_input_dim)
        static      : (batch, static_input_dim)

        Returns
        -------
        logit : (batch, 1) – apply sigmoid for probability
        """
        ts_embed = self.gru_branch(time_series)
        st_embed = self.mlp_branch(static)
        fused = torch.cat([ts_embed, st_embed], dim=1)
        return self.classifier(fused)

    def get_embeddings(self, time_series: torch.Tensor,
                       static: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return both branch embeddings (useful for explainability)."""
        ts_embed = self.gru_branch(time_series)
        st_embed = self.mlp_branch(static)
        return ts_embed, st_embed


class SensorOnlyModel(nn.Module):
    """Ablation: GRU branch only (no static features)."""

    def __init__(self, gru_input_dim: int = 5, gru_hidden_dim: int = 64,
                 gru_num_layers: int = 2, gru_dropout: float = 0.3,
                 fusion_hidden_dim: int = 64) -> None:
        super().__init__()
        self.gru_branch = GRUBranch(gru_input_dim, gru_hidden_dim,
                                     gru_num_layers, gru_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(gru_dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, time_series: torch.Tensor,
                static: torch.Tensor = None) -> torch.Tensor:
        ts_embed = self.gru_branch(time_series)
        return self.classifier(ts_embed)


class StaticOnlyModel(nn.Module):
    """Ablation: MLP branch only (no sensor time-series)."""

    def __init__(self, static_input_dim: int = 22, static_hidden_dim: int = 64,
                 static_embed_dim: int = 32, fusion_hidden_dim: int = 64,
                 fusion_dropout: float = 0.3) -> None:
        super().__init__()
        self.mlp_branch = MLPBranch(static_input_dim, static_hidden_dim,
                                     static_embed_dim, fusion_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(static_embed_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, time_series: torch.Tensor = None,
                static: torch.Tensor = None) -> torch.Tensor:
        st_embed = self.mlp_branch(static)
        return self.classifier(st_embed)


def build_model_from_config(config: Dict, model_type: str = "fusion") -> nn.Module:
    """Factory function to build a model from the config dictionary.

    Parameters
    ----------
    config : dict
        Full config (or ``config["model"]`` section).
    model_type : str
        One of ``"fusion"``, ``"sensor_only"``, ``"static_only"``.
    """
    m = config if "gru_input_dim" in config else config.get("model", config)

    if model_type == "fusion":
        return FusionModel(
            gru_input_dim=m["gru_input_dim"],
            gru_hidden_dim=m["gru_hidden_dim"],
            gru_num_layers=m["gru_num_layers"],
            gru_dropout=m["gru_dropout"],
            static_input_dim=m["static_input_dim"],
            static_hidden_dim=m["static_hidden_dim"],
            static_embed_dim=m["static_embed_dim"],
            fusion_hidden_dim=m["fusion_hidden_dim"],
            fusion_dropout=m["fusion_dropout"],
        )
    elif model_type == "sensor_only":
        return SensorOnlyModel(
            gru_input_dim=m["gru_input_dim"],
            gru_hidden_dim=m["gru_hidden_dim"],
            gru_num_layers=m["gru_num_layers"],
            gru_dropout=m["gru_dropout"],
            fusion_hidden_dim=m["fusion_hidden_dim"],
        )
    elif model_type == "static_only":
        return StaticOnlyModel(
            static_input_dim=m["static_input_dim"],
            static_hidden_dim=m["static_hidden_dim"],
            static_embed_dim=m["static_embed_dim"],
            fusion_hidden_dim=m["fusion_hidden_dim"],
            fusion_dropout=m["fusion_dropout"],
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
