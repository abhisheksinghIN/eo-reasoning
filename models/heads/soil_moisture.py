"""Soil-moisture prediction head."""

from __future__ import annotations

import torch
from torch import nn


class SoilMoistureHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.LayerNorm(embedding_dim),

            nn.Linear(
                embedding_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(0.1),

            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        raw = self.network(
            embedding
        ).squeeze(-1)

        # CLMS SSM is percent saturation.
        # This guarantees 0 <= SSM <= 100.
        return 100.0 * torch.sigmoid(raw)