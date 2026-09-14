"""JEPA-style temporal prediction in GeoFM latent space."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TemporalJEPAPredictor(nn.Module):
    """
    Predict the latent representation of the target date
    from earlier GeoFM representations.

    Example:
        z(t-2), z(t-1) -> z_hat(t)

    The target z(t) is stop-gradient in the JEPA loss.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
        )

        self.predictor = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(
                embedding_dim,
                embedding_dim,
            ),
            nn.GELU(),
            nn.Linear(
                embedding_dim,
                embedding_dim,
            ),
        )

    def forward(
        self,
        context_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        context_embeddings:
            [B, T_context, D]

        returns:
            [B, D]
        """

        output, _ = self.gru(
            context_embeddings
        )

        context_state = output[:, -1]

        return self.predictor(
            context_state
        )


def jepa_latent_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Cosine-distance JEPA loss.

    Target is detached so gradients do not flow
    through the target representation.
    """

    predicted = F.normalize(
        predicted,
        dim=-1,
    )

    target = F.normalize(
        target.detach(),
        dim=-1,
    )

    similarity = (
        predicted * target
    ).sum(dim=-1)

    return (
        1.0 - similarity
    ).mean()