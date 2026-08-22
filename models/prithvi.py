"""Prithvi-EO-1.0-100M encoder wrapper using TerraTorch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from terratorch import BACKBONE_REGISTRY

from models.temporal_encoder import TemporalEncoder


DEFAULT_MODEL_NAME = "prithvi_eo_v1_100"


@dataclass
class PrithviConfig:
    model_name: str = DEFAULT_MODEL_NAME
    device: Optional[str] = None
    num_frames: int = 3
    image_size: int = 224
    spatial_patch_size: int = 16


class PrithviModel:
    def __init__(self, config: Optional[PrithviConfig] = None):
        self.config = config or PrithviConfig()

        self.device = self.config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(
            f"Loading {self.config.model_name} "
            f"on device={self.device}"
        )

        self.model = BACKBONE_REGISTRY.build(
            self.config.model_name,
            pretrained=True,
            num_frames=self.config.num_frames,
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        self.temporal_pooler = TemporalEncoder(
            num_frames=self.config.num_frames,
            image_size=self.config.image_size,
            spatial_patch_size=self.config.spatial_patch_size,
        )

    @torch.inference_mode()
    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract the final transformer token representation.

        Expected input:
            pixel_values: [B, C, T, H, W]

        For this MVP:
            [B, 6, 3, 224, 224]

        Returns:
            Tensor [B, L, D]
        """

        if pixel_values.ndim != 5:
            raise ValueError(
                "Prithvi input must have shape [B, C, T, H, W]. "
                f"Received {tuple(pixel_values.shape)}"
            )

        if pixel_values.shape[1] != 6:
            raise ValueError(
                "Prithvi-EO v1 expects 6 spectral channels. "
                f"Received {pixel_values.shape[1]}."
            )

        if pixel_values.shape[2] != self.config.num_frames:
            raise ValueError(
                f"Expected {self.config.num_frames} temporal frames, "
                f"received {pixel_values.shape[2]}."
            )

        x = pixel_values.to(
            self.device,
            dtype=torch.float32,
        )

        # TerraTorch's Prithvi backbone exposes forward_features(),
        # which returns one token tensor per transformer layer.
        features = self.model.forward_features(x)

        if not isinstance(features, (list, tuple)):
            raise TypeError(
                "Expected TerraTorch Prithvi forward_features() "
                f"to return a list/tuple, got {type(features)!r}."
            )

        if len(features) == 0:
            raise RuntimeError(
                "Prithvi returned no transformer features."
            )

        hidden = features[-1]

        if not torch.is_tensor(hidden):
            raise TypeError(
                "Expected final Prithvi feature to be a tensor, "
                f"got {type(hidden)!r}."
            )

        if hidden.ndim != 3:
            raise ValueError(
                "Expected final Prithvi feature shape [B, L, D], "
                f"got {tuple(hidden.shape)}."
            )

        return hidden

    @torch.inference_mode()
    def temporal_embeddings(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert Prithvi tokens to one embedding per temporal frame.

        Returns:
            [B, T, D]
        """

        hidden = self.encode(pixel_values)

        return self.temporal_pooler.pool(hidden)

    @torch.inference_mode()
    def global_embedding(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return one global embedding for the temporal sequence.

        Returns:
            [B, D]
        """

        temporal = self.temporal_embeddings(pixel_values)

        return temporal.mean(dim=1)
