"""Prithvi-EO-1.0-100M encoder wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from models.temporal_encoder import TemporalEncoder

DEFAULT_MODEL_ID = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M"


@dataclass
class PrithviConfig:
    model_id: str = DEFAULT_MODEL_ID
    device: Optional[str] = None


class PrithviModel:
    def __init__(self, config: Optional[PrithviConfig] = None):
        self.config = config or PrithviConfig()
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        kwargs = {}
        if self.device == "cuda":
            kwargs["device_map"] = "auto"

        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(self.config.model_id, **kwargs)
        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        self.temporal_pooler = TemporalEncoder(num_frames=3, image_size=224, spatial_patch_size=16)

    @torch.inference_mode()
    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = pixel_values.to(self.device)
        output = self.model(pixel_values=x)

        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if isinstance(output, (tuple, list)) and output:
            return output[0]
        if torch.is_tensor(output):
            return output

        raise TypeError(f"Could not extract hidden states from output type {type(output)!r}")

    @torch.inference_mode()
    def temporal_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(pixel_values)
        return self.temporal_pooler.pool(hidden)

    @torch.inference_mode()
    def global_embedding(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.temporal_embeddings(pixel_values).mean(dim=1)
