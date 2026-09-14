"""TerraMind GeoFM wrapper for soil-moisture reasoning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from terratorch import BACKBONE_REGISTRY


S2_SUBSET_BANDS = [
    "BLUE",
    "GREEN",
    "RED",
    "NIR_NARROW",
    "SWIR_1",
    "SWIR_2",
]


@dataclass
class TerraMindConfig:
    model_name: str = "terramind_v1_base"
    use_s2: bool = False
    freeze_backbone: bool = True
    device: str | None = None

    # TerraMind v1 base embedding dimension.
    embedding_dim: int = 768


class TerraMindEncoder(nn.Module):
    def __init__(
        self,
        config: TerraMindConfig | None = None,
    ):
        super().__init__()

        self.config = config or TerraMindConfig()

        self.device_name = (
            self.config.device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        modalities = ["S1GRD"]

        kwargs = {
            "pretrained": True,
            "modalities": modalities,
            "merge_method": "mean",
        }

        if self.config.use_s2:
            modalities.append("S2L2A")

            kwargs["modalities"] = modalities

            kwargs["bands"] = {
                "S2L2A": S2_SUBSET_BANDS,
            }

        print(
            f"Loading {self.config.model_name} "
            f"modalities={modalities} "
            f"device={self.device_name}"
        )

        self.backbone = BACKBONE_REGISTRY.build(
            self.config.model_name,
            **kwargs,
        )

        self.backbone = self.backbone.to(
            self.device_name
        )

        if self.config.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

            self.backbone.eval()
# -------------------------------------
    def train(
        self,
        mode: bool = True,
    ):
        """
        Keep the pretrained TerraMind backbone in eval mode when frozen.
    
        The surrounding wrapper may be placed in training mode so that
        downstream JEPA / regression / physics components can train,
        while the frozen GeoFM remains deterministic.
        """
    
        super().train(mode)
    
        if self.config.freeze_backbone:
            self.backbone.eval()
    
        return self
# ------------------------------------
    def _encode_one(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns one global embedding per observation.

        s1:
            [B, 2, 224, 224]

        optional s2:
            [B, 6, 224, 224]

        output:
            [B, 768]
        """

        s1 = s1.to(
            self.device_name,
            dtype=torch.float32,
        )

        inputs = {
            "S1GRD": s1,
        }

        if self.config.use_s2:
            if s2 is None:
                raise ValueError(
                    "This TerraMind instance was configured "
                    "with use_s2=True but no S2 tensor was supplied."
                )

            inputs["S2L2A"] = s2.to(
                self.device_name,
                dtype=torch.float32,
            )

        if self.config.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(inputs)
        else:
            features = self.backbone(inputs)

        if not isinstance(
            features,
            (list, tuple),
        ):
            raise TypeError(
                "Expected TerraMind backbone to return "
                "layer feature tensors."
            )

        hidden = features[-1]

        if hidden.ndim != 3:
            raise ValueError(
                f"Expected [B,N,D], got {hidden.shape}."
            )

        # TerraMind does not use a CLS token here.
        # Global mean-pooling over patch tokens.
        return hidden.mean(dim=1)

    def forward(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._encode_one(
            s1=s1,
            s2=s2,
        )

    def encode_sequence(
        self,
        s1_sequence: torch.Tensor,
        s2_sequence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        s1_sequence:
            [B, T, 2, H, W]

        output:
            [B, T, D]
        """

        if s1_sequence.ndim != 5:
            raise ValueError(
                "Expected S1 sequence [B,T,C,H,W]."
            )

        embeddings = []

        for t in range(
            s1_sequence.shape[1]
        ):
            s2_t = (
                s2_sequence[:, t]
                if s2_sequence is not None
                else None
            )

            z_t = self._encode_one(
                s1=s1_sequence[:, t],
                s2=s2_t,
            )

            embeddings.append(z_t)

        return torch.stack(
            embeddings,
            dim=1,
        )