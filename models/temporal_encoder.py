"""Pool spatio-temporal Prithvi tokens into one embedding per frame."""

from __future__ import annotations

import torch


class TemporalEncoder:
    """Convert transformer tokens [B,L,D] to temporal embeddings [B,T,D]."""

    def __init__(self, num_frames: int = 3, image_size: int = 224, spatial_patch_size: int = 16):
        self.num_frames = num_frames
        self.image_size = image_size
        self.spatial_patch_size = spatial_patch_size

    @property
    def spatial_tokens_per_frame(self) -> int:
        side = self.image_size // self.spatial_patch_size
        return side * side

    def pool(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected hidden states [B,L,D], received {hidden_states.shape}.")

        expected = self.num_frames * self.spatial_tokens_per_frame
        n_tokens = hidden_states.shape[1]

        if n_tokens == expected + 1:
            tokens = hidden_states[:, 1:, :]
        elif n_tokens == expected:
            tokens = hidden_states
        else:
            raise ValueError(
                f"Unexpected token count {n_tokens}; expected {expected} or {expected + 1}. "
                "Check the checkpoint/input configuration."
            )

        batch, _, dim = tokens.shape
        tokens = tokens.reshape(batch, self.num_frames, self.spatial_tokens_per_frame, dim)
        return tokens.mean(dim=2)
