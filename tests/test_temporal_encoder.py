import torch

from models.temporal_encoder import TemporalEncoder


def test_temporal_pool_with_cls():
    pooler = TemporalEncoder(num_frames=3, image_size=32, spatial_patch_size=16)
    hidden = torch.arange(13 * 5, dtype=torch.float32).reshape(1, 13, 5)
    result = pooler.pool(hidden)
    assert result.shape == (1, 3, 5)


def test_temporal_pool_without_cls():
    pooler = TemporalEncoder(num_frames=3, image_size=32, spatial_patch_size=16)
    hidden = torch.randn(2, 12, 7)
    result = pooler.pool(hidden)
    assert result.shape == (2, 3, 7)
