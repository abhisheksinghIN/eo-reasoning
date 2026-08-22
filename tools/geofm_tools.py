"""Prithvi tool boundary."""

from __future__ import annotations

from typing import List

from data.preprocessing import prepare_prithvi_tensor
from models.embeddings import summarize_temporal_embeddings
from models.prithvi import DEFAULT_MODEL_ID, PrithviConfig, PrithviModel

_MODEL_CACHE = {}


def _get_model(model_id: str):
    if model_id not in _MODEL_CACHE:
        _MODEL_CACHE[model_id] = PrithviModel(PrithviConfig(model_id=model_id))
    return _MODEL_CACHE[model_id]


def run_prithvi_temporal(paths: List[str], model_id: str = DEFAULT_MODEL_ID) -> dict:
    """Run Prithvi on exactly three prepared Sentinel-2 TIFFs."""
    tensor, _ = prepare_prithvi_tensor(paths)
    model = _get_model(model_id)
    temporal = model.temporal_embeddings(tensor)
    z = temporal[0].detach().cpu().numpy()

    return {
        "model": model_id,
        "input_shape": list(tensor.shape),
        "temporal_embeddings": z.tolist(),
        "summary": summarize_temporal_embeddings(z),
    }
