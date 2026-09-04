"""Prithvi tool boundary."""

from __future__ import annotations

from typing import List

from data.preprocessing import prepare_prithvi_tensor
from models.embeddings import summarize_temporal_embeddings
#from models.prithvi import DEFAULT_MODEL_ID, PrithviConfig, PrithviModel
from analysis.prithvi_change import cosine_change_map


from models.prithvi import (
    DEFAULT_MODEL_NAME,
    PrithviConfig,
    PrithviModel,
)

_MODEL_CACHE = {}


def _get_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = PrithviModel(
            PrithviConfig(model_name=model_name)
        )
    return _MODEL_CACHE[model_name]


def run_prithvi_temporal(
    paths,
    model_name=DEFAULT_MODEL_NAME,
):
    tensor, _ = prepare_prithvi_tensor(paths)

    model = _get_model(model_name)

    #temporal = model.temporal_embeddings(tensor)
    # ---------------
    hidden = model.encode(tensor)
    
    temporal = model.temporal_pooler.pool(hidden)
    
    spatial = (
        model.spatial_temporal_embeddings_from_hidden(
            hidden
        )
    )
    
    spatial_change = cosine_change_map(spatial)    
    # ---------------

    z = temporal[0].detach().cpu().numpy()

#    return {
#        "model": model_name,
#        "input_shape": list(tensor.shape),
#        "temporal_embeddings": z.tolist(),
#        "summary": summarize_temporal_embeddings(z),
#    }
    return {
        "model": model_name,
        "input_shape": list(tensor.shape),
        "temporal_embeddings": z.tolist(),
        "summary": summarize_temporal_embeddings(z),
    
        # Private intermediate used by pipeline.py
        "_spatial_change_map": spatial_change.tolist(),
    }