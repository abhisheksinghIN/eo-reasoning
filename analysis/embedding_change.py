"""Temporal change metrics for GeoFM embeddings."""

from __future__ import annotations

import numpy as np

from models.embeddings import cosine_distance, l2_change


def embedding_change(embeddings) -> dict:
    z = np.asarray(embeddings, dtype=np.float64)
    if z.ndim != 2 or len(z) < 2:
        raise ValueError("Expected [T,D] embeddings with T>=2.")

    consecutive = []
    for i in range(1, len(z)):
        consecutive.append({
            "from_index": i - 1,
            "to_index": i,
            "cosine_distance": cosine_distance(z[i - 1], z[i]),
            "l2_change": l2_change(z[i - 1], z[i]),
        })

    return {
        "consecutive": consecutive,
        "start_end": {
            "cosine_distance": cosine_distance(z[0], z[-1]),
            "l2_change": l2_change(z[0], z[-1]),
        },
    }
