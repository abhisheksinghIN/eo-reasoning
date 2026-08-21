"""Numerical utilities for GeoFM embeddings."""

from __future__ import annotations

import numpy as np


def cosine_similarity(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def cosine_distance(a, b) -> float:
    similarity = cosine_similarity(a, b)
    return float("nan") if np.isnan(similarity) else 1.0 - similarity


def l2_change(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(b - a))


def summarize_temporal_embeddings(embeddings) -> dict:
    z = np.asarray(embeddings, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 2:
        raise ValueError("Expected embeddings with shape [T,D], T>=2.")

    cosine_distances = [cosine_distance(z[i - 1], z[i]) for i in range(1, len(z))]
    l2_changes = [l2_change(z[i - 1], z[i]) for i in range(1, len(z))]

    return {
        "n_frames": int(z.shape[0]),
        "embedding_dim": int(z.shape[1]),
        "cosine_distance_consecutive": [float(v) for v in cosine_distances],
        "l2_change_consecutive": [float(v) for v in l2_changes],
        "start_end_cosine_distance": cosine_distance(z[0], z[-1]),
        "start_end_l2_change": l2_change(z[0], z[-1]),
        "mean_cosine_distance": float(np.nanmean(cosine_distances)),
        "mean_l2_change": float(np.nanmean(l2_changes)),
    }
