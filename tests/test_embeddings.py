import numpy as np

from models.embeddings import cosine_distance, l2_change, summarize_temporal_embeddings


def test_cosine_distance_identical():
    x = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_distance(x, x)) < 1e-8


def test_l2_change():
    assert abs(l2_change([0, 0], [3, 4]) - 5.0) < 1e-8


def test_summary():
    z = np.array([[1, 0], [0.9, 0.1], [0.8, 0.2]], dtype=float)
    result = summarize_temporal_embeddings(z)
    assert result["n_frames"] == 3
    assert result["embedding_dim"] == 2
    assert len(result["cosine_distance_consecutive"]) == 2
