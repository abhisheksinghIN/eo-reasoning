"""PCA helper for visualizing temporal embeddings."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def reduce_embeddings(embeddings, n_components: int = 2) -> dict:
    z = np.asarray(embeddings, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("embeddings must have shape [N,D].")

    n_components = min(n_components, z.shape[0], z.shape[1])
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(z)
    return {
        "embedding": reduced,
        "explained_variance": pca.explained_variance_ratio_,
        "model": pca,
    }
