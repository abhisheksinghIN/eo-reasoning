import numpy as np

from sklearn.decomposition import PCA


def reduce_embeddings(
    embeddings,
    n_components=3,
):
    """
    Reduce GeoFM embeddings for visualization.
    """

    embeddings = np.asarray(
        embeddings
    )

    pca = PCA(
        n_components=n_components
    )

    reduced = pca.fit_transform(
        embeddings
    )

    return {
        "embedding": reduced,
        "explained_variance": (
            pca.explained_variance_ratio_
        ),
        "model": pca,
    }
