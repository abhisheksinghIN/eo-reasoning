import numpy as np


def cosine_distance(a, b):
    """
    Compute cosine distance between two embeddings.
    """

    a = np.asarray(a)
    b = np.asarray(b)

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return np.nan

    similarity = np.dot(a, b) / denominator

    return 1.0 - similarity


def embedding_change(
    embedding_t1,
    embedding_t2,
):
    """
    Compute temporal representation change.
    """

    embedding_t1 = np.asarray(
        embedding_t1
    )

    embedding_t2 = np.asarray(
        embedding_t2
    )

    return np.array([
        cosine_distance(a, b)
        for a, b in zip(
            embedding_t1,
            embedding_t2
        )
    ])
