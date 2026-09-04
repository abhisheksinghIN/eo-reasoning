"""Spatial change analysis for Prithvi representations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from affine import Affine

from data.preprocessing import (
    read_process_tiff,
    valid_surface_mask,
)


NODATA = -9999.0


def cosine_change_map(
    spatial_embeddings: torch.Tensor,
) -> np.ndarray:
    """
    Calculate patch-wise cosine distance between
    the first and final Prithvi observations.

    Input:
        [B, T, H, W, D]

    Output:
        [H, W]
    """

    start = spatial_embeddings[0, 0]
    end = spatial_embeddings[0, -1]

    similarity = F.cosine_similarity(
        start,
        end,
        dim=-1,
    )

    distance = 1.0 - similarity

    return (
        distance
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def _common_patch_mask(
    start_path: str | Path,
    end_path: str | Path,
    patch_size: int = 16,
    min_valid_fraction: float = 0.5,
) -> np.ndarray:
    """Determine which 16×16 Prithvi patches are sufficiently valid."""

    start = read_process_tiff(start_path)
    end = read_process_tiff(end_path)

    valid = (
        valid_surface_mask(start)
        & valid_surface_mask(end)
    )

    h, w = valid.shape

    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(
            "Raster dimensions must be divisible by "
            f"patch size {patch_size}."
        )

    patch_fraction = (
        valid
        .reshape(
            h // patch_size,
            patch_size,
            w // patch_size,
            patch_size,
        )
        .mean(axis=(1, 3))
    )

    return patch_fraction >= min_valid_fraction


def write_prithvi_change_geotiff(
    change: np.ndarray,
    start_path: str | Path,
    end_path: str | Path,
    output_path: str | Path,
) -> str:
    """Write native 14×14 Prithvi latent-change GeoTIFF."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    valid_patches = _common_patch_mask(
        start_path,
        end_path,
    )

    result = np.full(
        change.shape,
        NODATA,
        dtype=np.float32,
    )

    result[valid_patches] = change[valid_patches]

    with rasterio.open(start_path) as src:

        profile = src.profile.copy()

        x_scale = src.width / change.shape[1]
        y_scale = src.height / change.shape[0]

        transform = (
            src.transform
            * Affine.scale(
                x_scale,
                y_scale,
            )
        )

    profile.update(
        count=1,
        width=change.shape[1],
        height=change.shape[0],
        dtype="float32",
        nodata=NODATA,
        transform=transform,
        compress="deflate",
    )

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:

        dst.write(result, 1)

        dst.set_band_description(
            1,
            "Prithvi cosine distance",
        )

        dst.update_tags(
            product="Prithvi latent representation change",
            metric="cosine_distance",
            interpretation=(
                "Higher values indicate stronger change "
                "in learned representation."
            ),
        )

    return str(output_path)
