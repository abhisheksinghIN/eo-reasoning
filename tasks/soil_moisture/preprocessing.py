"""Preprocessing for TerraMind soil-moisture experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F


# Official TerraMind S1GRD normalization statistics.
# Input order: VV, VH
S1GRD_MEAN = torch.tensor(
    [-12.599, -20.293],
    dtype=torch.float32,
)

S1GRD_STD = torch.tensor(
    [5.195, 5.890],
    dtype=torch.float32,
)


# Optional Sentinel-2 subset already available in your current
# Sentinel-2 stack:
#
# B02, B03, B04, B8A, B11, B12
#
# TerraMind names:
# BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2

S2_SUBSET_BANDS = [
    "BLUE",
    "GREEN",
    "RED",
    "NIR_NARROW",
    "SWIR_1",
    "SWIR_2",
]

S2_SUBSET_MEAN = torch.tensor(
    [
        1503.317,   # BLUE
        1718.197,   # GREEN
        1853.910,   # RED
        3132.220,   # NIR_NARROW
        2424.884,   # SWIR_1
        1857.648,   # SWIR_2
    ],
    dtype=torch.float32,
)

S2_SUBSET_STD = torch.tensor(
    [
        2141.107,
        2038.973,
        2134.138,
        1753.829,
        1434.261,
        1334.311,
    ],
    dtype=torch.float32,
)


def prepare_s1_terramind(
    path: str | Path,
    image_size: int = 224,
) -> torch.Tensor:
    """
    Convert current S1 GeoTIFF to TerraMind S1GRD input.

    Current raster:
        0 = VV linear power
        1 = VH linear power
        2 = local incidence angle
        3 = dataMask

    Returns:
        [2, H, W] standardized tensor
    """

    with rasterio.open(path) as src:
        x = src.read()

    if x.shape[0] != 4:
        raise ValueError(
            f"Expected 4 S1 bands, got {x.shape}."
        )

    vv = torch.from_numpy(
        x[0].astype(np.float32)
    )

    vh = torch.from_numpy(
        x[1].astype(np.float32)
    )

    valid = torch.from_numpy(
        x[3] > 0
    )

    # TerraMind S1GRD normalization is in dB.
    eps = 1e-6

    vv_db = 10.0 * torch.log10(
        torch.clamp(vv, min=eps)
    )

    vh_db = 10.0 * torch.log10(
        torch.clamp(vh, min=eps)
    )

    image = torch.stack(
        [vv_db, vh_db],
        dim=0,
    )

    mean = S1GRD_MEAN[:, None, None]
    std = S1GRD_STD[:, None, None]

    image = (image - mean) / std

    # Zero after standardization corresponds approximately
    # to TerraMind's training mean.
    image[:, ~valid] = 0.0

    if image.shape[-2:] != (
        image_size,
        image_size,
    ):
        image = F.interpolate(
            image.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return image


def prepare_s2_terramind(
    path: str | Path,
    image_size: int = 224,
) -> torch.Tensor:
    """
    Prepare optional S2 subset from the existing 9-band stack.

    Current stack:
        0 B02
        1 B03
        2 B04
        3 B8A
        4 B11
        5 B12
        6 B08
        7 SCL
        8 dataMask

    TerraMind receives the first six channels.
    """

    with rasterio.open(path) as src:
        x = src.read()

    if x.shape[0] != 9:
        raise ValueError(
            f"Expected existing 9-band S2 stack, got {x.shape}."
        )

    image = torch.from_numpy(
        x[:6].astype(np.float32)
    )

    valid = torch.from_numpy(
        x[8] > 0
    )

    mean = S2_SUBSET_MEAN[:, None, None]
    std = S2_SUBSET_STD[:, None, None]

    image = (image - mean) / std
    image[:, ~valid] = 0.0

    if image.shape[-2:] != (
        image_size,
        image_size,
    ):
        image = F.interpolate(
            image.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return image