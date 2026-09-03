"""Generate georeferenced NDVI temporal-change products."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from data.preprocessing import (
    read_process_tiff,
    valid_surface_mask,
)


NODATA = -9999.0


def _safe_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    out = np.full(
        numerator.shape,
        np.nan,
        dtype=np.float32,
    )

    np.divide(
        numerator,
        denominator,
        out=out,
        where=np.abs(denominator) > 1e-8,
    )

    return out


def ndvi_map(frame: np.ndarray) -> np.ndarray:
    """
    Calculate pixel-wise Sentinel-2 NDVI.

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
    """

    frame = np.asarray(frame, dtype=np.float32)

    red = frame[2] / 10000.0
    nir = frame[6] / 10000.0

    return _safe_ratio(
        nir - red,
        nir + red,
    )


def write_ndvi_change_geotiff(
    start_path: str | Path,
    end_path: str | Path,
    output_path: str | Path,
) -> str:
    """
    Write a GeoTIFF containing:

        delta_NDVI = NDVI(end) - NDVI(start)

    Only pixels valid on both dates are included.
    """

    start = read_process_tiff(start_path)
    end = read_process_tiff(end_path)

    start_ndvi = ndvi_map(start)
    end_ndvi = ndvi_map(end)

    common_valid = (
        valid_surface_mask(start)
        & valid_surface_mask(end)
        & np.isfinite(start_ndvi)
        & np.isfinite(end_ndvi)
    )

    change = np.full(
        start_ndvi.shape,
        NODATA,
        dtype=np.float32,
    )

    change[common_valid] = (
        end_ndvi[common_valid]
        - start_ndvi[common_valid]
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(start_path) as src:
        profile = src.profile.copy()

    profile.update(
        count=1,
        dtype="float32",
        nodata=NODATA,
        compress="deflate",
    )

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:

        dst.write(change, 1)

        dst.set_band_description(
            1,
            "delta_ndvi",
        )

        dst.update_tags(
            product="NDVI temporal change",
            definition="NDVI(end) - NDVI(start)",
        )

    return str(output_path)
