"""Validation and QA utilities for soil-moisture collocation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio


def read_raster(path: str | Path) -> np.ndarray:
    """Read all raster bands as [C, H, W]."""

    with rasterio.open(path) as src:
        return src.read()


def finite_stats(
    values: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Return basic statistics over finite valid values."""

    values = np.asarray(values, dtype=np.float32)

    valid = np.isfinite(values)

    if mask is not None:
        valid &= mask

    x = values[valid]

    if x.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "minimum": float("nan"),
            "maximum": float("nan"),
            "valid_fraction": 0.0,
        }

    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "minimum": float(np.min(x)),
        "maximum": float(np.max(x)),
        "valid_fraction": float(np.mean(valid)),
    }


def summarize_sentinel1(
    path: str | Path,
) -> dict[str, Any]:
    """
    Expected S1 layout for the first MVP:

        0 = VV
        1 = VH
        2 = local incidence angle
        3 = dataMask
    """

    array = read_raster(path)

    if array.shape[0] != 4:
        raise ValueError(
            "Expected Sentinel-1 raster with 4 bands "
            "[VV, VH, localIncidenceAngle, dataMask], "
            f"received shape {array.shape}."
        )

    vv = array[0].astype(np.float32)
    vh = array[1].astype(np.float32)
    angle = array[2].astype(np.float32)
    data_mask = array[3] > 0

    # Sentinel Hub linear backscatter should not be negative.
    valid = (
        data_mask
        & np.isfinite(vv)
        & np.isfinite(vh)
        & np.isfinite(angle)
        & (vv >= 0)
        & (vh >= 0)
        & (angle >= 0)
        & (angle <= 90)
    )

    return {
        "vv": finite_stats(vv, valid),
        "vh": finite_stats(vh, valid),
        "incidence_angle": finite_stats(angle, valid),
        "valid_fraction": float(np.mean(valid)),
    }


def summarize_clms_ssm(
    path: str | Path,
) -> dict[str, Any]:
    """
    Expected CLMS layout:

        0 = SSM
        1 = SSM_NOISE
        2 = dataMask
    """

    array = read_raster(path)

    if array.shape[0] != 3:
        raise ValueError(
            "Expected CLMS SSM raster with 3 bands "
            "[SSM, SSM_NOISE, dataMask], "
            f"received shape {array.shape}."
        )

    ssm = array[0].astype(np.float32)
    noise = array[1].astype(np.float32)
    data_mask = array[2] > 0

    valid_ssm = (
        data_mask
        & np.isfinite(ssm)
        & (ssm >= 0)
        & (ssm <= 100)
    )

    valid_noise = (
        data_mask
        & np.isfinite(noise)
        & (noise >= 0)
    )

    return {
        "ssm_percent_saturation": finite_stats(
            ssm,
            valid_ssm,
        ),
        "ssm_noise": finite_stats(
            noise,
            valid_noise,
        ),
        "valid_fraction": float(
            np.mean(valid_ssm)
        ),
    }


def validate_collocation(
    sentinel1_path: str | Path,
    ssm_path: str | Path,
) -> dict[str, Any]:
    """Run QA on both products."""

    s1 = summarize_sentinel1(sentinel1_path)
    ssm = summarize_clms_ssm(ssm_path)

    issues: list[str] = []

    if s1["valid_fraction"] < 0.5:
        issues.append(
            "Sentinel-1 valid fraction is below 50%."
        )

    if ssm["valid_fraction"] < 0.5:
        issues.append(
            "CLMS SSM valid fraction is below 50%."
        )

    status = "PASS" if not issues else "REVIEW"

    return {
        "status": status,
        "issues": issues,
        "sentinel1": s1,
        "reference": ssm,
    }