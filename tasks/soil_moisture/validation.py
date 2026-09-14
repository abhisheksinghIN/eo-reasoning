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


#def finite_stats(
#    values: np.ndarray,
#    mask: np.ndarray | None = None,
#) -> dict[str, float]:
#    """Return basic statistics over finite valid values."""
#
#    values = np.asarray(values, dtype=np.float32)
#
#    valid = np.isfinite(values)
#
#    if mask is not None:
#        valid &= mask
#
#    x = values[valid]
#
#    if x.size == 0:
#        return {
#            "mean": float("nan"),
#            "std": float("nan"),
#            "minimum": float("nan"),
#            "maximum": float("nan"),
#            "valid_fraction": 0.0,
#        }
#
#    return {
#        "mean": float(np.mean(x)),
#        "std": float(np.std(x)),
#        "minimum": float(np.min(x)),
#        "maximum": float(np.max(x)),
#        "valid_fraction": float(np.mean(valid)),
#    }

def finite_stats(
    values: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute robust descriptive statistics for valid finite pixels.
    """

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    valid = np.isfinite(
        values
    )

    if mask is not None:
        mask = np.asarray(
            mask,
            dtype=bool,
        )

        if mask.shape != values.shape:
            raise ValueError(
                "Mask and values must have the same shape: "
                f"{mask.shape} != {values.shape}"
            )

        valid &= mask

    x = values[
        valid
    ]

    if x.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
            "minimum": float("nan"),
            "maximum": float("nan"),
            "valid_fraction": 0.0,
        }

    return {
        "mean": float(
            np.mean(x)
        ),

        "median": float(
            np.median(x)
        ),

        "std": float(
            np.std(x)
        ),

        "p05": float(
            np.percentile(
                x,
                5,
            )
        ),

        "p25": float(
            np.percentile(
                x,
                25,
            )
        ),

        "p75": float(
            np.percentile(
                x,
                75,
            )
        ),

        "p95": float(
            np.percentile(
                x,
                95,
            )
        ),

        "minimum": float(
            np.min(x)
        ),

        "maximum": float(
            np.max(x)
        ),

        "valid_fraction": float(
            np.mean(valid)
        ),
    }    

#def summarize_sentinel1(
#    path: str | Path,
#) -> dict[str, Any]:
#    """
#    Expected S1 layout for the first MVP:
#
#        0 = VV
#        1 = VH
#        2 = local incidence angle
#        3 = dataMask
#    """
#
#    array = read_raster(path)
#
#    if array.shape[0] != 4:
#        raise ValueError(
#            "Expected Sentinel-1 raster with 4 bands "
#            "[VV, VH, localIncidenceAngle, dataMask], "
#            f"received shape {array.shape}."
#        )
#
#    vv = array[0].astype(np.float32)
#    vh = array[1].astype(np.float32)
#    angle = array[2].astype(np.float32)
#    data_mask = array[3] > 0
#
#    # Sentinel Hub linear backscatter should not be negative.
#    valid = (
#        data_mask
#        & np.isfinite(vv)
#        & np.isfinite(vh)
#        & np.isfinite(angle)
#        & (vv >= 0)
#        & (vh >= 0)
#        & (angle >= 0)
#        & (angle <= 90)
#    )
#
#    return {
#        "vv": finite_stats(vv, valid),
#        "vh": finite_stats(vh, valid),
#        "incidence_angle": finite_stats(angle, valid),
#        "valid_fraction": float(np.mean(valid)),
#    }
def summarize_sentinel1(
    path: str | Path,
) -> dict:
    """
    Summarize a Sentinel-1 GeoTIFF used by the soil-moisture pipeline.

    Expected raster layout
    ----------------------
    Band 1:
        VV backscatter in linear power.

    Band 2:
        VH backscatter in linear power.

    Band 3:
        Local incidence angle in degrees.

    Band 4:
        dataMask, where values > 0 indicate valid pixels.

    Returns
    -------
    dict
        Robust statistics for:

        - VV linear power
        - VH linear power
        - VV in dB
        - VH in dB
        - VH/VV linear ratio
        - local incidence angle

        plus overall Sentinel-1 valid-pixel fraction.

    Notes
    -----
    TerraMind preprocessing should still use the original linear
    VV/VH raster and convert it independently to dB.

    These summary statistics are intended for:

        - QA
        - dataset metadata
        - baseline regression
        - physics inputs

    They are not used as TerraMind image inputs.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Sentinel-1 raster not found: {path}"
        )

    with rasterio.open(path) as src:
        data = src.read()

        crs = (
            src.crs.to_string()
            if src.crs is not None
            else None
        )

        resolution = (
            float(src.res[0]),
            float(src.res[1]),
        )

        width = int(src.width)
        height = int(src.height)

    # ---------------------------------------------------------
    # Validate raster structure.
    # ---------------------------------------------------------

    if data.ndim != 3:
        raise ValueError(
            "Expected Sentinel-1 raster with shape "
            f"[bands, height, width], got {data.shape}."
        )

    if data.shape[0] != 4:
        raise ValueError(
            "Expected four Sentinel-1 bands "
            "[VV, VH, localIncidenceAngle, dataMask], "
            f"got {data.shape[0]} bands."
        )

    # ---------------------------------------------------------
    # Extract bands.
    # ---------------------------------------------------------

    vv = data[0].astype(
        np.float32,
        copy=False,
    )

    vh = data[1].astype(
        np.float32,
        copy=False,
    )

    incidence_angle = data[2].astype(
        np.float32,
        copy=False,
    )

    data_mask = data[3].astype(
        np.float32,
        copy=False,
    )

    # ---------------------------------------------------------
    # Base valid-pixel mask.
    #
    # Require:
    #   - Sentinel Hub dataMask > 0
    #   - finite VV
    #   - finite VH
    #   - finite incidence angle
    #
    # Individual finite_stats() calls perform their own
    # additional finite-value checks.
    # ---------------------------------------------------------

    valid = (
        (data_mask > 0)
        & np.isfinite(vv)
        & np.isfinite(vh)
        & np.isfinite(incidence_angle)
    )

    valid_fraction = float(
        np.mean(valid)
    )

    if not np.any(valid):
        raise ValueError(
            f"No valid Sentinel-1 pixels found in {path}."
        )

    # ---------------------------------------------------------
    # Derived SAR features.
    #
    # The source VV/VH values are LINEAR_POWER.
    #
    # dB:
    #   sigma0_dB = 10 log10(sigma0_linear)
    #
    # Ratio:
    #   VH / VV
    # ---------------------------------------------------------

    eps = np.float32(
        1e-10
    )

    # Preserve invalid pixels as NaN instead of artificially
    # generating valid-looking numbers there.
    vv_db = np.full_like(
        vv,
        np.nan,
        dtype=np.float32,
    )

    vh_db = np.full_like(
        vh,
        np.nan,
        dtype=np.float32,
    )

    vh_vv_ratio = np.full_like(
        vv,
        np.nan,
        dtype=np.float32,
    )

    # VV/VH should normally be >= 0 because the downloader
    # requests LINEAR_POWER.
    #
    # log10 requires strictly positive values.
    vv_positive = (
        valid
        & (vv > eps)
    )

    vh_positive = (
        valid
        & (vh > eps)
    )

    vv_db[vv_positive] = (
        10.0
        * np.log10(
            vv[vv_positive]
        )
    )

    vh_db[vh_positive] = (
        10.0
        * np.log10(
            vh[vh_positive]
        )
    )

    # Ratio only requires a valid, positive denominator.
    ratio_valid = (
        valid
        & (vv > eps)
    )

    vh_vv_ratio[ratio_valid] = (
        vh[ratio_valid]
        / vv[ratio_valid]
    )

    # ---------------------------------------------------------
    # Physical / QA sanity indicators.
    #
    # These do not remove data. They simply expose potentially
    # suspicious values for later QA.
    # ---------------------------------------------------------

    negative_vv_fraction = float(
        np.mean(
            valid
            & (vv < 0)
        )
    )

    negative_vh_fraction = float(
        np.mean(
            valid
            & (vh < 0)
        )
    )

    invalid_angle_fraction = float(
        np.mean(
            valid
            & (
                (incidence_angle <= 0)
                | (incidence_angle >= 90)
            )
        )
    )

    # ---------------------------------------------------------
    # Robust statistics.
    #
    # finite_stats() should return:
    #
    # mean
    # median
    # std
    # p05
    # p25
    # p75
    # p95
    # minimum
    # maximum
    # valid_fraction
    # ---------------------------------------------------------

    summary = {
        "vv": finite_stats(
            vv,
            valid,
        ),

        "vh": finite_stats(
            vh,
            valid,
        ),

        "vv_db": finite_stats(
            vv_db,
            valid,
        ),

        "vh_db": finite_stats(
            vh_db,
            valid,
        ),

        "vh_vv_ratio": finite_stats(
            vh_vv_ratio,
            valid,
        ),

        "incidence_angle": finite_stats(
            incidence_angle,
            valid,
        ),

        "valid_fraction": valid_fraction,

        "qa": {
            "negative_vv_fraction": (
                negative_vv_fraction
            ),

            "negative_vh_fraction": (
                negative_vh_fraction
            ),

            "invalid_incidence_angle_fraction": (
                invalid_angle_fraction
            ),
        },

        "raster": {
            "path": str(path),
            "crs": crs,
            "resolution": resolution,
            "width": width,
            "height": height,
            "bands": 4,
        },
    }

    return summary

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