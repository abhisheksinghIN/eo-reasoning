"""Preprocessing for Prithvi and spectral-index analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import rasterio
import torch

#PRITHVI_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07"]
#ALL_BANDS = PRITHVI_BANDS + ["B08", "B11", "SCL", "dataMask"]
PRITHVI_BANDS = [
    "B02",
    "B03",
    "B04",
    "B8A",
    "B11",
    "B12",
]

ALL_BANDS = PRITHVI_BANDS + [
    "B08",
    "SCL",
    "dataMask",
]

PRITHVI_MEAN = np.array(
    [
        775.2290211032589,
        1080.992780391705,
        1228.5855250417867,
        2497.2022620507532,
        2204.2139147975554,
        1610.8324823273745,
    ],
    dtype=np.float32,
)

PRITHVI_STD = np.array(
    [
        1281.526139861424,
        1270.0297974547493,
        1399.4802505642526,
        1368.3446143747644,
        1291.6764008585435,
        1154.505683480695,
    ],
    dtype=np.float32,
)

NO_DATA_FLOAT = 0.0001


def read_process_tiff(path: str | Path) -> np.ndarray:
    with rasterio.open(path) as src:
        array = src.read()

    if array.shape[0] != len(ALL_BANDS):
        raise ValueError(
            f"Expected {len(ALL_BANDS)} bands {ALL_BANDS}; "
            f"received shape {array.shape} from {path}."
        )
    return array


def prepare_prithvi_frame(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[0] != len(ALL_BANDS):
        raise ValueError(
            f"Expected frame [{len(ALL_BANDS)},H,W], received {frame.shape}."
        )

    x = frame[:6].astype(np.float32)
    data_mask = frame[9] > 0

    x = (x - PRITHVI_MEAN[:, None, None]) / PRITHVI_STD[:, None, None]
    x[:, ~data_mask] = NO_DATA_FLOAT
    return x.astype(np.float32)


def prepare_prithvi_tensor(
    paths: Iterable[str | Path],
    expected_size: int = 224,
) -> Tuple[torch.Tensor, List[np.ndarray]]:
    paths = list(paths)
    if len(paths) != 3:
        raise ValueError("Exactly three frames are required for the v1 MVP.")

    raw_frames = [read_process_tiff(path) for path in paths]

    for frame in raw_frames:
        if frame.shape[-2:] != (expected_size, expected_size):
            raise ValueError(
                f"Expected {expected_size}x{expected_size}; got {frame.shape[-2:]}. "
                "Request aligned patches at the model size rather than resizing here."
            )

    normalized = [prepare_prithvi_frame(frame) for frame in raw_frames]
    cube = np.stack(normalized, axis=1)  # C,T,H,W
    tensor = torch.from_numpy(cube).unsqueeze(0)  # B,C,T,H,W
    return tensor, raw_frames


def valid_surface_mask(frame: np.ndarray) -> np.ndarray:
    #scl = frame[8].astype(np.int16)
    #data_mask = frame[9] > 0
    scl = frame[7].astype(np.int16)
    data_mask = frame[8] > 0
    invalid_scl = np.isin(scl, [0, 1, 3, 8, 9, 10, 11])
    return data_mask & (~invalid_scl)


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(num.shape, np.nan, dtype=np.float32)
    np.divide(num, den, out=out, where=np.abs(den) > 1e-8)
    return out


def spectral_indices(frame: np.ndarray) -> dict:
    frame = np.asarray(frame, dtype=np.float32)
    valid = valid_surface_mask(frame)

    b02 = frame[0] / 10000.0
    b04 = frame[2] / 10000.0
    #b08 = frame[6] / 10000.0
    #b11 = frame[7] / 10000.0
    b11 = frame[4] / 10000.0
    b08 = frame[6] / 10000.0

    ndvi = _safe_ratio(b08 - b04, b08 + b04)
    ndmi = _safe_ratio(b08 - b11, b08 + b11)
    evi_denom = b08 + 6.0 * b04 - 7.5 * b02 + 1.0
    evi = _safe_ratio(2.5 * (b08 - b04), evi_denom)

    def masked_mean(x):
        values = x[valid & np.isfinite(x)]
        return float(np.mean(values)) if values.size else float("nan")

    return {
        "ndvi": masked_mean(ndvi),
        "ndmi": masked_mean(ndmi),
        "evi": masked_mean(evi),
        "valid_fraction": float(np.mean(valid)),
    }
