"""Temporal statistics for scalar EO indicators."""

from __future__ import annotations

import numpy as np


def temporal_statistics(values) -> dict:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("values must be a non-empty 1D sequence.")

    valid = x[np.isfinite(x)]
    if not valid.size:
        return {k: float("nan") for k in [
            "mean", "std", "minimum", "maximum", "start", "end", "absolute_change", "relative_change"
        ]}

    start = float(x[0])
    end = float(x[-1])
    abs_change = end - start
    rel_change = float("nan") if not np.isfinite(start) or abs(start) < 1e-8 else abs_change / abs(start)

    return {
        "mean": float(np.nanmean(x)),
        "std": float(np.nanstd(x)),
        "minimum": float(np.nanmin(x)),
        "maximum": float(np.nanmax(x)),
        "start": start,
        "end": end,
        "absolute_change": float(abs_change),
        "relative_change": float(rel_change),
    }


def temporal_slope(values, times=None) -> float:
    x = np.asarray(values, dtype=np.float64)
    times = np.arange(len(x), dtype=np.float64) if times is None else np.asarray(times, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(times)
    if valid.sum() < 2:
        return float("nan")
    return float(np.polyfit(times[valid], x[valid], 1)[0])


def summarize_indicator(values) -> dict:
    result = temporal_statistics(values)
    result["slope"] = temporal_slope(values)
    return result
