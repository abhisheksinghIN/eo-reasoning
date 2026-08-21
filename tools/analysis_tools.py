"""Spectral and temporal analysis tools."""

from __future__ import annotations

from typing import List

from analysis.temporal import summarize_indicator
from data.preprocessing import read_process_tiff, spectral_indices


def analyze_spectral_timeseries(paths: List[str], dates: List[str]) -> dict:
    if len(paths) != len(dates):
        raise ValueError("paths and dates must have equal length.")

    per_date = []
    for path, date in zip(paths, dates):
        metrics = spectral_indices(read_process_tiff(path))
        per_date.append({"date": date, **metrics})

    result = {"per_date": per_date}
    for key in ["ndvi", "ndmi", "evi", "valid_fraction"]:
        result[key] = summarize_indicator([row[key] for row in per_date])
    return result
