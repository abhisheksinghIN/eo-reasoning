"""Baseline Sentinel-1 / CLMS soil-moisture collocation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from data.sentinel1 import (
    download_sentinel1_patch,
    search_sentinel1,
)
from data.copernicus_ssm import download_ssm_patch

from tasks.soil_moisture.validation import (
    validate_collocation,
)


DEFAULT_CACHE = Path(".cache/soil_moisture")


def select_sentinel1_scene(
    items: list[dict[str, Any]],
    orbit_state: str | None = None,
) -> dict[str, Any]:
    """
    Select one S1 acquisition.

    For the first MVP:
    - require an actual returned scene;
    - optionally enforce orbit direction;
    - prefer VV+VH scenes.
    """

    if orbit_state:
        orbit_state = orbit_state.upper()

        items = [
            item
            for item in items
            if str(
                item.get("orbit_state", "")
            ).upper()
            == orbit_state
        ]

    if not items:
        raise ValueError(
            "No suitable Sentinel-1 acquisitions found."
        )

    dual_pol = []

    for item in items:
        polarizations = item.get("polarization") or []

        polarizations = {
            str(p).upper()
            for p in polarizations
        }

        if {"VV", "VH"}.issubset(polarizations):
            dual_pol.append(item)

    candidates = dual_pol or items

    candidates = sorted(
        candidates,
        key=lambda x: x.get("datetime") or "",
    )

    # For now choose the first suitable acquisition.
    return candidates[0]


def run_soil_moisture_collocation(
    bbox: list[float],
    start_date: str,
    end_date: str,
    orbit_state: str | None = None,
    output_dir: str | Path = DEFAULT_CACHE,
) -> dict[str, Any]:
    """
    Run the first soil-moisture baseline experiment.

    Workflow:
        S1 search
          -> select actual acquisition
          -> download S1
          -> download same-date CLMS SSM
          -> QA
          -> structured collocation result
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    scenes = search_sentinel1(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        limit=30,
    )

    if not scenes:
        raise RuntimeError(
            "No Sentinel-1 acquisitions found "
            "for the requested AOI/time range."
        )

    scene = select_sentinel1_scene(
        scenes,
        orbit_state=orbit_state,
    )

    date = scene["date"]

    if not date:
        raise RuntimeError(
            "Selected Sentinel-1 scene has no date."
        )

#    s1_path = download_sentinel1_patch(
#        bbox=bbox,
#        date=date,
#        output_dir=output_dir / "sentinel1",
#    )
    s1_path = download_sentinel1_patch(
        bbox=bbox,
        date=date,
        output_dir=output_dir / "sentinel1",
        orbit_state=scene.get("orbit_state"),
    )
    ssm_path = download_ssm_patch(
        bbox=bbox,
        date=date,
        output_path=(
            output_dir
            / "clms_ssm"
            / f"ssm_{date}.tif"
        ),
    )

    qa = validate_collocation(
        sentinel1_path=s1_path,
        ssm_path=ssm_path,
    )

    return {
        "task": "soil_moisture_collocation",
        "aoi": bbox,
        "date": date,

        "sentinel1_scene": {
            "id": scene.get("id"),
            "datetime": scene.get("datetime"),
            "orbit_state": scene.get(
                "orbit_state"
            ),
            "relative_orbit": scene.get(
                "relative_orbit"
            ),
            "polarization": scene.get(
                "polarization"
            ),
        },

        "sentinel1": qa["sentinel1"],

        "reference": {
            "product": (
                "CLMS Surface Soil Moisture "
                "Europe 1 km daily v1"
            ),
            **qa["reference"],
        },

        "quality": {
            "status": qa["status"],
            "issues": qa["issues"],
        },

        "provenance": {
            "sentinel1": (
                "Copernicus Data Space Ecosystem"
            ),
            "reference": (
                "Copernicus Land Monitoring Service"
            ),
        },

        "limitations": [
            (
                "CLMS SSM is used as a reference "
                "product, not independent ground truth."
            ),
            (
                "The baseline currently performs "
                "collocation only; it does not estimate "
                "soil moisture with a learned model."
            ),
            (
                "Sentinel-1 and CLMS SSM have different "
                "native spatial resolutions."
            ),
        ],

        "artifacts": {
            "sentinel1_path": str(s1_path),
            "ssm_path": str(ssm_path),
        },
    }