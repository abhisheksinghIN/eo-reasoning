"""Temporal Sentinel-1 / CLMS dataset utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from data.sentinel1 import (
    download_sentinel1_patch,
    search_sentinel1,
)

from data.copernicus_ssm import (
    download_ssm_patch,
)

from tasks.soil_moisture.validation import (
    summarize_clms_ssm,
    summarize_sentinel1,
)


DEFAULT_CACHE = Path(
    ".cache/soil_moisture"
)


def _has_vv_vh(
    scene: dict[str, Any],
) -> bool:
    polarizations = (
        scene.get("polarization")
        or []
    )

    polarizations = {
        str(p).upper()
        for p in polarizations
    }

    return {
        "VV",
        "VH",
    }.issubset(
        polarizations
    )


#def select_timeseries_scenes(
#    scenes: list[dict[str, Any]],
#    orbit_state: str = "descending",
#    relative_orbit: int | None = None,
#) -> list[dict[str, Any]]:
#    """
#    Select a clean temporal Sentinel-1 sequence.
#
#    Requirements:
#      - requested orbit direction;
#      - optional exact relative orbit;
#      - IW mode;
#      - VV + VH;
#      - valid date/datetime;
#      - maximum one observation per date.
#    """
#
#    requested_orbit = (
#        orbit_state.upper()
#    )
#
#    selected = []
#
#    for scene in scenes:
#
#        scene_orbit = str(
#            scene.get(
#                "orbit_state",
#                "",
#            )
#        ).upper()
#
#        if (
#            scene_orbit
#            != requested_orbit
#        ):
#            continue
#
#        if (
#            relative_orbit
#            is not None
#        ):
#            value = scene.get(
#                "relative_orbit"
#            )
#
#            if value is None:
#                continue
#
#            if int(value) != int(
#                relative_orbit
#            ):
#                continue
#
#        mode = str(
#            scene.get(
#                "instrument_mode",
#                "",
#            )
#        ).upper()
#
#        if mode != "IW":
#            continue
#
#        if not _has_vv_vh(
#            scene
#        ):
#            continue
#
#        if not scene.get(
#            "datetime"
#        ):
#            continue
#
#        if not scene.get(
#            "date"
#        ):
#            continue
#
#        selected.append(
#            scene
#        )
#
#    selected.sort(
#        key=lambda x: x[
#            "datetime"
#        ]
#    )
#
#    # Keep one acquisition per date.
#    unique = []
#    seen_dates = set()
#
#    for scene in selected:
#        date = scene["date"]
#
#        if date in seen_dates:
#            continue
#
#        unique.append(
#            scene
#        )
#
#        seen_dates.add(
#            date
#        )
#
#    return unique
def select_timeseries_scenes(
    scenes: list[dict[str, Any]],
    orbit_state: str = "descending",
    relative_orbit: int | None = None,
    allowed_platforms: set[str] | None = None,
    min_gap_days: float | None = None,
) -> list[dict[str, Any]]:
    """
    Select a clean Sentinel-1 temporal sequence.

    Filters
    -------
    - orbit direction
    - optional exact relative orbit
    - IW acquisition mode
    - VV + VH polarization
    - optional platform whitelist
    - one acquisition per date
    - optional minimum temporal separation

    Parameters
    ----------
    scenes
        Sentinel-1 STAC records.

    orbit_state
        Usually "descending" or "ascending".

    relative_orbit
        Exact Sentinel-1 relative orbit, if required.

    allowed_platforms
        Optional satellite whitelist.

        Examples:
            {"S1C", "S1D"}

        or:
            {"SENTINEL-1C", "SENTINEL-1D"}

        Matching is normalized internally.

    min_gap_days
        Optional minimum temporal separation between
        retained acquisitions.

        Leave None for discovery.

        A value such as 4.0 can later be used to reject
        unusually close acquisitions.

    Returns
    -------
    list[dict]
        Chronologically sorted clean Sentinel-1 scenes.
    """

    from datetime import datetime

    requested_orbit = (
        orbit_state
        .strip()
        .upper()
    )

    def normalize_platform(
        value: str | None,
        scene_id: str,
    ) -> str:

        if value:
            p = (
                str(value)
                .upper()
                .replace("_", "-")
            )
        else:
            p = ""

        # Normalize common STAC/platform variants.
        aliases = {
            "SENTINEL-1A": "S1A",
            "SENTINEL-1B": "S1B",
            "SENTINEL-1C": "S1C",
            "SENTINEL-1D": "S1D",
            "S1A": "S1A",
            "S1B": "S1B",
            "S1C": "S1C",
            "S1D": "S1D",
        }

        if p in aliases:
            return aliases[p]

        prefix = (
            str(scene_id)
            .upper()[:3]
        )

        if prefix in {
            "S1A",
            "S1B",
            "S1C",
            "S1D",
        }:
            return prefix

        return p

    normalized_allowed = None

    if allowed_platforms is not None:
        normalized_allowed = {
            normalize_platform(
                platform,
                "",
            )
            for platform
            in allowed_platforms
        }

    candidates = []

    for scene in scenes:

        scene_id = str(
            scene.get(
                "id",
                "",
            )
        )

        # -------------------------------------------------
        # Orbit direction
        # -------------------------------------------------

        scene_orbit = str(
            scene.get(
                "orbit_state",
                "",
            )
        ).strip().upper()

        if scene_orbit != requested_orbit:
            continue

        # -------------------------------------------------
        # Relative orbit
        # -------------------------------------------------

        if relative_orbit is not None:

            value = scene.get(
                "relative_orbit"
            )

            if value is None:
                continue

            if int(value) != int(
                relative_orbit
            ):
                continue

        # -------------------------------------------------
        # Acquisition mode
        # -------------------------------------------------

        mode = str(
            scene.get(
                "instrument_mode",
                "",
            )
        ).strip().upper()

        if mode != "IW":
            continue

        # -------------------------------------------------
        # Polarizations
        # -------------------------------------------------

        polarizations = (
            scene.get(
                "polarization"
            )
            or []
        )

        polarizations = {
            str(p).upper()
            for p in polarizations
        }

        if not {
            "VV",
            "VH",
        }.issubset(
            polarizations
        ):
            continue

        # -------------------------------------------------
        # Date / time
        # -------------------------------------------------

        datetime_string = (
            scene.get(
                "datetime"
            )
        )

        date_string = (
            scene.get(
                "date"
            )
        )

        if not datetime_string:
            continue

        if not date_string:
            continue

        # -------------------------------------------------
        # Satellite platform
        # -------------------------------------------------

        platform = normalize_platform(
            scene.get(
                "platform"
            ),
            scene_id,
        )

        if (
            normalized_allowed
            is not None
            and platform
            not in normalized_allowed
        ):
            continue

        item = dict(
            scene
        )

        item[
            "platform"
        ] = platform

        candidates.append(
            item
        )

    # -----------------------------------------------------
    # Chronological ordering
    # -----------------------------------------------------

    candidates.sort(
        key=lambda scene: scene[
            "datetime"
        ]
    )

    # -----------------------------------------------------
    # Keep one acquisition per calendar date.
    # -----------------------------------------------------

    unique = []
    seen_dates = set()

    for scene in candidates:

        date = scene["date"]

        if date in seen_dates:
            continue

        unique.append(
            scene
        )

        seen_dates.add(
            date
        )

    # -----------------------------------------------------
    # Optional temporal-gap filter.
    # -----------------------------------------------------

    if (
        min_gap_days is not None
        and unique
    ):

        filtered = [
            unique[0]
        ]

        previous_datetime = datetime.fromisoformat(
            unique[0][
                "datetime"
            ].replace(
                "Z",
                "+00:00",
            )
        )

        for scene in unique[1:]:

            current_datetime = datetime.fromisoformat(
                scene[
                    "datetime"
                ].replace(
                    "Z",
                    "+00:00",
                )
            )

            gap_days = (
                current_datetime
                - previous_datetime
            ).total_seconds() / 86400.0

            if gap_days < min_gap_days:
                continue

            filtered.append(
                scene
            )

            previous_datetime = (
                current_datetime
            )

        unique = filtered

    return unique

#def discover_soil_moisture_timeseries(
#    bbox: list[float],
#    start_date: str,
#    end_date: str,
#    orbit_state: str = "descending",
#    relative_orbit: int | None = None,
#) -> list[dict[str, Any]]:
#    """
#    Search STAC exactly once and then filter locally.
#    """
#
#    scenes = search_sentinel1(
#        bbox=bbox,
#        start_date=start_date,
#        end_date=end_date,
#        limit=100,
#    )
#
#    return select_timeseries_scenes(
#        scenes=scenes,
#        orbit_state=orbit_state,
#        relative_orbit=relative_orbit,
#    )
def discover_soil_moisture_timeseries(
    bbox: list[float],
    start_date: str,
    end_date: str,
    orbit_state: str = "descending",
    relative_orbit: int | None = None,
    allowed_platforms: set[str] | None = None,
    min_gap_days: float | None = None,
) -> list[dict[str, Any]]:
    """
    Search Sentinel-1 STAC once and filter locally.
    """

    scenes = search_sentinel1(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        limit=100,
    )

    return select_timeseries_scenes(
        scenes=scenes,
        orbit_state=orbit_state,
        relative_orbit=relative_orbit,
        allowed_platforms=allowed_platforms,
        min_gap_days=min_gap_days,
    )

def materialize_timeseries(
    bbox: list[float],
    start_date: str,
    end_date: str,
    orbit_state: str = "descending",
    relative_orbit: int | None = None,
    allowed_platforms: set[str] | None = None,
    min_gap_days: float | None = None,
    max_scenes: int | None = None,
    width: int = 224,
    height: int = 224,
    output_dir: str | Path = DEFAULT_CACHE,
) -> list[dict[str, Any]]:
    """
    Discover, download, and summarize a clean Sentinel-1 / CLMS
    soil-moisture time series.

    The STAC catalogue is searched only once for the requested
    date range. Scene filtering is then performed locally.

    Parameters
    ----------
    bbox
        AOI in WGS84:
        [min_lon, min_lat, max_lon, max_lat].

    start_date
        Start date in YYYY-MM-DD format.

    end_date
        End date in YYYY-MM-DD format.

    orbit_state
        Sentinel-1 orbit direction.
        Usually "descending" or "ascending".

    relative_orbit
        Optional exact Sentinel-1 relative orbit.

        For the current Po Valley experiment:
            168

    allowed_platforms
        Optional set of Sentinel-1 platforms.

        Example:
            {"S1C", "S1D"}

        If None, all platforms passing the other filters
        are accepted.

    min_gap_days
        Optional minimum number of days between retained
        acquisitions.

        Example:
            4.0

        Leave as None during initial discovery if you want
        to retain all valid acquisitions.

    max_scenes
        Optional maximum number of scenes to materialize.

        Useful for smoke tests.

        Example:
            max_scenes=3

    width
        Output Sentinel-1 raster width.

    height
        Output Sentinel-1 raster height.

    output_dir
        Root cache/output directory.

    Returns
    -------
    list[dict]
        One record for every retained Sentinel-1 acquisition,
        including:

        - acquisition metadata
        - exact S1 path
        - CLMS SSM path
        - Sentinel-1 statistics
        - CLMS reference statistics
    """

    output_dir = Path(
        output_dir
    )

    # ---------------------------------------------------------
    # 1. Discover scenes ONCE for the complete date range.
    # ---------------------------------------------------------

    scenes = discover_soil_moisture_timeseries(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        orbit_state=orbit_state,
        relative_orbit=relative_orbit,
        allowed_platforms=allowed_platforms,
        min_gap_days=min_gap_days,
    )

    if not scenes:
        raise RuntimeError(
            "No Sentinel-1 scenes matched the requested "
            "time-series filters."
        )

    # ---------------------------------------------------------
    # 2. Optionally restrict the number of scenes.
    #
    # Useful for the three-date engineering smoke test.
    # ---------------------------------------------------------

    if max_scenes is not None:

        if max_scenes < 1:
            raise ValueError(
                "max_scenes must be >= 1 or None."
            )

        scenes = scenes[
            :max_scenes
        ]

    # ---------------------------------------------------------
    # 3. Ensure output directories exist.
    # ---------------------------------------------------------

    sentinel1_dir = (
        output_dir
        / "sentinel1"
    )

    clms_dir = (
        output_dir
        / "clms_ssm"
    )

    sentinel1_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    clms_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 4. Materialize each selected acquisition.
    # ---------------------------------------------------------

    records = []

    for index, scene in enumerate(
        scenes,
        start=1,
    ):

        date = scene.get(
            "date"
        )

        acquisition_datetime = scene.get(
            "datetime"
        )

        scene_id = scene.get(
            "id"
        )

        platform = scene.get(
            "platform"
        )

        scene_orbit_state = scene.get(
            "orbit_state"
        )

        scene_relative_orbit = scene.get(
            "relative_orbit"
        )

        polarization = scene.get(
            "polarization"
        )

        if not date:
            raise ValueError(
                f"Scene has no date: {scene}"
            )

        if not acquisition_datetime:
            raise ValueError(
                f"Scene has no acquisition datetime: "
                f"{scene_id}"
            )

        # -----------------------------------------------------
        # Progress information
        # -----------------------------------------------------

        print(
            f"[{index}/{len(scenes)}] "
            f"{date} "
            f"{platform or ''} "
            f"orbit={scene_relative_orbit}"
        )

        # -----------------------------------------------------
        # Sentinel-1
        #
        # IMPORTANT:
        # Pass the exact STAC acquisition datetime.
        #
        # This prevents Sentinel Hub from accidentally
        # selecting another descending acquisition from
        # the same calendar day.
        # -----------------------------------------------------

        s1_path = download_sentinel1_patch(
            bbox=bbox,
            date=date,
            output_dir=sentinel1_dir,
            width=width,
            height=height,
            orbit_state=scene_orbit_state,
            acquisition_datetime=acquisition_datetime,
        )

        # -----------------------------------------------------
        # CLMS reference for exactly the same calendar date.
        # -----------------------------------------------------

        ssm_path = download_ssm_patch(
            bbox=bbox,
            date=date,
            output_path=(
                clms_dir
                / f"ssm_{date}.tif"
            ),
        )

        # -----------------------------------------------------
        # QA / statistics
        # -----------------------------------------------------

        s1_summary = summarize_sentinel1(
            s1_path
        )

        ssm_summary = summarize_clms_ssm(
            ssm_path
        )

        # -----------------------------------------------------
        # Structured temporal record
        # -----------------------------------------------------

        record = {
            "date": date,

            "datetime": acquisition_datetime,

            "scene_id": scene_id,

            "platform": platform,

            "orbit_state": (
                scene_orbit_state
            ),

            "relative_orbit": (
                scene_relative_orbit
            ),

            "polarization": polarization,

            "s1_path": str(
                s1_path
            ),

            "ssm_path": str(
                ssm_path
            ),

            "sentinel1": (
                s1_summary
            ),

            "reference": (
                ssm_summary
            ),
        }

        records.append(
            record
        )

    # ---------------------------------------------------------
    # 5. Ensure final records remain chronological.
    # ---------------------------------------------------------

    records.sort(
        key=lambda record: record[
            "datetime"
        ]
    )

    return records

#"""Build multi-date Sentinel-1 / CLMS SSM collocations."""
#
#from __future__ import annotations
#
#import csv
#from pathlib import Path
#from typing import Any
#
#from data.sentinel1 import search_sentinel1
#
#
#def select_timeseries_scenes(
#    scenes: list[dict[str, Any]],
#    orbit_state: str = "descending",
#    relative_orbit: int | None = None,
#) -> list[dict[str, Any]]:
#
#    orbit_state = orbit_state.lower()
#
#    selected = []
#
#    for scene in scenes:
#
#        if (
#            str(scene.get("orbit_state", "")).lower()
#            != orbit_state
#        ):
#            continue
#
#        if (
#            relative_orbit is not None
#            and scene.get("relative_orbit")
#            != relative_orbit
#        ):
#            continue
#
#        polarizations = {
#            str(p).upper()
#            for p in (
#                scene.get("polarization") or []
#            )
#        }
#
#        if not {"VV", "VH"}.issubset(
#            polarizations
#        ):
#            continue
#
#        selected.append(scene)
#
#    selected.sort(
#        key=lambda x: x.get("datetime") or ""
#    )
#
#    # Avoid duplicate acquisitions/dates.
#    unique = {}
#
#    for scene in selected:
#        unique.setdefault(
#            scene["date"],
#            scene,
#        )
#
#    return list(unique.values())
#
#
#def discover_soil_moisture_timeseries(
#    bbox: list[float],
#    start_date: str,
#    end_date: str,
#    orbit_state: str = "descending",
#    relative_orbit: int | None = None,
#):
#    scenes = search_sentinel1(
#        bbox=bbox,
#        start_date=start_date,
#        end_date=end_date,
#        limit=100,
#    )
#
#    return select_timeseries_scenes(
#        scenes=scenes,
#        orbit_state=orbit_state,
#        relative_orbit=relative_orbit,
#    )