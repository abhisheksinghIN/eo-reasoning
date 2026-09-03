"""Deterministic end-to-end EO analysis pipeline."""

from __future__ import annotations

from typing import List

from data.sentinel2 import download_temporal_stack
from tools.analysis_tools import analyze_spectral_timeseries
from tools.evidence_tools import build_evidence_object
from tools.geofm_tools import run_prithvi_temporal
from tools.physics_tools import check_vegetation_consistency

from pathlib import Path
import numpy as np
from analysis.change_map import (write_ndvi_change_geotiff,)
from analysis.prithvi_change import (write_prithvi_change_geotiff,)



def analyze_temporal_aoi(
    bbox: list,
    dates: List[str],
    cache_dir: str = ".cache/sentinel2",
) -> dict:
    """Run CDSE -> S2 -> Prithvi -> spectral/latent change -> evidence."""
    if len(dates) != 3:
        raise ValueError("Provide exactly three dates for the Prithvi v1 MVP.")

    paths = download_temporal_stack(
        bbox=bbox,
        dates=dates,
        output_dir=cache_dir,
        width=224,
        height=224,
    )
    spectral = analyze_spectral_timeseries(paths=paths, dates=dates)
    geofm = run_prithvi_temporal(paths=paths)
# --------------
    output_dir = Path(".cache/outputs")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    ndvi_change_path = write_ndvi_change_geotiff(
        start_path=paths[0],
        end_path=paths[-1],
        output_path=(
            output_dir
            / f"delta_ndvi_{dates[0]}_{dates[-1]}.tif"
        ),
    )
    
    spatial_change = np.asarray(
        geofm.pop("_spatial_change_map"),
        dtype=np.float32,
    )
    
    prithvi_change_path = (
        write_prithvi_change_geotiff(
            change=spatial_change,
            start_path=paths[0],
            end_path=paths[-1],
            output_path=(
                output_dir
                / (
                    f"prithvi_change_"
                    f"{dates[0]}_{dates[-1]}.tif"
                )
            ),
        )
    )
# --------------                
    physical = check_vegetation_consistency(
        ndvi_change=spectral["ndvi"]["absolute_change"],
        ndmi_change=spectral["ndmi"]["absolute_change"],
        embedding_cosine_distance=geofm["summary"]["start_end_cosine_distance"],
    )

    result = build_evidence_object(
        bbox=bbox,
        dates=dates,
        spectral=spectral,
        geofm=geofm,
        physical_consistency=physical,
    )
#    result["artifacts"] = {"sentinel2_paths": paths}
    result["artifacts"] = {"sentinel2_paths": paths, "ndvi_change_geotiff": ndvi_change_path, "prithvi_change_geotiff": prithvi_change_path,}
    return result
