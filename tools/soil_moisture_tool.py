"""Deterministic soil-moisture analysis tool for LLM orchestration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from data.sentinel1 import (
    download_sentinel1_patch,
)
from data.copernicus_ssm import (
    download_ssm_patch,
)

from tasks.soil_moisture.dataset import (
    discover_soil_moisture_timeseries,
)
from tasks.soil_moisture.preprocessing import (
    prepare_s1_terramind,
)
from tasks.soil_moisture.validation import (
    summarize_sentinel1,
    summarize_clms_ssm,
)
from tasks.soil_moisture.model import (
    GeoFMJEPASoilMoisture,
)


CACHE_ROOT = Path(
    ".cache/soil_moisture"
)

MODEL_SEED = 42

MODEL_STATUS = "untrained_prototype"


def _set_seed(
    seed: int = MODEL_SEED,
) -> None:
    """Set deterministic prototype initialization."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_bbox(
    bbox: list[float],
) -> list[float]:
    """
    Validate a WGS84 bounding box.
    """

    if len(bbox) != 4:
        raise ValueError(
            "bbox must contain exactly four values: "
            "[min_lon, min_lat, max_lon, max_lat]."
        )

    bbox = [
        float(value)
        for value in bbox
    ]

    min_lon, min_lat, max_lon, max_lat = bbox

    if min_lon >= max_lon:
        raise ValueError(
            "bbox min_lon must be smaller than max_lon."
        )

    if min_lat >= max_lat:
        raise ValueError(
            "bbox min_lat must be smaller than max_lat."
        )

    if not (
        -180 <= min_lon <= 180
        and -180 <= max_lon <= 180
        and -90 <= min_lat <= 90
        and -90 <= max_lat <= 90
    ):
        raise ValueError(
            "bbox coordinates are outside valid WGS84 ranges."
        )

    return bbox


@lru_cache(maxsize=1)
def _get_prototype_model() -> GeoFMJEPASoilMoisture:
    """
    Load Model v0 once per Python process.

    IMPORTANT:
    This currently uses randomly initialized JEPA / SSM / physics
    components. TerraMind itself is pretrained and frozen.

    Once a validated fine-tuned checkpoint exists, checkpoint
    loading will be added here without changing the LLM interface.
    """

    _set_seed(
        MODEL_SEED
    )

    model = GeoFMJEPASoilMoisture(
        use_s2=False,
        use_jepa=True,
        use_physics=True,
        freeze_geofm=True,
    )

    model.eval()

    return model


def _materialize_scene(
    scene: dict,
    bbox: list[float],
) -> dict:
    """
    Download one exact Sentinel-1 acquisition and same-date CLMS SSM.
    """

    date = scene["date"]

    s1_dir = (
        CACHE_ROOT
        / "sentinel1"
    )

    ssm_dir = (
        CACHE_ROOT
        / "clms_ssm"
    )

    s1_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ssm_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    s1_path = download_sentinel1_patch(
        bbox=bbox,
        date=date,
        output_dir=s1_dir,
        width=224,
        height=224,
        orbit_state=scene.get(
            "orbit_state"
        ),
        acquisition_datetime=scene.get(
            "datetime"
        ),
    )

    ssm_path = download_ssm_patch(
        bbox=bbox,
        date=date,
        output_path=(
            ssm_dir
            / f"ssm_{date}.tif"
        ),
    )

    return {
        "date": date,
        "datetime": scene["datetime"],
        "scene_id": scene["id"],
        "platform": scene.get(
            "platform"
        ),
        "orbit_state": scene.get(
            "orbit_state"
        ),
        "relative_orbit": scene.get(
            "relative_orbit"
        ),
        "polarization": scene.get(
            "polarization"
        ),
        "s1_path": str(
            s1_path
        ),
        "ssm_path": str(
            ssm_path
        ),
        "sentinel1": summarize_sentinel1(
            s1_path
        ),
        "reference": summarize_clms_ssm(
            ssm_path
        ),
    }


def analyze_soil_moisture_aoi(
    bbox: list[float],
    start_date: str,
    end_date: str,
    orbit_state: str = "descending",
    relative_orbit: int = 168,
) -> dict:
    """
    Analyze prototype surface soil moisture for an Earth-observation AOI.

    Use this tool whenever a user asks for quantitative soil-moisture
    analysis from Sentinel-1 over a bounding box and date range.

    The tool searches Sentinel-1, selects a same-geometry temporal
    sequence, runs TerraMind + temporal JEPA + SAR physics, and
    provides CLMS Surface Soil Moisture as a reference product.

    Args:
        bbox:
            WGS84 bounding box as
            [min_lon, min_lat, max_lon, max_lat].

        start_date:
            Start date in YYYY-MM-DD format.

        end_date:
            End date in YYYY-MM-DD format.

        orbit_state:
            Sentinel-1 orbit direction.
            Usually "descending" or "ascending".

        relative_orbit:
            Sentinel-1 relative orbit number.
            The current Po Valley demonstrator uses 168.

    Returns:
        Structured EO evidence.

        IMPORTANT:
        When model.status is "untrained_prototype",
        retrieval_ssm and predictive_ssm are developmental outputs
        and MUST NOT be described as validated soil-moisture estimates.
    """

    bbox = _validate_bbox(
        bbox
    )

    # ---------------------------------------------------------
    # 1. Discover the complete Sentinel-1 range only once.
    # ---------------------------------------------------------

    scenes = discover_soil_moisture_timeseries(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        orbit_state=orbit_state,
        relative_orbit=relative_orbit,
        allowed_platforms=None,
        min_gap_days=None,
    )

    if len(scenes) < 3:
        return {
            "task": "soil_moisture_analysis",
            "status": "insufficient_observations",
            "requested": {
                "bbox": bbox,
                "start_date": start_date,
                "end_date": end_date,
                "orbit_state": orbit_state,
                "relative_orbit": relative_orbit,
            },
            "available_observations": len(
                scenes
            ),
            "message": (
                "At least three compatible Sentinel-1 "
                "observations are required for the "
                "current temporal JEPA prototype."
            ),
        }

    # ---------------------------------------------------------
    # 2. Use the latest three compatible observations.
    #
    # This makes the final scene the target date.
    # ---------------------------------------------------------

    selected_scenes = scenes[
        -3:
    ]

    records = [
        _materialize_scene(
            scene=scene,
            bbox=bbox,
        )
        for scene in selected_scenes
    ]

    # ---------------------------------------------------------
    # 3. Prepare TerraMind S1GRD temporal sequence.
    # ---------------------------------------------------------

    frames = []

    for record in records:

        frame = prepare_s1_terramind(
            record["s1_path"],
            image_size=224,
        )

        if not torch.isfinite(
            frame
        ).all():
            raise RuntimeError(
                "Non-finite TerraMind input for "
                f"{record['date']}."
            )

        frames.append(
            frame
        )

    # [T,C,H,W]
    sequence = torch.stack(
        frames,
        dim=0,
    )

    # [B,T,C,H,W]
    sequence = sequence.unsqueeze(
        0
    )

    # ---------------------------------------------------------
    # 4. Extract target-date physical/reference evidence.
    # ---------------------------------------------------------

    target = records[-1]

    target_s1 = target[
        "sentinel1"
    ]

    target_reference = target[
        "reference"
    ]

    incidence_angle_value = float(
        target_s1[
            "incidence_angle"
        ]["median"]
    )

    observed_vv_value = float(
        target_s1[
            "vv"
        ]["median"]
    )

    observed_vh_value = float(
        target_s1[
            "vh"
        ]["median"]
    )

    reference_ssm_value = float(
        target_reference[
            "ssm_percent_saturation"
        ]["median"]
    )

    reference_noise_value = float(
        target_reference[
            "ssm_noise"
        ]["median"]
    )

    incidence_angle = torch.tensor(
        [
            incidence_angle_value
        ],
        dtype=torch.float32,
    )

    # ---------------------------------------------------------
    # 5. Run current GeoFM + JEPA + physics prototype.
    # ---------------------------------------------------------

    model = _get_prototype_model()

    with torch.no_grad():

        output = model(
            s1_sequence=sequence,
            incidence_angle=incidence_angle,
        )

        retrieval_ssm = float(
            output[
                "retrieval_ssm"
            ].detach().cpu().item()
        )

        predictive_ssm = float(
            output[
                "predictive_ssm"
            ].detach().cpu().item()
        )

        predicted_vv = float(
            output[
                "predicted_vv"
            ].detach().cpu().item()
        )

        latent_cosine_similarity = float(
            F.cosine_similarity(
                output[
                    "predicted_latent"
                ],
                output[
                    "target_latent"
                ],
                dim=-1,
            )
            .detach()
            .cpu()
            .item()
        )

    latent_cosine_distance = (
        1.0
        - latent_cosine_similarity
    )

    physics_residual = (
        predicted_vv
        - observed_vv_value
    )

    physics_abs_residual = abs(
        physics_residual
    )

    # ---------------------------------------------------------
    # 6. Structured evidence only.
    #
    # No natural-language scientific interpretation occurs
    # inside this deterministic tool.
    # ---------------------------------------------------------

    return {
        "task": "soil_moisture_analysis",
        "status": "success",

        "requested": {
            "bbox": bbox,
            "start_date": start_date,
            "end_date": end_date,
            "orbit_state": orbit_state,
            "relative_orbit": relative_orbit,
        },

        "model": {
            "geofm": "terramind_v1_base",
            "geofm_pretrained": True,
            "geofm_frozen": True,

            "jepa": True,

            "physics": True,
            "physics_type": (
                "WCM-inspired SAR observation "
                "consistency proxy"
            ),

            "sentinel2_used": False,

            "checkpoint": None,

            "status": MODEL_STATUS,

            "warning": (
                "The GeoFM downstream soil-moisture head, "
                "JEPA predictor, and physics parameters have "
                "not yet been trained and validated. Numerical "
                "model outputs are developmental only."
            ),
        },

        "observations": [
            {
                "date": record[
                    "date"
                ],
                "datetime": record[
                    "datetime"
                ],
                "platform": record[
                    "platform"
                ],
                "scene_id": record[
                    "scene_id"
                ],
                "orbit_state": record[
                    "orbit_state"
                ],
                "relative_orbit": record[
                    "relative_orbit"
                ],
                "polarization": record[
                    "polarization"
                ],
            }
            for record in records
        ],

        "target_observation": {
            "date": target[
                "date"
            ],

            "platform": target[
                "platform"
            ],

            "scene_id": target[
                "scene_id"
            ],

            "vv_linear_median": (
                observed_vv_value
            ),

            "vh_linear_median": (
                observed_vh_value
            ),

            "incidence_angle_deg_median": (
                incidence_angle_value
            ),

            "valid_fraction": float(
                target_s1[
                    "valid_fraction"
                ]
            ),
        },

        "model_evidence": {
            "retrieval_ssm_percent": (
                retrieval_ssm
            ),

            "predictive_ssm_percent": (
                predictive_ssm
            ),

            "predicted_vv_linear": (
                predicted_vv
            ),

            "jepa_latent_cosine_distance": (
                latent_cosine_distance
            ),

            "physics_vv_residual": (
                physics_residual
            ),

            "physics_vv_absolute_residual": (
                physics_abs_residual
            ),
        },

        "reference": {
            "product": (
                "CLMS Surface Soil Moisture "
                "Europe 1 km daily v1"
            ),

            "ssm_percent_saturation_median": (
                reference_ssm_value
            ),

            "ssm_noise_percent_median": (
                reference_noise_value
            ),

            "role": (
                "reference / weak supervision; "
                "not independent ground truth"
            ),
        },

        "interpretation_status": (
            "not_validated"
        ),
    }