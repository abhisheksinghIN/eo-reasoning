"""Sentinel-1 discovery for the soil-moisture baseline."""

from __future__ import annotations

import hashlib
from pathlib import Path

import requests

from data.cdse_auth import get_access_token
from data.stac_client import CDSESTACClient


def search_sentinel1(
    bbox: list,
    start_date: str,
    end_date: str,
    limit: int = 20,
):
    stac = CDSESTACClient()

    items = stac.search(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        collections=["sentinel-1-grd"],
        limit=limit,
    )

    results = []

    for item in items:
        props = item.properties

        results.append(
            {
                "id": item.id,
                "datetime": (
                    item.datetime.isoformat()
                    if item.datetime
                    else None
                ),
                "date": (
                    item.datetime.date().isoformat()
                    if item.datetime
                    else None
                ),
                "instrument_mode": props.get(
                    "sar:instrument_mode"
                ),
                "orbit_state": props.get(
                    "sat:orbit_state"
                ),
                "relative_orbit": props.get(
                    "sat:relative_orbit"
                ),
                "polarization": (
                    props.get("sar:polarizations")
                    or props.get("s1:polarization")
                ),
            }
        )

    return results
    
    
PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"

S1_OUTPUT_BANDS = [
    "VV",
    "VH",
    "localIncidenceAngle",
    "dataMask",
]


def _s1_evalscript() -> str:
    return """
//VERSION=3

function setup() {
    return {
        input: [{
            bands: [
                "VV",
                "VH",
                "localIncidenceAngle",
                "dataMask"
            ],
            units: [
                "LINEAR_POWER",
                "LINEAR_POWER",
                "DN",
                "DN"
            ]
        }],
        output: {
            bands: 4,
            sampleType: "FLOAT32"
        }
    };
}

function evaluatePixel(s) {
    return [
        s.VV,
        s.VH,
        s.localIncidenceAngle,
        s.dataMask
    ];
}
"""


def _cache_name(
    bbox,
    date: str,
    width: int,
    height: int,
    orbit_state: str | None,
) -> str:
    band_signature = ",".join(S1_OUTPUT_BANDS)

    key = (
        f"{list(bbox)}|"
        f"{date}|"
        f"{width}|"
        f"{height}|"
        f"{orbit_state}|"
        f"{band_signature}|"
        f"SIGMA0_ELLIPSOID|"
        f"v1"
    ).encode("utf-8")

    digest = hashlib.sha1(key).hexdigest()[:12]

    return f"s1_{date}_{digest}.tif"


def download_sentinel1_patch(
    bbox: list,
    date: str,
    output_dir: str | Path = ".cache/soil_moisture/sentinel1",
    width: int = 256,
    height: int = 256,
    orbit_state: str | None = None,
    force: bool = False,
) -> str:
    """
    Download an orthorectified Sentinel-1 GRD patch.

    Output bands:
        0 = VV linear power
        1 = VH linear power
        2 = local incidence angle [degrees]
        3 = dataMask
    """

    if len(bbox) != 4:
        raise ValueError(
            "bbox must be [min_lon, min_lat, max_lon, max_lat]."
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = out_dir / _cache_name(
        bbox=bbox,
        date=date,
        width=width,
        height=height,
        orbit_state=orbit_state,
    )

    if out_path.exists() and not force:
        return str(out_path)

    token = get_access_token()

    data_filter = {
        "timeRange": {
            "from": f"{date}T00:00:00Z",
            "to": f"{date}T23:59:59Z",
        },
        "acquisitionMode": "IW",
        "polarization": "DV",
        "mosaickingOrder": "mostRecent",
    }

    if orbit_state:
        orbit = orbit_state.upper()

        if orbit not in {
            "ASCENDING",
            "DESCENDING",
        }:
            raise ValueError(
                "orbit_state must be ASCENDING or DESCENDING."
            )

        data_filter["orbitDirection"] = orbit

    request_json = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {
                    "crs": (
                        "http://www.opengis.net/"
                        "def/crs/OGC/1.3/CRS84"
                    )
                },
            },
            "data": [
                {
                    "type": "sentinel-1-grd",
                    "dataFilter": data_filter,
                    "processing": {
                        "orthorectify": "true",
                        "backCoeff": "SIGMA0_ELLIPSOID",
                        "demInstance": "COPERNICUS_30",
                    },
                }
            ],
        },

        "output": {
            "width": int(width),
            "height": int(height),

            "responses": [
                {
                    "identifier": "default",
                    "format": {
                        "type": "image/tiff"
                    },
                }
            ],
        },

        "evalscript": _s1_evalscript(),
    }

    response = requests.post(
        PROCESS_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "image/tiff",
        },
        json=request_json,
        timeout=180,
    )

    if not response.ok:
        raise RuntimeError(
            "Sentinel-1 Process API failed.\n"
            f"HTTP {response.status_code}\n"
            f"{response.text[:1000]}"
        )

    out_path.write_bytes(
        response.content
    )

    return str(out_path)
    
