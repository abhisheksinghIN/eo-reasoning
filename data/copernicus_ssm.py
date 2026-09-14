"""Copernicus CLMS Surface Soil Moisture retrieval."""

from __future__ import annotations

from pathlib import Path

import requests

from data.cdse_auth import get_access_token

from rasterio.warp import transform_bounds


PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"

SSM_COLLECTION_ID = (
    "df9e9783-f580-433a-b798-3acd2760b94e"
)

SSM_DATA_TYPE = f"byoc-{SSM_COLLECTION_ID}"


def _evalscript() -> str:
    return """
//VERSION=3

function setup() {
    return {
        input: [
            "SSM",
            "SSM_NOISE",
            "dataMask"
        ],
        output: {
            bands: 3,
            sampleType: "FLOAT32"
        }
    };
}

function evaluatePixel(s) {
    // CLMS SSM source format uses a scale factor of 0.5.
    return [
        s.SSM * 0.5,
        s.SSM_NOISE * 0.5,
        s.dataMask
    ];
}
"""

def download_ssm_patch(
    bbox: list,
    date: str,
    output_path: str | Path,
    crs_epsg: int = 32632,
    resolution: int = 1000,
) -> str:

    if len(bbox) != 4:
        raise ValueError(
            "bbox must be [min_lon, min_lat, max_lon, max_lat]."
        )

    # Input bbox is WGS84 lon/lat.
    # CLMS output is requested in a metric projected CRS so that
    # resx/resy=1000 really means 1 km.
    projected_bbox = transform_bounds(
        "EPSG:4326",
        f"EPSG:{crs_epsg}",
        *[float(x) for x in bbox],
        densify_pts=21,
    )

    token = get_access_token()

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
#def download_ssm_patch(
#    bbox: list,
#    date: str,
#    output_path: str | Path,
#    crs_epsg: int = 32632,
#    resolution: int = 1000,
#) -> str:
#
#    token = get_access_token()
#
#    output_path = Path(output_path)
#    output_path.parent.mkdir(
#        parents=True,
#        exist_ok=True,
#    )
#################################
    request_json = {
        "input": {
            "bounds": {
                "bbox": list(projected_bbox),
                "properties": {
                    "crs": (
                        "http://www.opengis.net/"
                        f"def/crs/EPSG/0/{crs_epsg}"
                    )
                },
            },
            "data": [
                {
                    "type": SSM_DATA_TYPE,
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{date}T00:00:00Z",
                            "to": f"{date}T23:59:59Z",
                        }
                    },
                }
            ],
        },
        "output": {
            "resx": resolution,
            "resy": resolution,
            "responses": [
                {
                    "identifier": "default",
                    "format": {
                        "type": "image/tiff"
                    },
                }
            ],
        },
        "evalscript": _evalscript(),
    }

    response = requests.post(
        PROCESS_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "image/tiff",
        },
        json=request_json,
        timeout=120,
    )

    response.raise_for_status()

    output_path.write_bytes(response.content)

    return str(output_path)