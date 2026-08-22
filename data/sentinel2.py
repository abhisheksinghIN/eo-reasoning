"""Sentinel-2 discovery and patch retrieval through CDSE."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List

import requests

from data.cdse_auth import get_access_token
from data.stac_client import CDSESTACClient

PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"
OUTPUT_BANDS = [
    "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B11", "SCL", "dataMask",
]


def search_sentinel2(
    bbox: list,
    start_date: str,
    end_date: str,
    limit: int = 10,
    max_cloud_cover: float = 30.0,
):
    stac = CDSESTACClient()
    items = stac.search(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        collections=["sentinel-2-l2a"],
        limit=limit,
        max_cloud_cover=max_cloud_cover,
    )

    return [
        {
            "id": item.id,
            "datetime": item.datetime.isoformat() if item.datetime else None,
            "date": item.datetime.date().isoformat() if item.datetime else None,
            "cloud_cover": item.properties.get("eo:cloud_cover"),
            "assets": list(item.assets.keys()),
        }
        for item in items
    ]


def _evalscript() -> str:
    return """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02","B03","B04","B05","B06","B07",
              "B08","B11","SCL","dataMask"],
      units: ["DN","DN","DN","DN","DN","DN",
              "DN","DN","DN","DN"]
    }],
    output: {
      bands: 10,
      sampleType: "UINT16"
    }
  };
}

function evaluatePixel(s) {
  return [
    s.B02, s.B03, s.B04, s.B05, s.B06, s.B07,
    s.B08, s.B11, s.SCL, s.dataMask
  ];
}
"""


def _cache_name(bbox: Iterable[float], date: str, width: int, height: int) -> str:
    key = f"{list(bbox)}|{date}|{width}|{height}".encode("utf-8")
    digest = hashlib.sha1(key).hexdigest()[:12]
    return f"s2_{date}_{digest}.tif"


def download_sentinel2_patch(
    bbox: list,
    date: str,
    output_dir: str | Path = ".cache/sentinel2",
    width: int = 224,
    height: int = 224,
    force: bool = False,
) -> str:
    if len(bbox) != 4:
        raise ValueError("bbox must be [min_lon, min_lat, max_lon, max_lat].")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _cache_name(bbox, date, width, height)

    if out_path.exists() and not force:
        return str(out_path)

    token = get_access_token()

    request_json = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                },
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{date}T00:00:00Z",
                            "to": f"{date}T23:59:59Z",
                        },
                        "mosaickingOrder": "leastCC",
                    },
                    "processing": {"harmonizeValues": "true"},
                }
            ],
        },
        "output": {
            "width": int(width),
            "height": int(height),
            "responses": [
                {
                    "identifier": "default",
                    "format": {"type": "image/tiff"},
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

    content_type = response.headers.get("content-type", "")
    if "tiff" not in content_type.lower() and len(response.content) < 1024:
        raise RuntimeError(
            f"Unexpected Process API response: {content_type}: "
            f"{response.text[:300]}"
        )

    out_path.write_bytes(response.content)
    return str(out_path)


def download_temporal_stack(
    bbox: list,
    dates: List[str],
    output_dir: str | Path = ".cache/sentinel2",
    width: int = 224,
    height: int = 224,
) -> List[str]:
    if len(dates) != 3:
        raise ValueError(
            "Prithvi-EO-1.0-100M was trained with three temporal frames; "
            "provide exactly three dates for this MVP."
        )

    return [
        download_sentinel2_patch(
            bbox=bbox,
            date=date,
            output_dir=output_dir,
            width=width,
            height=height,
        )
        for date in dates
    ]
