"""Thin tool wrapper around CDSE/STAC discovery."""

from data.sentinel2 import search_sentinel2


def search_sentinel2_tool(
    bbox: list,
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 30.0,
    limit: int = 10,
) -> dict:
    """Find Sentinel-2 L2A observations for an AOI and date range."""
    items = search_sentinel2(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        max_cloud_cover=max_cloud_cover,
        limit=limit,
    )
    return {
        "query": {
            "bbox": bbox,
            "start_date": start_date,
            "end_date": end_date,
            "max_cloud_cover": max_cloud_cover,
        },
        "count": len(items),
        "items": items,
    }
