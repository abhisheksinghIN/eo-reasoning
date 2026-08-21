from data.stac_client import CDSESTACClient


stac = CDSESTACClient()


def search_sentinel2(
    bbox: list,
    start_date: str,
    end_date: str,
    limit: int = 5,
):
    """
    Search CDSE for Sentinel-2 observations.

    Args:
        bbox: [min_lon, min_lat, max_lon, max_lat]
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
        limit: maximum number of results
    """

    items = stac.search(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        collections=["sentinel-2-l2a"],
        limit=limit,
    )

    return [
        {
            "id": item.id,
            "datetime": item.datetime.isoformat(),
            "assets": list(item.assets.keys()),
        }
        for item in items
    ]
