"""Small CDSE STAC client."""

from __future__ import annotations

from typing import Iterable, Optional

DEFAULT_STAC_URL = "https://stac.dataspace.copernicus.eu/v1/"


class CDSESTACClient:
    def __init__(self, url: str = DEFAULT_STAC_URL):
        from pystac_client import Client
        self.catalog = Client.open(url)

    def search(
        self,
        bbox,
        start_date: str,
        end_date: str,
        collections: Iterable[str],
        limit: int = 10,
        max_cloud_cover: Optional[float] = None,
    ):
        datetime_range = (
            f"{start_date}T00:00:00Z/"
            f"{end_date}T23:59:59Z"
        )

        kwargs = {
            "bbox": bbox,
            "datetime": datetime_range,
            "collections": list(collections),
            "limit": limit,
        }

        if max_cloud_cover is not None:
            kwargs["query"] = {"eo:cloud_cover": {"lt": float(max_cloud_cover)}}

        try:
            search = self.catalog.search(**kwargs)
            items = list(search.items())
        except Exception:
            kwargs.pop("query", None)
            search = self.catalog.search(**kwargs)
            items = list(search.items())
            if max_cloud_cover is not None:
                items = [
                    item
                    for item in items
                    if float(item.properties.get("eo:cloud_cover", 100.0))
                    < float(max_cloud_cover)
                ]

        return items[:limit]
