#"""Small CDSE STAC client with rate-limit handling."""

from __future__ import annotations

import random
import time
from typing import Iterable, Optional

from pystac_client.exceptions import APIError


DEFAULT_STAC_URL = "https://stac.dataspace.copernicus.eu/v1/"


class CDSESTACClient:
    def __init__(self, url: str = DEFAULT_STAC_URL):
        from pystac_client import Client

        self.catalog = Client.open(url)

    def _execute_search(
        self,
        kwargs: dict,
        max_attempts: int = 5,
    ):
        """
        Execute STAC search with exponential backoff for HTTP 429.
        """

        for attempt in range(max_attempts):
            try:
                search = self.catalog.search(**kwargs)
                return list(search.items())

            except APIError as exc:
                status = getattr(exc, "status_code", None)

                if status != 429:
                    raise

                if attempt == max_attempts - 1:
                    raise RuntimeError(
                        "CDSE STAC rate limit persisted after "
                        f"{max_attempts} attempts."
                    ) from exc

                # 1, 2, 4, 8 ... seconds + small random jitter.
                delay = min(2**attempt, 16)
                delay += random.uniform(0.0, 0.5)

                print(
                    f"CDSE STAC rate limited (HTTP 429). "
                    f"Retrying in {delay:.1f}s "
                    f"[{attempt + 1}/{max_attempts}]..."
                )

                time.sleep(delay)

        raise RuntimeError(
            "Unexpected failure while executing CDSE STAC search."
        )

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
            kwargs["query"] = {
                "eo:cloud_cover": {
                    "lt": float(max_cloud_cover)
                }
            }

        try:
            items = self._execute_search(kwargs)

        except APIError as exc:
            status = getattr(exc, "status_code", None)

            # Only fall back if the server rejects the query syntax.
            # Do NOT fall back for 429 or server failures.
            if (
                max_cloud_cover is None
                or status not in {400, 422}
            ):
                raise

            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("query", None)

            items = self._execute_search(
                fallback_kwargs
            )

            items = [
                item
                for item in items
                if float(
                    item.properties.get(
                        "eo:cloud_cover",
                        100.0,
                    )
                )
                < float(max_cloud_cover)
            ]

        return items[:limit]


##"""Small CDSE STAC client."""
#
#from __future__ import annotations
#
#from typing import Iterable, Optional
#
#DEFAULT_STAC_URL = "https://stac.dataspace.copernicus.eu/v1/"
#
#
#class CDSESTACClient:
#    def __init__(self, url: str = DEFAULT_STAC_URL):
#        from pystac_client import Client
#        self.catalog = Client.open(url)
#
#    def search(
#        self,
#        bbox,
#        start_date: str,
#        end_date: str,
#        collections: Iterable[str],
#        limit: int = 10,
#        max_cloud_cover: Optional[float] = None,
#    ):
#        datetime_range = (
#            f"{start_date}T00:00:00Z/"
#            f"{end_date}T23:59:59Z"
#        )
#
#        kwargs = {
#            "bbox": bbox,
#            "datetime": datetime_range,
#            "collections": list(collections),
#            "limit": limit,
#        }
#
#        if max_cloud_cover is not None:
#            kwargs["query"] = {"eo:cloud_cover": {"lt": float(max_cloud_cover)}}
#
#        try:
#            search = self.catalog.search(**kwargs)
#            items = list(search.items())
#        except Exception:
#            kwargs.pop("query", None)
#            search = self.catalog.search(**kwargs)
#            items = list(search.items())
#            if max_cloud_cover is not None:
#                items = [
#                    item
#                    for item in items
#                    if float(item.properties.get("eo:cloud_cover", 100.0))
#                    < float(max_cloud_cover)
#                ]
#
#        return items[:limit]
