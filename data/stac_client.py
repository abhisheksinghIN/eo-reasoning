from datetime import datetime
from pystac_client import Client


class CDSESTACClient:

    def __init__(
        self,
        url="https://stac.dataspace.copernicus.eu/v1/"
    ):
        self.catalog = Client.open(url)

    def search(
        self,
        bbox,
        start_date,
        end_date,
        collections,
        limit=10,
    ):
        datetime_range = (
            f"{start_date}T00:00:00Z/"
            f"{end_date}T23:59:59Z"
        )

        search = self.catalog.search(
            bbox=bbox,
            datetime=datetime_range,
            collections=collections,
            limit=limit,
        )

        return list(search.items())
