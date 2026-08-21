"""OAuth2 authentication for Copernicus Data Space Ecosystem."""

from __future__ import annotations

import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

_TOKEN: Optional[str] = None
_TOKEN_EXPIRES_AT: float = 0.0


def get_access_token(force_refresh: bool = False) -> str:
    """Return a reusable CDSE access token."""
    global _TOKEN, _TOKEN_EXPIRES_AT

    now = time.time()
    if not force_refresh and _TOKEN and now < (_TOKEN_EXPIRES_AT - 60):
        return _TOKEN

    client_id = os.getenv("CDSE_CLIENT_ID")
    client_secret = os.getenv("CDSE_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError(
            "CDSE_CLIENT_ID and CDSE_CLIENT_SECRET are required. "
            "Copy .env.example to .env and fill the credentials."
        )

    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    _TOKEN = payload["access_token"]
    _TOKEN_EXPIRES_AT = now + float(payload.get("expires_in", 600))
    return _TOKEN
