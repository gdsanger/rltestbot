import os
import sys
from typing import List

import json

import numpy as np
import pandas as pd

MEXC_SDK_PATH = os.environ.get("MEXC_SDK_PATH")
if MEXC_SDK_PATH and os.path.exists(MEXC_SDK_PATH):
    sys.path.insert(0, MEXC_SDK_PATH)
else:
    default_sdk = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "mexc-api-sdk", "dist", "python"
    )
    if os.path.exists(default_sdk):
        sys.path.insert(0, default_sdk)
    else:
        local_sdk = os.path.join(os.path.dirname(__file__), "mexc_sdk", "src")
        if os.path.exists(local_sdk):
            sys.path.insert(0, local_sdk)

from mexc_sdk import Spot
from dotenv import load_dotenv, find_dotenv


SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "config", "settings.yml")

# Load environment variables from a .env file if present
load_dotenv(find_dotenv())


def _load_api_credentials():
    """Load API key and secret from environment variables."""
    key = os.environ.get("MEXC_API_KEY") or os.environ.get("MEXC_KEY")
    secret = os.environ.get("MEXC_API_SECRET") or os.environ.get("MEXC_SECRET")
    if not key or not secret:
        raise EnvironmentError(
            "MEXC_API_KEY and MEXC_API_SECRET must be set as environment variables"
        )
    return key, secret


def fetch_recent_candles(
    symbol: str = "BTCUSDT",
    limit: int = 500,
    interval: str = "1m",
    return_df: bool = True,
):
    """Fetch recent OHLCV candles from MEXC."""

    # Recent changes to the MEXC API require authentication even for
    # public market data endpoints like ``klines``.  Load credentials if
    # available and fall back to an unauthenticated client otherwise.
    try:
        key, secret = _load_api_credentials()
        client = Spot(key, secret)
        use_auth = True
    except EnvironmentError:
        client = Spot()
        use_auth = False

    params = {"symbol": symbol.upper(), "interval": interval}
    if limit is not None:
        params["limit"] = limit

    if use_auth:
        # For authenticated requests we must explicitly sign the
        # query, otherwise the server returns an error.
        res = client.sign_request("GET", "/klines", params)
        klines = json.loads(res.getBody())
    else:
        klines = client.klines(symbol, interval, {"limit": limit})

    data: List[List[float]] = [
        [float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])]
        for k in klines
    ]

    if return_df:
        return pd.DataFrame(data, columns=["open", "high", "low", "close", "volume"])
    return np.array(data, dtype=np.float32)


def get_account_balance(asset: str = "USDT") -> float:
    """Return available balance for a given asset."""
    api_key, api_secret = _load_api_credentials()

    client = Spot(api_key, api_secret)
    account_info = client.account()

    for b in account_info.get("balances", []):
        if b.get("asset") == asset:
            return float(b.get("free", 0))
    return 0.0
