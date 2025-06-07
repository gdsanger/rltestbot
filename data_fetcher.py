import os
from typing import List

import numpy as np
import pandas as pd
from binance.client import Client


def fetch_recent_candles(
    symbol: str = "BTCUSDT",
    limit: int = 500,
    interval: str = Client.KLINE_INTERVAL_1MINUTE,
    return_df: bool = True,
    testnet: bool = False,
):
    """Fetch recent candles for a trading pair from Binance.

    Parameters
    ----------
    symbol : str
        Trading pair symbol, e.g. ``"BTCUSDT"``.
    limit : int
        Number of candles to fetch. Binance allows up to 1000.
    interval : str
        Kline interval constant from :class:`binance.client.Client`.
    return_df : bool
        If ``True``, return a :class:`pandas.DataFrame`; otherwise return a
        :class:`numpy.ndarray`.
    testnet : bool
        Whether to use the Binance Spot Testnet.

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        The OHLCV data in the requested format.
    """

    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")

    if testnet:
        try:
            client = Client(api_key, api_secret, testnet=True)
        except TypeError:
            client = Client(api_key, api_secret)
            client.API_URL = "https://testnet.binance.vision/api"
    else:
        client = Client(api_key, api_secret)

    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    data: List[List[float]] = [
        [float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in klines
    ]

    client.close_connection()

    if return_df:
        return pd.DataFrame(data, columns=["open", "high", "low", "close", "volume"])
    return np.array(data, dtype=np.float32)
