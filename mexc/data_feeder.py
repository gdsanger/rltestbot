# mexc_data_feeder.py

import os
import hmac
import hashlib
import time
import requests
from urllib.parse import urlencode
from dotenv import load_dotenv

# .env laden
load_dotenv()

API_KEY = os.getenv("MEXC_API_KEY")
API_SECRET = os.getenv("MEXC_API_SECRET")

BASE_URL = "https://api.mexc.com"

HEADERS = {
    "Content-Type": "application/json",
    "X-MEXC-APIKEY": API_KEY
}


def _sign_params(params: dict) -> dict:
    """Signiere die Parameter für private API-Calls"""
    query_string = urlencode(params)
    signature = hmac.new(
        API_SECRET.encode("utf-8"),
        query_string.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    params["signature"] = signature
    return params


def get_account_info() -> dict:
    """Hole Kontoinformationen"""
    path = "/api/v3/account"
    timestamp = int(time.time() * 1000)
    params = {
        "timestamp": timestamp
    }
    signed_params = _sign_params(params)
    response = requests.get(f"{BASE_URL}{path}", headers=HEADERS, params=signed_params)
    response.raise_for_status()
    return response.json()


def fetch_klines(symbol: str, interval: str = "1m", limit: int = 1000):
    url = "https://api.mexc.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    # Format: [open_time, open, high, low, close, volume, ...]
    ohlcv = [[float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] for c in data]
    return ohlcv


# --- Test ---
if __name__ == "__main__":
    print("Kontoinformationen:")
    account = get_account_info()
    print(account)

    print("\nKlines BTC/USDT:")
    candles = fetch_klines("BTCUSDT", "1m", 5)
    for c in candles:
        print(c)
