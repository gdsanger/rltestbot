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

def fetch_recent_candles(symbol: str, limit: int = 60, return_df: bool = True):
    url = f"https://api.mexc.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": "1m",
        "limit": limit,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    raw_data = response.json()

    if return_df:
        df = pd.DataFrame(raw_data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df = df.astype({"open": float, "high": float, "low": float, "close": float})
        return df
    else:
        return [[float(k[1]), float(k[2]), float(k[3]), float(k[4])] for k in raw_data]

def get_account_balance(asset: str = "EUR") -> float:
    api_key = API_KEY
    api_secret = API_SECRET
    url = "https://api.mexc.com/api/v3/account"

    timestamp = int(time.time() * 1000)
    query_string = f"timestamp={timestamp}"
    signature = hmac.new(
        api_secret.encode(),
        query_string.encode(),
        hashlib.sha256
    ).hexdigest()

    headers = {
        "X-MEXC-APIKEY": api_key
    }

    full_url = f"{url}?{query_string}&signature={signature}"
    response = requests.get(full_url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"MEXC API error: {response.text}")

    balances = response.json()["balances"]
    for b in balances:
        if b["asset"] == asset:
            return float(b["free"])

    # Wenn der gesuchte Asset nicht gefunden wird, wird 0 zurückgegeben anstatt
    # eine Exception auszulösen. Dadurch kann z.B. beim Veräußern eines Assets,
    # das nicht vorhanden ist, die Anwendung trotzdem fortfahren.
    return 0.0
