import requests
import time
import hmac
import hashlib
import os
from dotenv import load_dotenv
import urllib.parse

# .env laden
load_dotenv()

API_KEY = os.getenv("MEXC_API_KEY")
API_SECRET = os.getenv("MEXC_API_SECRET")

BASE_URL = "https://api.mexc.com"  # Spot-Trading API

# 🧠 Cache für Symbol-Spezifikationen
symbol_specs = {}

def sign(params: dict) -> str:
    query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
    return hmac.new(API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()

def fetch_symbol_specs(symbol: str):
    global symbol_specs
    if symbol in symbol_specs:
        return symbol_specs[symbol]

    response = requests.get(f"{BASE_URL}/api/v3/exchangeInfo")
    response.raise_for_status()
    data = response.json()

    for s in data.get("symbols", []):
        if s["symbol"] == symbol:
            qty_precision = 0
            price_precision = 0
            for f in s["filters"]:
                if f["filterType"] == "LOT_SIZE":
                    step_size = float(f["stepSize"])
                    qty_precision = abs(round(-1 * (step_size).as_integer_ratio()[1]).bit_length() - 1)
                elif f["filterType"] == "PRICE_FILTER":
                    tick_size = float(f["tickSize"])
                    price_precision = abs(round(-1 * (tick_size).as_integer_ratio()[1]).bit_length() - 1)

            symbol_specs[symbol] = {
                "qty_precision": qty_precision,
                "price_precision": price_precision
            }
            return symbol_specs[symbol]

    raise Exception(f"Symbol {symbol} not found in exchangeInfo")

def place_order(symbol: str, side: str, quantity: float, price: float = None):
    specs = fetch_symbol_specs(symbol)
    qty_precision = specs["qty_precision"]
    price_precision = specs["price_precision"]

    timestamp = int(time.time() * 1000)
    path = "/api/v3/order"
    url = BASE_URL + path

    params = {
        "symbol": symbol,
        "side": side.upper(),
        "type": "MARKET" if price is None else "LIMIT",
        "quantity": f"{quantity}",
        "timestamp": timestamp,
    }
    if price is not None:
        params["price"] = f"{price}"
        params["timeInForce"] = "GTC"

    query_string = urllib.parse.urlencode(sorted(params.items()))
    print(f"Placing order with params: {query_string}")
    signature = hmac.new(API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    full_url = f"{BASE_URL}/api/v3/order?{query_string}&signature={signature}"

    headers = {
        "X-MEXC-APIKEY": API_KEY
    }

    response = requests.post(full_url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Order Error: {response.status_code} - {response.text}")
    return response.json()
