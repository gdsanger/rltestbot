import requests
import threading

# Threadsafe Cache für Symbol-Präzisionen
_symbol_precision_cache = {}
_cache_lock = threading.Lock()

def get_symbol_precision(symbol: str):
    """
    Holt die baseAssetPrecision und baseSizePrecision (minimale Ordermenge) für ein Symbol.
    Gibt ein Tupel (decimal_places, min_quantity) zurück.
    """
    url = f"https://api.mexc.com/api/v3/exchangeInfo?symbol={symbol}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    if not data.get("symbols"):
        raise ValueError(f"Symbol {symbol} nicht gefunden in exchangeInfo")

    info = data["symbols"][0]
    base_precision = int(info.get("baseAssetPrecision", 6))
    min_qty = float(info.get("baseSizePrecision", "0.0001"))

    return base_precision, min_qty


def get_cached_symbol_precision(symbol: str):
    """
    Gibt cached basePrecision + minQty zurück oder holt sie einmalig.
    """
    with _cache_lock:
        if symbol not in _symbol_precision_cache:
            _symbol_precision_cache[symbol] = get_symbol_precision(symbol)
        return _symbol_precision_cache[symbol]


def adjust_quantity(symbol: str, quantity: float):
    """
    Rundet die Menge korrekt und stellt sicher, dass sie nicht unterhalb der minimalen Ordermenge liegt.
    """
    precision, min_qty = get_cached_symbol_precision(symbol)
    adjusted = round(quantity, precision)

    if adjusted < min_qty:
        print(f"[{symbol}] ⚠️ Gerundete Menge {adjusted} war kleiner als Mindestmenge {min_qty}. Verwende {min_qty}")
        return min_qty
    return adjusted