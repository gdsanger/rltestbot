import os
import time
import csv
from datetime import datetime
import yaml
import argparse

from stable_baselines3 import PPO
from .mexc_precision import adjust_quantity
from .mexc_env import MexcEnv
from .data_feeder import fetch_recent_candles, get_account_balance
from .mexc_api import place_order, fetch_symbol_specs

def init_csv(path: str):
    if not os.path.exists(path):
        with open(path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "symbol",
                "action",
                "price",
                "position",
                "unrealized_profit",
            ])


def log_trade(path: str, symbol: str, action: int, price: float, position: int, unrealized: float):
    with open(path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            symbol,
            action,
            price,
            position,
            unrealized,
        ])


def detect_strategy_from_model(model, strategies: dict, fallback: str) -> str:
    """Return the strategy name that matches the loaded model."""
    obs_dim = model.observation_space.shape[-1]
    indicator_count = obs_dim - 7  # 5 base cols + position + pnl
    for name, cfg in strategies.items():
        indicators = cfg.get("indicators", [])
        if len(indicators) == indicator_count:
            return name
    return fallback


def main():
    parser = argparse.ArgumentParser()
    base_dir = os.path.dirname(__file__)
    settings_path = os.path.join(base_dir, "config", "settings.yml")
    with open(settings_path, "r") as f:
        settings = yaml.safe_load(f)

    parser.add_argument(
        "--strategy",
        default=settings.get("default_strategy", "macd_atr_stochrsi"),
        help="Name der Trading-Strategie",
    )
    args = parser.parse_args()
    strategy = args.strategy

    window_size = settings.get("train", {}).get("window_size", 60)
    agents_dir = os.path.join(base_dir, "agents")
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "live_trades.csv")
    init_csv(csv_path)

    models = {}
    envs = {}
    observations = {}
    for symbol in settings.get("symbols", []):
        model_path = os.path.join(agents_dir, f"ppo_{symbol.lower()}")
        model = PPO.load(model_path)
        models[symbol] = model
        strategies_cfg = settings.get("strategies", {})
        detected = detect_strategy_from_model(model, strategies_cfg, strategy)
        if detected != strategy:
            print(
                f"[{symbol}] ⚠️ Using strategy '{detected}' to match loaded model (requested '{strategy}')"
            )
        env = MexcEnv(
            symbol=symbol,
            window_size=window_size,
            log_enabled=False,
            strategy=detected,
            config={"rewards": settings.get("rewards", {})},
        )
        obs = env.reset()
        envs[symbol] = env
        observations[symbol] = obs

    invest_ratio = settings.get("live_trading", {}).get("invest_ratio", 0.3)

    print("Lokale Zeit:", time.time())
    print("🚀 Starte Live-Handel (live) auf MEXC..." )
    print(f"Aktueller EUR-Kontostand: {get_account_balance('EUR')}")
    while True:
        balance = get_account_balance("EUR")
        invest_amount = balance * invest_ratio
       
        for symbol, model in models.items():
            obs = observations[symbol]
            action, _ = model.predict(obs, deterministic=True)
            price = envs[symbol].data[envs[symbol].current_step - 1][3]
            quantity_raw = invest_amount / price
            print(f"quantity_raw: {quantity_raw}")
            quantity = adjust_quantity(symbol, quantity_raw)
            print(f"quantity: {quantity}")
            # Order ausführen
            if action == 1:  # Buy
                base_asset = symbol.replace("EUR", "").replace("EUR", "")
                asset_balance = get_account_balance(base_asset)
                if asset_balance > 0:
                    print(f"[{symbol}] ⚠️ Bereits {asset_balance} {base_asset} vorhanden, überspringe Kauf")
                else:
                    try:
                        specs = fetch_symbol_specs(symbol)
                        use_market = "MARKET" in specs.get("order_types", [])
                        if use_market:
                            place_order(symbol=symbol, side="BUY", quantity=quantity)
                        else:
                            place_order(symbol=symbol, side="BUY", quantity=quantity, price=price)
                        print(f"[{symbol}] ✅ BUY Order ausgeführt")
                    except Exception as e:
                        print(f"[{symbol}] ❌ Fehler bei BUY: {e}")

            elif action == 2:  # Sell
                base_asset = symbol.replace("EUR", "").replace("EUR", "")
                asset_balance = get_account_balance(base_asset)
                print(f"[{symbol}] Kontostand {base_asset}: {asset_balance}")
                if asset_balance > 0:
                    sell_quantity = adjust_quantity(symbol, asset_balance)
                    try:
                        specs = fetch_symbol_specs(symbol)
                        use_market = "MARKET" in specs.get("order_types", [])
                        if use_market:
                            place_order(symbol=symbol, side="SELL", quantity=sell_quantity)
                        else:
                            place_order(symbol=symbol, side="SELL", quantity=sell_quantity, price=price)
                        print(f"[{symbol}] ✅ SELL Order ausgeführt")
                    except Exception as e:
                        print(f"[{symbol}] ❌ Fehler bei SELL: {e}")
                else:
                    print(f"[{symbol}] ⚠️ Kein Bestand zum Verkaufen")
            obs, _, done, info = envs[symbol].step(action)
            observations[symbol] = obs

            price = envs[symbol].data[envs[symbol].current_step - 1][3]
            position = info.get("position", envs[symbol].position)
            unrealized = info.get("unrealized_profit", envs[symbol].unrealized_profit)

            log_trade(csv_path, symbol, int(action), price, position, unrealized)
            print(
                f"[{symbol}] Action: {action} | Price: {price:.2f} | Position: {position} | P/L: {unrealized:.2f}"
            )

            if done:
                new_data = fetch_recent_candles(symbol=symbol, limit=window_size, return_df=False)
                current_strategy = envs[symbol].strategy
                envs[symbol] = MexcEnv(
                    symbol=symbol,
                    window_size=window_size,
                    data=new_data,
                    log_enabled=False,
                    strategy=current_strategy,
                    config={"rewards": settings.get("rewards", {})},
                )
                observations[symbol] = envs[symbol].reset()

        time.sleep(60)


if __name__ == "__main__":
    main()
