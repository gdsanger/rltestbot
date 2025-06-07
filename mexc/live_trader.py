import os
import time
import csv
from datetime import datetime
import yaml

from stable_baselines3 import PPO

from .mexc_env import MexcEnv
from .data_feeder import fetch_recent_candles, get_account_balance


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


def main():
    base_dir = os.path.dirname(__file__)
    settings_path = os.path.join(base_dir, "config", "settings.yml")
    with open(settings_path, "r") as f:
        settings = yaml.safe_load(f)

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
        models[symbol] = PPO.load(model_path)
        initial_data = fetch_recent_candles(symbol=symbol, limit=window_size, return_df=False)
        envs[symbol] = MexcEnv(symbol=symbol, window_size=window_size, data=initial_data, log_enabled=False)
        observations[symbol] = envs[symbol].reset()

    invest_ratio = settings.get("live_trading", {}).get("invest_ratio", 0.1)
    take_profit = settings.get("live_trading", {}).get("take_profit")
    stop_loss = settings.get("live_trading", {}).get("stop_loss")

    print("🚀 Starte Live-Handel (simuliert) auf MEXC...")
    while True:
        balance = get_account_balance("USDT")
        invest_amount = balance * invest_ratio
        for symbol, model in models.items():
            obs = observations[symbol]
            action, _ = model.predict(obs, deterministic=True)
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
                envs[symbol] = MexcEnv(symbol=symbol, window_size=window_size, data=new_data, log_enabled=False)
                observations[symbol] = envs[symbol].reset()

        time.sleep(60)


if __name__ == "__main__":
    main()
