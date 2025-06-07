import os
import time
import csv
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from crypto_env import CryptoEnv
from data_fetcher import fetch_recent_candles


CSV_PATH = "live_trades.csv"
MODEL_PATH = "ppo_cryptoenv"
SYMBOL = "BTCUSDT"
WINDOW_SIZE = 60
SLEEP_SECONDS = 60  # 1 Minute

def init_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "action", "price", "position", "unrealized_profit"])

def log_trade(action, price, position, unrealized):
    with open(CSV_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            action,
            price,
            position,
            unrealized
        ])

def main():
    print("⏳ Initialisiere Agent und Umgebung...")
    model = PPO.load(MODEL_PATH)

    initial_data = fetch_recent_candles(limit=WINDOW_SIZE, return_df=False, testnet=True)
    env = CryptoEnv(window_size=WINDOW_SIZE, data=initial_data, log_enabled=False)

    obs = env.reset()
    init_csv()
    print("🚀 Starte Live-Handel im Testnet...")

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        price = env.data[-1][3]
        log_trade(action, price, info["position"], info["unrealized_profit"])

        print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] Action: {action} | "
              f"Price: {price:.2f} | Position: {info['position']} | P/L: {info['unrealized_profit']:.2f}")

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
