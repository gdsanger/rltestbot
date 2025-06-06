import os
from collections import deque
from typing import Deque, List

import gym
import numpy as np
from gym import spaces
from binance.client import Client


class BinanceTradingEnv(gym.Env):
    """Custom Environment for BTC/USDT trading using live Binance data."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, symbol: str = "BTCUSDT", window_size: int = 60):
        super().__init__()
        self.symbol = symbol
        self.window_size = window_size

        api_key = os.environ.get("BINANCE_API_KEY")
        api_secret = os.environ.get("BINANCE_API_SECRET")
        self.client = Client(api_key, api_secret)

        # Observation space: OHLCV values for the last `window_size` minutes
        self.observation_space = spaces.Box(
            low=0,
            high=np.finfo(np.float32).max,
            shape=(self.window_size, 5),
            dtype=np.float32,
        )

        # Actions: 0 - Hold, 1 - Buy, 2 - Sell
        self.action_space = spaces.Discrete(3)

        self.data: Deque[List[float]] = deque(maxlen=self.window_size)
        self.position = 0  # -1 = short, 0 = flat, 1 = long
        self.entry_price = 0.0
        self.unrealized_profit = 0.0

    def _get_latest_candle(self) -> List[float]:
        """Fetch the latest 1 minute candle."""
        kline = self.client.get_klines(
            symbol=self.symbol,
            interval=Client.KLINE_INTERVAL_1MINUTE,
            limit=1,
        )[0]
        open_p, high, low, close, volume = (
            float(kline[1]),
            float(kline[2]),
            float(kline[3]),
            float(kline[4]),
            float(kline[5]),
        )
        return [open_p, high, low, close, volume]

    def _fetch_initial_data(self):
        klines = self.client.get_klines(
            symbol=self.symbol,
            interval=Client.KLINE_INTERVAL_1MINUTE,
            limit=self.window_size,
        )
        for k in klines:
            self.data.append(
                [float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])]
            )

    def reset(self):
        self.data.clear()
        self._fetch_initial_data()
        self.position = 0
        self.entry_price = 0.0
        self.unrealized_profit = 0.0
        return np.array(self.data, dtype=np.float32)

    def step(self, action: int):
        assert self.action_space.contains(action)

        candle = self._get_latest_candle()
        self.data.append(candle)
        price = candle[3]  # closing price

        prev_unrealized = self.unrealized_profit

        if action == 1:  # Buy
            if self.position != 1:
                self.position = 1
                self.entry_price = price
        elif action == 2:  # Sell
            if self.position != -1:
                self.position = -1
                self.entry_price = price

        if self.position == 1:
            self.unrealized_profit = price - self.entry_price
        elif self.position == -1:
            self.unrealized_profit = self.entry_price - price
        else:
            self.unrealized_profit = 0.0

        reward = self.unrealized_profit - prev_unrealized
        obs = np.array(self.data, dtype=np.float32)
        done = False
        info = {
            "position": self.position,
            "unrealized_profit": self.unrealized_profit,
        }
        return obs, reward, done, info

    def render(self, mode="human"):
        if mode == "human":
            last = self.data[-1] if self.data else [0, 0, 0, 0, 0]
            print(
                f"Close: {last[3]} | Position: {self.position} | Unrealized P/L: {self.unrealized_profit}"
            )

    def close(self):
        self.client.close_connection()
