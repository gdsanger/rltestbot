import numpy as np
import gym
from gym import spaces
from binance.client import Client
from data_fetcher import fetch_recent_candles


class CryptoEnv(gym.Env):
    """Trading environment using OHLCV data from the Binance Spot Testnet."""

    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 window_size: int = 60,
                 max_steps: int = 1000,
                 symbol: str = "BTCUSDT",
                 testnet: bool = True):
        super().__init__()
        self.window_size = window_size
        self.max_steps = max_steps
        self.symbol = symbol
        self.testnet = testnet
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, 5),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)
        self.data = None
        self.current_step = None
        self.position = None
        self.entry_price = None
        self.unrealized_profit = None

    def _generate_data(self):
        """Fetch historical candles from Binance testnet."""
        limit = self.max_steps + self.window_size
        data = fetch_recent_candles(
            symbol=self.symbol,
            limit=limit,
            interval=Client.KLINE_INTERVAL_1MINUTE,
            return_df=False,
            testnet=self.testnet,
        )
        return data.tolist()

    def reset(self):
        self.data = self._generate_data()
        self.current_step = self.window_size
        self.position = 0
        self.entry_price = 0.0
        self.unrealized_profit = 0.0
        return np.array(
            self.data[self.current_step - self.window_size : self.current_step],
            dtype=np.float32,
        )

    def step(self, action: int):
        assert self.action_space.contains(action)

        if self.current_step >= len(self.data) - 1:
            obs = np.array(
                self.data[self.current_step - self.window_size : self.current_step],
                dtype=np.float32,
            )
            return obs, 0.0, True, {}

        candle = self.data[self.current_step]
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
        self.current_step += 1
        obs = np.array(
            self.data[self.current_step - self.window_size : self.current_step],
            dtype=np.float32,
        )
        done = self.current_step >= len(self.data)
        info = {"position": self.position, "unrealized_profit": self.unrealized_profit}
        return obs, reward, done, info

    def render(self, mode="human"):
        if mode == "human":
            last = self.data[self.current_step - 1]
            print(
                f"Close: {last[3]} | Position: {self.position} | Unrealized P/L: {self.unrealized_profit}"
            )

    def close(self):
        pass
