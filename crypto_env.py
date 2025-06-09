import numpy as np
import gym
from gym import spaces
from binance.client import Client
from data_fetcher import fetch_recent_candles
from datetime import datetime
import pandas as pd


class CryptoEnv(gym.Env):
    """Trading environment using OHLCV data from the Binance Spot Testnet."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, window_size: int = 60, max_steps: int = 1000, data=None, log_enabled: bool = False):
        super().__init__()
        self.window_size = window_size
        self.external_data = data
        self.max_steps = max_steps if data is None else len(data) - window_size - 1
        self.log_enabled = log_enabled
        self.trade_log = []

        # Observation includes OHLCV data plus current position and
        # unrealized profit so the agent knows the state of its open trade.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, 7),
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
        if self.external_data is not None:
            self.data = self.external_data
        else:
            self.data = self._generate_data()
        self.current_step = self.window_size
        self.position = 0
        self.entry_price = 0.0
        self.unrealized_profit = 0.0
        if self.log_enabled:
            self.trade_log = []
        window = np.array(
            self.data[self.current_step - self.window_size : self.current_step],
            dtype=np.float32,
        )
        position_feat = np.full((self.window_size, 1), self.position, dtype=np.float32)
        pnl_feat = np.full((self.window_size, 1), self.unrealized_profit, dtype=np.float32)
        return np.hstack((window, position_feat, pnl_feat))

    def step(self, action: int):
        assert self.action_space.contains(action)

        if self.current_step >= len(self.data) - 1:
            window = np.array(
                self.data[self.current_step - self.window_size : self.current_step],
                dtype=np.float32,
            )
            position_feat = np.full((self.window_size, 1), self.position, dtype=np.float32)
            pnl_feat = np.full((self.window_size, 1), self.unrealized_profit, dtype=np.float32)
            obs = np.hstack((window, position_feat, pnl_feat))
            return obs, 0.0, True, {}

        candle = self.data[self.current_step]
        price = candle[3]  # closing price
        prev_unrealized = self.unrealized_profit
        realized_pnl = 0.0

        if action == 1:  # Buy
            if self.position != 1:
                if self.position == -1:
                    realized_pnl = self.entry_price - price
                self.position = 1
                self.entry_price = price
        elif action == 2:  # Sell
            if self.position != -1:
                if self.position == 1:
                    realized_pnl = price - self.entry_price
                self.position = -1
                self.entry_price = price

        if self.position == 1:
            self.unrealized_profit = price - self.entry_price
        elif self.position == -1:
            self.unrealized_profit = self.entry_price - price
        else:
            self.unrealized_profit = 0.0

        delta_unrealized = self.unrealized_profit - prev_unrealized
        if action == 0:
            if delta_unrealized > 0:
                reward = 2 * delta_unrealized
            elif delta_unrealized < 0:
                reward = 5 * delta_unrealized
            else:
                reward = 0.0
        else:
            reward = delta_unrealized

        if self.log_enabled:
            self.trade_log.append(
                {
                    "step": self.current_step,
                    "timestamp": datetime.utcnow().isoformat(),
                    "action": action,
                    "price": price,
                    "position": self.position,
                    "realized_pnl": realized_pnl,
                }
            )

        self.current_step += 1
        window = np.array(
            self.data[self.current_step - self.window_size : self.current_step],
            dtype=np.float32,
        )
        position_feat = np.full((self.window_size, 1), self.position, dtype=np.float32)
        pnl_feat = np.full((self.window_size, 1), self.unrealized_profit, dtype=np.float32)
        obs = np.hstack((window, position_feat, pnl_feat))
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

    def save_trade_log(self, path: str = "trading_log.csv"):
        if self.log_enabled and self.trade_log:
            pd.DataFrame(self.trade_log).to_csv(path, index=False)
