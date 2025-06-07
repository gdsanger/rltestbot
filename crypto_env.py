import numpy as np
import gym
from gym import spaces


class CryptoEnv(gym.Env):
    """Simple cryptocurrency trading environment with synthetic data."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, window_size: int = 60, max_steps: int = 1000, data=None):
        super().__init__()
        self.window_size = window_size
        self.external_data = data
        self.max_steps = max_steps if data is None else len(data) - window_size - 1
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
        price = 100.0
        data = []
        for _ in range(self.max_steps + self.window_size):
            change = np.random.randn() * 0.5
            open_p = price
            close = price + change
            high = max(open_p, close) + abs(np.random.randn())
            low = min(open_p, close) - abs(np.random.randn())
            volume = np.random.rand() * 100
            data.append([open_p, high, low, close, volume])
            price = close
        return data

    def reset(self):
        if self.external_data is not None:
            self.data = self.external_data
        else:
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
