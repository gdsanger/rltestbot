import numpy as np
import gym
from gym import spaces
from datetime import datetime
import pandas as pd
import pandas_ta as ta
from .data_feeder import fetch_klines


class MexcEnv(gym.Env):
    """Trading environment using OHLCV data from MEXC."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        window_size: int = 60,
        max_steps: int = 1000,
        data=None,
        log_enabled: bool = False,
        config=None,
        strategy: str = "macd_atr_stochrsi",
    ):
        super().__init__()
        self.symbol = symbol
        self.window_size = window_size
        self.external_data = data
        self.max_steps = max_steps if data is None else len(data) - window_size - 1
        self.log_enabled = log_enabled
        self.trade_log = []

        self.config = config or {}
        self.strategy = strategy

        rewards = self.config.get("rewards", {})
        self.reward_hold = rewards.get("hold", 0.0)
        self.reward_buy = rewards.get("buy", 1.0)
        self.reward_profit_multiplier = rewards.get("profit_multiplier", 10.0)
        self.reward_loss_multiplier = rewards.get("loss_multiplier", 20.0)

        self.base_columns = ["open", "high", "low", "close", "volume"]
        self.strategy_columns = self._strategy_columns()
        self.feature_columns = self.base_columns + self.strategy_columns

        # Observation now also contains the current position and the
        # unrealized profit so the agent is aware of the open trade state.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, len(self.feature_columns) + 2),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)
        self.data = None
        self.current_step = None
        self.position = None
        self.entry_price = None
        self.unrealized_profit = None

        self.feature_indices = {col: idx for idx, col in enumerate(self.feature_columns)}

    def _strategy_columns(self) -> list:
        if self.strategy == "ma_rsi":
            return ["rsi", "ma"]
        if self.strategy == "bollinger_bands":
            return ["bb_lower", "bb_middle", "bb_upper"]
        # default macd_atr_stochrsi
        return ["macd", "macd_signal", "macd_hist", "atr", "stochrsi"]

    def _generate_data(self):
        limit = self.max_steps + self.window_size + 50  # Puffer für Indikatoren
        raw_data = fetch_klines(symbol=self.symbol, limit=limit, interval="1m")

        df = pd.DataFrame(raw_data, columns=self.base_columns)
        df[self.base_columns] = df[self.base_columns].astype(float)

        if self.strategy == "ma_rsi":
            df["rsi"] = ta.rsi(df["close"], length=14)
            df["ma"] = ta.sma(df["close"], length=20)
        elif self.strategy == "bollinger_bands":
            bb = ta.bbands(df["close"])
            if isinstance(bb, pd.DataFrame):
                df["bb_lower"] = bb.iloc[:, 0]
                df["bb_middle"] = bb.iloc[:, 1]
                df["bb_upper"] = bb.iloc[:, 2]
        else:
            macd = ta.macd(df["close"])
            if isinstance(macd, pd.DataFrame):
                df["macd"] = macd.iloc[:, 0]
                df["macd_signal"] = macd.iloc[:, 1]
                df["macd_hist"] = macd.iloc[:, 2]
            else:
                df["macd"] = macd
                df["macd_signal"] = np.nan
                df["macd_hist"] = np.nan
            df["atr"] = ta.atr(df["high"], df["low"], df["close"])
            stochrsi = ta.stochrsi(df["close"])
            if isinstance(stochrsi, pd.DataFrame):
                df["stochrsi"] = stochrsi.iloc[:, 0]
            else:
                df["stochrsi"] = stochrsi

        df = df.dropna().reset_index(drop=True)
        result = df[self.feature_columns].dropna().values
        return result[-(self.max_steps + self.window_size):]

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
                self.data[self.current_step - self.window_size: self.current_step],
                dtype=np.float32,
            )
            position_feat = np.full((self.window_size, 1), self.position, dtype=np.float32)
            pnl_feat = np.full((self.window_size, 1), self.unrealized_profit, dtype=np.float32)
            obs = np.hstack((window, position_feat, pnl_feat))
            return obs, 0.0, True, {}

        candle = self.data[self.current_step]
        price = candle[self.feature_indices["close"]]
        reward = 0.0
        done = False
        realized_pnl = 0.0

        prev_unrealized = self.unrealized_profit

        if action == 1:  # BUY / Long
            if self.position != 1:
                if self.position == -1:
                    realized_pnl = self.entry_price - price
                self.position = 1
                self.entry_price = price
        elif action == 2:  # SELL / Short
            if self.position != -1:
                if self.position == 1:
                    realized_pnl = price - self.entry_price
                self.position = -1
                self.entry_price = price


        # ==== PNL-Anzeige ====
        if self.position == 1:
            self.unrealized_profit = price - self.entry_price
        elif self.position == -1:
            self.unrealized_profit = self.entry_price - price
        else:
            self.unrealized_profit = 0.0

        delta_unrealized = self.unrealized_profit - prev_unrealized
        if action == 0:
            reward = 0.0
        else:
            reward = delta_unrealized

        # ==== Logging ====
        if self.log_enabled:
            self.trade_log.append({
                "step": self.current_step,
                "timestamp": datetime.utcnow().isoformat(),
                "action": action,
                "price": price,
                "position": self.position,
                "realized_pnl": realized_pnl,
                "reward": reward,
            })

        self.current_step += 1
        window = np.array(
            self.data[self.current_step - self.window_size: self.current_step],
            dtype=np.float32,
        )
        position_feat = np.full((self.window_size, 1), self.position, dtype=np.float32)
        pnl_feat = np.full((self.window_size, 1), self.unrealized_profit, dtype=np.float32)
        obs = np.hstack((window, position_feat, pnl_feat))
        done = done or self.current_step >= len(self.data)
        info = {"position": self.position, "unrealized_profit": self.unrealized_profit}

        if self.current_step % 100 == 0:
            print(
                f"Step {self.current_step} | Reward: {reward:.4f} | Action: {action} | Price: {price:.4f} | PnL: {realized_pnl:.4f}"
            )

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

    def export_trade_log(self, filename="trade_log.csv"):
        import pandas as pd
        if self.trade_log:
            df = pd.DataFrame(self.trade_log)
            df.to_csv(filename, index=False)
            print(f"[i] Trade-Log gespeichert unter {filename}")
        else:
            print("[i] Kein Trade-Log zum Speichern vorhanden.")

    def calculate_sharpe_ratio(self, trades, risk_free_rate=0.0):
        import numpy as np
        returns = np.array([t["realized_pnl"] for t in trades if t["realized_pnl"] != 0])

        if len(returns) < 2:
            return 0.0  # nicht genug Trades

        excess_returns = returns - risk_free_rate
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns)

        if std_return == 0:
            return 0.0

        sharpe = mean_return / std_return
        return sharpe
    
    def analyze_trade_log(self, bins=10):
        import matplotlib.pyplot as plt
        import seaborn as sns

        if not self.trade_log:
            print("❌ Kein Trade-Log vorhanden.")
            return

        df = pd.DataFrame(self.trade_log)
        if df.empty or "price" not in df or "action" not in df:
            print("❌ Trade-Log unvollständig.")
            return

        # RSI & MA rekonstruieren aus Zeitpunkten
        analysis_data = self.data[[entry["step"] for entry in self.trade_log]]
        df["close"] = analysis_data[:, self.feature_indices["close"]]
        if "rsi" in self.feature_indices and "ma" in self.feature_indices:
            df["rsi"] = analysis_data[:, self.feature_indices["rsi"]]
            df["ma"] = analysis_data[:, self.feature_indices["ma"]]
        else:
            print("⚠️ Analyse nur für RSI/MA Strategie verfügbar")
            return

        # Heatmap: RSI vs. Aktion
        plt.figure(figsize=(10, 6))
        df["rsi_bin"] = pd.cut(df["rsi"], bins=bins)
        action_map = {0: "Hold", 1: "Buy", 2: "Sell"}
        df["action_label"] = df["action"].map(action_map)

        heat = pd.crosstab(df["rsi_bin"], df["action_label"])
        sns.heatmap(heat, annot=True, fmt="d", cmap="YlGnBu")
        plt.title(f"🔍 Action Count by RSI Range")
        plt.ylabel("RSI-Bereich")
        plt.xlabel("Aktion")
        plt.tight_layout()
        plt.show()

        # Optional: Profit nach RSI
        if "realized_pnl" in df:
            profit_per_rsi = df.groupby("rsi_bin")["realized_pnl"].sum()
            profit_per_rsi.plot(kind="bar", color="skyblue", title="💸 Profit nach RSI-Zone")
            plt.ylabel("Profit / Verlust")
            plt.tight_layout()
            plt.show()
