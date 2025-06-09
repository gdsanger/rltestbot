import os
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from .mexc_env import MexcEnv
import pandas as pd
import numpy as np
from trade_analysis import summarize_trades


class EarlyStopCallback(BaseCallback):
    def __init__(self, reward_threshold: float = -50.0, check_freq: int = 1000, verbose=1):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold
        self.check_freq = check_freq
        self.rewards = []

    def _on_step(self) -> bool:
        if self.locals.get("rewards") is not None:
            self.rewards.append(self.locals["rewards"][-1])
        if self.n_calls % self.check_freq == 0 and len(self.rewards) >= self.check_freq:
            mean_reward = sum(self.rewards[-self.check_freq:]) / self.check_freq
            if self.verbose:
                print(f"🔍 Checkpoint: Mean reward of last {self.check_freq} steps: {mean_reward:.2f}")
            if mean_reward < self.reward_threshold:
                print(f"🚨 Früher Abbruch: Mean reward {mean_reward:.2f} < {self.reward_threshold}")
                return False
        return True


def main():
    base_path = os.path.dirname(__file__)
    settings_path = os.path.join(base_path, "config", "settings.yml")
    with open(settings_path, "r") as f:
        settings = yaml.safe_load(f)

    window_size = settings.get("train", {}).get("window_size", 60)
    timesteps = settings.get("train", {}).get("total_timesteps", 100_000)
    device = settings.get("train", {}).get("device", "auto")

    agents_dir = os.path.join(base_path, "agents")
    data_dir = os.path.join(base_path, "data")
    os.makedirs(agents_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    for symbol_entry in settings.get("symbols", []):
        if isinstance(symbol_entry, dict):
            symbol, config = list(symbol_entry.items())[0]
        else:
            symbol = symbol_entry
            config = {}

        print(f"🚀 Training startet für {symbol}")

        env = MexcEnv(
            symbol=symbol,
            window_size=window_size,
            log_enabled=True,
            config=config  # <== NEU: Konfig an die Umgebung übergeben
        )

        model = PPO("MlpPolicy", env, verbose=1, device=device)

        callback = EarlyStopCallback(reward_threshold=-100, check_freq=500)

        model.learn(total_timesteps=timesteps, callback=callback)
        model.save(os.path.join(agents_dir, f"ppo_{symbol.lower()}"))
        log_path = os.path.join(data_dir, f"training_log_{symbol}.csv")
        env.save_trade_log(log_path)

        summary = summarize_trades(log_path)
        print("\n===== Training Summary =====")
        print(f"Trades: {summary['total_trades']}")
        print(f"Gewinn-Trades: {summary['winning_trades']}")
        print(f"Verlust-Trades: {summary['losing_trades']}")
        print(f"Gesamter PnL: {summary['total_pnl']:.2f}")
        wl = summary['win_loss_ratio']
        ratio_str = f"{wl:.2f}" if wl not in (float('inf'), float('-inf')) else 'inf'
        print(f"Win/Loss-Verhältnis: {ratio_str}")

        env.analyze_trade_log()
        print(f"✅ Training abgeschlossen für {symbol}\n")

        # 📊 Sharpe Ratio berechnen
        try:
            df = pd.read_csv(os.path.join(data_dir, f"training_log_{symbol}.csv"))
            returns = df["realized_pnl"].dropna()
            returns = returns[returns != 0.0]  # Nur abgeschlossene Trades

            if len(returns) > 1:
                sharpe = returns.mean() / (returns.std() + 1e-9)
                print(f"📈 Sharpe Ratio für {symbol}: {sharpe:.4f}")
            else:
                print(f"⚠️ Nicht genug abgeschlossene Trades für Sharpe Ratio bei {symbol}")
        except Exception as e:
            print(f"❌ Fehler bei der Sharpe-Berechnung für {symbol}: {e}")



if __name__ == "__main__":
    main()
