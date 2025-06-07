import os
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from .mexc_env import MexcEnv


class EarlyStopCallback(BaseCallback):
    def __init__(self, reward_threshold: float = -50.0, check_freq: int = 5, verbose=1):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            ep_info = self.locals.get("infos", [{}])[-1]
            mean_reward = ep_info.get("episode", {}).get("r", None)
            if mean_reward is not None and mean_reward < self.reward_threshold:
                print(f"🚨 Abbruch: Mean reward {mean_reward} < {self.reward_threshold}")
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

    for symbol in settings.get("symbols", []):
        print(f"🚀 Training startet für {symbol}")
        env = MexcEnv(symbol=symbol, window_size=window_size, log_enabled=True)

        model = PPO("MlpPolicy", env, verbose=1, device=device)

        callback = EarlyStopCallback(reward_threshold=-100, check_freq=500)

        model.learn(total_timesteps=timesteps, callback=callback)
        model.save(os.path.join(agents_dir, f"ppo_{symbol.lower()}"))
        env.save_trade_log(os.path.join(data_dir, f"training_log_{symbol}.csv"))
        print(f"✅ Training abgeschlossen für {symbol}\n")


if __name__ == "__main__":
    main()
