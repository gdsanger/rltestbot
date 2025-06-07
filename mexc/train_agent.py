import os
import yaml
from stable_baselines3 import PPO

from .mexc_env import MexcEnv


def main():
    settings_path = os.path.join(os.path.dirname(__file__), "config", "settings.yml")
    with open(settings_path, "r") as f:
        settings = yaml.safe_load(f)

    window_size = settings.get("train", {}).get("window_size", 60)
    timesteps = settings.get("train", {}).get("total_timesteps", 100000)
    agents_dir = os.path.join(os.path.dirname(__file__), "agents")
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(agents_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    for symbol in settings.get("symbols", []):
        env = MexcEnv(symbol=symbol, window_size=window_size, log_enabled=True)
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=timesteps)
        model.save(os.path.join(agents_dir, f"ppo_{symbol.lower()}"))
        env.save_trade_log(os.path.join(data_dir, f"training_log_{symbol}.csv"))


if __name__ == "__main__":
    main()
