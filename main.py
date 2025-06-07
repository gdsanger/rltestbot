import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from crypto_env import CryptoEnv
from data_fetcher import fetch_recent_candles


def train_and_test(total_timesteps: int = 10000, window_size: int = 60):
    # Fetch real market data from Binance
    data = fetch_recent_candles(limit=1000, return_df=False)

    # Create environment with real data
    env = DummyVecEnv([lambda: CryptoEnv(window_size=window_size, data=data, log_enabled=True)])

    # Train PPO agent
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save("ppo_cryptoenv_real")

    # Evaluate trained agent and collect equity curve
    obs = env.reset()
    done = False
    equity = [0.0]
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        equity.append(equity[-1] + reward[0])

    # Plot equity curve after training
    plt.figure(figsize=(10, 5))
    plt.plot(equity)
    plt.xlabel("Step")
    plt.ylabel("Equity")
    plt.title("Equity Curve")
    plt.tight_layout()
    plt.savefig("equity_curve.png")
    plt.show()
    env.envs[0].save_trade_log()


if __name__ == "__main__":
    train_and_test()
