from stable_baselines3 import PPO
from crypto_env import CryptoEnv
from data_fetcher import fetch_recent_candles
from equity_curve import calculate_equity_curve, compare_with_buy_and_hold


def evaluate(model_path: str = "ppo_cryptoenv", steps: int = 500, window_size: int = 60) -> None:
    """Load a trained agent and evaluate it on fresh market data."""
    data = fetch_recent_candles(limit=steps + window_size, return_df=False)
    env = CryptoEnv(window_size=window_size, data=data, log_enabled=True)

    model = PPO.load(model_path)

    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)

    env.save_trade_log()

    equity = calculate_equity_curve("trading_log.csv")
    print(f"Final equity: {equity.iloc[-1]:.2f}")

    fig = compare_with_buy_and_hold("trading_log.csv", return_plot=True)
    fig.savefig("equity_vs_bh.png")


if __name__ == "__main__":
    evaluate()
