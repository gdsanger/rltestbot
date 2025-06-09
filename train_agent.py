from stable_baselines3 import PPO
from crypto_env import CryptoEnv
from trade_analysis import summarize_trades


def main():
    env = CryptoEnv(log_enabled=True)
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=100_000)
    model.save("ppo_cryptoenv")
    env.save_trade_log()
    summary = summarize_trades()
    print("\n===== Training Summary =====")
    print(f"Trades: {summary['total_trades']}")
    print(f"Gewinn-Trades: {summary['winning_trades']}")
    print(f"Verlust-Trades: {summary['losing_trades']}")
    print(f"Gesamter PnL: {summary['total_pnl']:.2f}")
    wl = summary['win_loss_ratio']
    ratio_str = f"{wl:.2f}" if wl not in (float('inf'), float('-inf')) else 'inf'
    print(f"Win/Loss-Verhältnis: {ratio_str}")


if __name__ == "__main__":
    main()
