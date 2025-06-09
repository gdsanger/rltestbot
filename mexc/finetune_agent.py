import os
import yaml
import argparse
from stable_baselines3 import PPO
from .mexc_env import MexcEnv
from trade_analysis import summarize_trades


def main():
    parser = argparse.ArgumentParser()
    base_dir = os.path.dirname(__file__)
    settings_path = os.path.join(base_dir, "config", "settings.yml")
    with open(settings_path, "r") as f:
        settings = yaml.safe_load(f)

    parser.add_argument("--symbol", required=True, help="Handelspaar, z.B. ATOMUSDC")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="Anzahl zusätzlicher Training-Schritte",
    )
    parser.add_argument(
        "--strategy",
        default=settings.get("default_strategy", "macd_atr_stochrsi"),
        help="Trading-Strategie",
    )
    args = parser.parse_args()

    symbol = args.symbol.upper()
    agents_dir = os.path.join(base_dir, "agents")
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(agents_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    model_path = os.path.join(agents_dir, f"ppo_{symbol.lower()}")
    log_path = os.path.join(data_dir, f"training_log_{symbol}.csv")

    print(f"🔄 Finetuning {symbol} für {args.timesteps} Schritte")

    env = MexcEnv(
        symbol=symbol,
        window_size=settings.get("train", {}).get("window_size", 60),
        log_enabled=True,
        strategy=args.strategy,
        config={"rewards": settings.get("rewards", {})},
    )

    model = PPO.load(model_path, env=env)
    model.learn(total_timesteps=args.timesteps)
    model.save(model_path)

    env.save_trade_log(log_path)

    summary = summarize_trades(log_path, start_step=args.timesteps // 2)
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
