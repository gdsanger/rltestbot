from stable_baselines3 import PPO
from crypto_env import CryptoEnv


def main():
    env = CryptoEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000)
    model.save("ppo_cryptoenv")


if __name__ == "__main__":
    main()
