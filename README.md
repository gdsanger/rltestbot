# 🧠 CryptoRL – Reinforcement Learning für Kryptowährungshandel

Ein experimentelles Projekt zum Trainieren eines Reinforcement-Learning-Agenten mit dem Ziel, selbstständig auf dem Binance-Testnet zu handeln.

## Features
- Binance-Testnet-Integration (BTC/USDT)
- PPO-Agent mit `stable-baselines3`
- Training & Evaluation direkt im Jupyter Notebook
- Live-Marktdaten abrufbar
- Custom Gym-Environment `BinanceTradingEnv` für Echtzeitdaten

## Voraussetzungen

```bash
pip install -r requirements.txt
```

## Start

```bash
jupyter lab
```

Dann öffne `CryptoRL_Starter_with_Agent.ipynb` und folge den Schritten.
### Nutzung des Environments

```python
from binance_env import BinanceTradingEnv
env = BinanceTradingEnv()
obs = env.reset()
```


## Haftungsausschluss

**Dies ist kein Finanzrat. Nur zu Forschungs- und Lernzwecken.**
