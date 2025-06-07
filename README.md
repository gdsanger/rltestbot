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

### Training eines PPO-Agenten

Neben dem Notebook kann ein Agent auch per Skript trainiert werden. Das Skript
`train_agent.py` verwendet das vereinfachte `CryptoEnv` und speichert nach
100.000 Schritten das Modell.

```bash
python train_agent.py
```


## Haftungsausschluss

**Dies ist kein Finanzrat. Nur zu Forschungs- und Lernzwecken.**
