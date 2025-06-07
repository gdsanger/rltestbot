# 🧠 CryptoRL – Reinforcement Learning für Kryptowährungshandel

Ein experimentelles Projekt zum Trainieren eines Reinforcement-Learning-Agenten mit dem Ziel, selbstständig auf dem Binance-Testnet zu handeln.

## Features
- Binance-Testnet-Integration (BTC/USDT)
- PPO-Agent mit `stable-baselines3`
- Training & Evaluation direkt im Jupyter Notebook
- Live-Marktdaten abrufbar
- Custom Gym-Environment `BinanceTradingEnv` für Echtzeitdaten
- Logging der Trades in `trading_log.csv`

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
`train_agent.py` verwendet das vereinfachte `CryptoEnv`, welches nun echte
OHLCV-Daten vom Binance-Testnet nutzt, und speichert nach 100.000 Schritten das
Modell.

```bash
python train_agent.py
```

Vor dem Start sollten die Umgebungsvariablen `BINANCE_API_KEY` und
`BINANCE_API_SECRET` mit den Zugangsdaten des Testnet-Accounts gesetzt sein.



## Equity-Kurve berechnen

Das Modul `equity_curve.py` bietet die Funktion `calculate_equity_curve`, um eine
Portfolio-Kurve aus dem gespeicherten `trading_log.csv` zu erzeugen. Dabei wird
folgende Formel verwendet:

```
Equity = Startkapital + realisierte Gewinne + (offene Position × aktueller Preis - Einstiegspreis)
```

Beispiel:

```python
from equity_curve import calculate_equity_curve
curve = calculate_equity_curve("trading_log.csv")
curve.plot()
```

Die Close-Preise werden automatisch über `fetch_recent_candles` von Binance
abgerufen. Alternativ können eigene Preisdaten übergeben werden.

### Buy&Hold Benchmark

Um die Agent-Performance einordnen zu können, bietet `equity_curve.py` die
Funktion `compare_with_buy_and_hold`. Dabei wird zu Beginn das gesamte
Startkapital in BTC investiert und der Kursverlauf als Benchmark verfolgt.

```python
from equity_curve import compare_with_buy_and_hold
fig = compare_with_buy_and_hold("trading_log.csv", return_plot=True)
fig.show()
```

## Haftungsausschluss

**Dies ist kein Finanzrat. Nur zu Forschungs- und Lernzwecken.**
