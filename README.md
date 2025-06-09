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

Zusätzlich wird das MEXC SDK benötigt. Eine Kopie liegt bereits unter
`mexc/mexc_sdk/src` bei. Alternativ kannst du das offizielle Repository
klonen und lokal installieren:

```bash
git clone https://github.com/MEXCofficial/mexc-api-sdk.git
pip install -e mexc-api-sdk/dist/python
```

Wenn du die beigelegte Version verwendest, genügt es die Umgebungsvariable
`MEXC_SDK_PATH` auf `mexc/mexc_sdk/src` zu setzen. Das Skript prüft diesen
Pfad automatisch.

Alternativ kannst du den Pfad `mexc-api-sdk/dist/python` über die
Umgebungsvariable `PYTHONPATH` einbinden.

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

Für die MEXC-Integration werden die Schlüssel ebenfalls über Umgebungsvariablen
geladen. Lege dazu eine `.env`-Datei im Projektverzeichnis mit
`MEXC_API_KEY` und `MEXC_API_SECRET` an. Die Module laden diese Werte beim
Start automatisch.

### Regelmäßiges Finetuning

Um ein vorhandenes Modell weiter zu verbessern, kann `mexc/finetune_agent.py`
genutzt werden. Das Skript lädt den Agenten für das angegebene Handelspaar und
führt weitere Trainingsschritte aus.

```bash
python mexc/finetune_agent.py --symbol ATOMUSDC --timesteps 50000
```

Dieses Kommando lässt sich beispielsweise täglich per Cron ausführen, um den
Agenten aktuell zu halten.



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
