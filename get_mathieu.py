import os
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta
import time

# Clés API Binance (mettre None pour accès public)
client = Client(api_key=None, api_secret=None)

# Vérification et création du dossier "History"
if not os.path.exists("History"):
    os.makedirs("History")

# Définition des paires et des timeframes
paires = ["BTCUSDT", "ETHUSDT", "BCHUSDT", "SOLUSDT", "DOGEUSDT"]
timeframes = ["1h"]  # 1h = 1 heure, etc.

# Fonction pour récupérer l'History complet
def get_full_history(pair, timeframe, start_date="2017-01-01"):
    limit = 500
    all_data = []
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)

    while True:
        try:
            candles = client.get_klines(
                symbol=pair,
                interval=timeframe,
                startTime=start_ts,
                limit=limit
            )
        except Exception as e:
            print(f"Erreur pour {pair} {timeframe}: {e}")
            break

        if not candles:
            print(f"Aucune donnée trouvée pour {pair} {timeframe} à partir de {start_ts}")
            break

        all_data.extend(candles)

        # Calcul du prochain timestamp (ajouter 1 ms pour éviter doublon)
        last_ts = candles[-1][0]
        start_ts = last_ts + 1

        # Pause pour respecter les limitations d'API
        time.sleep(0.5)

    # Conversion en DataFrame
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])

    # Formatage des dates
    df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
    df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')

    return df

# Boucle sur les paires et les timeframes
for pair in paires:
    for tf in timeframes:
        print(f"Téléchargement en cours pour {pair} en timeframe {tf}")
        df = get_full_history(pair, tf)

        if df is not None and not df.empty:
            start_str = df["open_time"].min().strftime("%Y%m%d")
            end_str = df["open_time"].max().strftime("%Y%m%d")
            filename = f"History/{pair}_{tf}_{start_str}_{end_str}.csv"

            df.to_csv(filename, index=False)
            print(f"Fichier sauvegardé : {filename}")
        else:
            print(f"Aucune donnée à sauvegarder pour {pair} {tf}")

print("Téléchargement terminé !")