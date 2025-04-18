import requests
import pandas as pd
from datetime import datetime




def get_binance_feed(symbol, interval, start_time):
    response = requests.get(f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={start_time}&limit=1000")
    data = response.json()
    return data

# initialize df
df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume","ignore"])


for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
    interval = "1h"
    # Start time: January 1, 2017 (in milliseconds)
    start_time = 1483225200000



    data = get_binance_feed(symbol, interval, start_time)

    # 1499040000000,      // Open time
    # "0.01634790",       // Open
    # "0.80000000",       // High
    # "0.01575800",       // Low
    # "0.01577100",       // Close
    # "148976.11427815",  // Volume
    # 1499644799999,      // Close time
    # "2434.19055334",    // Quote asset volume
    # 308,                // Number of trades
    # "1756.87402397",    // Taker buy base asset volume
    # "28.46694368",      // Taker buy quote asset volume
    # "17928899.62484339" // Ignore.



    df = pd.concat([df, pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume","ignore"])])

    df.to_csv(f"{symbol}_{interval}.csv", index=False)
    # Current timestamp in milliseconds
    today_time = int(datetime.now().timestamp() * 1000)
    
    # Make it run until the last timestamp is inferior to today's timestamp
    while float(data[-1][0]) < today_time:
        # print(f"Fetching more data for {symbol}... Current timestamp: {datetime.fromtimestamp(float(data[-1][0])/1000)}")
        data = get_binance_feed(symbol, interval, int(float(data[-1][0])))
        df = pd.concat([df, pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume","ignore"])])

    # drop ignore column
    df.drop(columns=["ignore"], inplace=True)

    df.to_csv(f"{symbol}_{interval}.csv", index=False)





