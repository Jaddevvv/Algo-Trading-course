from binance import Client
import pandas as pd
from datetime import datetime

# Initialize Binance client
client = Client(None, None)

# Initialize DataFrame
df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore"])

for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
    print(f"Fetching data for {symbol}")
    # Start time: January 1, 2017 (in milliseconds)
    start_time = 1483225200000
    
    # Get current timestamp in milliseconds
    today_time = int(datetime.now().timestamp() * 1000)
    
    # Get initial data
    klines = client.get_klines(
        symbol=symbol,
        interval=Client.KLINE_INTERVAL_1HOUR,
        limit=1000,
        startTime=start_time
    )
    
    # Convert to DataFrame and concatenate
    temp_df = pd.DataFrame(
        klines,
        columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", 
                "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
    )
    df = pd.concat([df, temp_df])
    
    # Save initial data
    df.to_csv(f"{symbol}_1h_2.csv", index=False)
    
    # Continue fetching until we reach current time
    while float(klines[-1][0]) < today_time:
        
        # Get next batch starting from last timestamp
        klines = client.get_klines(
            symbol=symbol,
            interval=Client.KLINE_INTERVAL_1HOUR,
            limit=1000,
            startTime=int(float(klines[-1][0]))
        )
        
        # Convert to DataFrame and concatenate
        temp_df = pd.DataFrame(
            klines,
            columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", 
                    "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
        )
        df = pd.concat([df, temp_df])
        
        # Save progress
        df.to_csv(f"{symbol}_1h_2.csv", index=False)

    # Drop ignore column for final save
    df.drop(columns=["ignore"], inplace=True)
    df.to_csv(f"{symbol}_1h_2.csv", index=False)
    print(f"Data collection completed for {symbol}")

print("Data collection completed!")
