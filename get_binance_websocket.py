import websocket
import json
import pandas as pd
from datetime import datetime
import time

def on_message(ws, message):
    global df
    data = json.loads(message)
    
    if 'result' in data and data['result']:
        kline_data = data['result']
        temp_df = pd.DataFrame(kline_data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume","ignore"])
        df = pd.concat([df, temp_df])
        
        # Save after each update
        symbol = data['id'].split('_')[0]  # Extract symbol from request ID
        interval = data['id'].split('_')[1]  # Extract interval from request ID
        df.to_csv(f"{symbol}_{interval}_ws.csv", index=False)

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket connection closed")

def on_open(ws):
    global current_time
    
    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        # Start time: January 1, 2017 (in milliseconds)
        start_time = 1483225200000
        
        while start_time < current_time:
            request = {
                "id": f"{symbol}_1h",
                "method": "klines",
                "params": {
                    "symbol": symbol,
                    "interval": "1h",
                    "startTime": start_time,
                    "limit": 1000
                }
            }
            ws.send(json.dumps(request))
            time.sleep(0.1)  # Add small delay to avoid rate limits
            start_time += 3600000 * 1000  # Move forward by 1000 hours

if __name__ == "__main__":
    # Initialize global DataFrame
    df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume","ignore"])
    
    # Get current time in milliseconds
    current_time = int(datetime.now().timestamp() * 1000)
    
    # Binance WebSocket URL
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(
        "wss://ws-api.binance.com:443/ws-api/v3",
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )
    
    ws.run_forever()
