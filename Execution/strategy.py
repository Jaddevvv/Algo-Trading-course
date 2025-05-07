import pandas as pd
import numpy as np

def load_and_prepare_data(file_path, lookback_days=365):
    """
    Loads data from a CSV file, converts timestamp, sets index,
    and filters for the specified lookback period.
    """
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    if 'Close' in df.columns and 'close' not in df.columns:
        df.rename(columns={'Close': 'close'}, inplace=True)
    elif 'price' in df.columns and 'close' not in df.columns:
        df.rename(columns={'price': 'close'}, inplace=True)
    elif 'Price' in df.columns and 'close' not in df.columns:
        df.rename(columns={'Price': 'close'}, inplace=True)

    end_date = df.index.max()
    start_date = end_date - pd.Timedelta(days=lookback_days)
    df_filtered = df[df.index >= start_date].copy()
    return df_filtered

def calculate_moving_averages(df, short_window=75, long_window=200):
    """
    Calculates short and long Simple Moving Averages.
    """
    df.loc[:, f'SMA{short_window}'] = df['close'].rolling(window=short_window).mean()
    df.loc[:, f'SMA{long_window}'] = df['close'].rolling(window=long_window).mean()
    return df

def backtest_ma_crossover_strategy(df, short_sma_col, long_sma_col, initial_capital=10000, position_size_fraction=0.10):
    """
    Backtests the MA Crossover strategy.
    - Buy when short SMA crosses above long SMA.
    - Sell when short SMA crosses below long SMA.
    Only one long position at a time.
    """
    capital = initial_capital
    in_position = False
    entry_price = 0
    shares_held = 0
    trades_log = []

    # Ensure columns exist
    if short_sma_col not in df.columns or long_sma_col not in df.columns:
        print(f"Error: SMA columns ('{short_sma_col}' or '{long_sma_col}') not found in DataFrame.")
        return [], capital

    # We need a previous state to detect a crossover, so we start from the second available row
    for i in range(1, len(df)):
        current_short_sma = df[short_sma_col].iloc[i]
        current_long_sma = df[long_sma_col].iloc[i]
        prev_short_sma = df[short_sma_col].iloc[i-1]
        prev_long_sma = df[long_sma_col].iloc[i-1]
        current_price = df['close'].iloc[i]
        current_timestamp = df.index[i]

        # Buy Signal: Short SMA crosses above Long SMA
        if prev_short_sma <= prev_long_sma and current_short_sma > current_long_sma:
            if not in_position:
                if capital <= 0: continue
                investment_amount = capital * position_size_fraction
                shares_to_buy = investment_amount / current_price
                
                entry_price = current_price
                shares_held = shares_to_buy
                capital -= investment_amount # Cost of shares deducted
                in_position = True
                trades_log.append({
                    'type': 'BUY',
                    'price': round(current_price, 2),
                    'shares': round(shares_held, 5),
                    'capital_after_entry': round(capital, 2),
                    'timestamp': current_timestamp
                })

        # Sell Signal: Short SMA crosses below Long SMA
        elif prev_short_sma >= prev_long_sma and current_short_sma < current_long_sma:
            if in_position:
                proceeds = current_price * shares_held
                profit = proceeds - (entry_price * shares_held) # Calculate profit from this specific trade
                capital += proceeds # Add proceeds back to capital (which includes the original cost + profit/loss)
                
                trades_log.append({
                    'type': 'SELL',
                    'entry_price': round(entry_price, 2),
                    'exit_price': round(current_price, 2),
                    'shares': round(shares_held, 5),
                    'profit': round(profit, 2),
                    'capital_after_exit': round(capital, 2),
                    'timestamp': current_timestamp
                })
                in_position = False
                entry_price = 0
                shares_held = 0
                
    # If still in position at the end, you might want to close it or log it
    # For this version, we'll just note the final capital based on trades executed.

    return trades_log, capital

def print_ma_results(trades_log, final_capital, initial_capital=10000):
    print("\n--- MA Crossover Strategy Backtest Results ---")
    if not trades_log:
        print("No trades were executed.")
        print(f"Final Capital: ${final_capital:.2f}")
        return

    total_profit = final_capital - initial_capital
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Capital: ${final_capital:.2f}")
    print(f"Total Profit/Loss: ${total_profit:.2f}")

    buys = sum(1 for trade in trades_log if trade['type'] == 'BUY')
    sells = sum(1 for trade in trades_log if trade['type'] == 'SELL')
    print(f"Total Buy Signals Acted On: {buys}")
    print(f"Total Sell Signals (Positions Closed): {sells}")

    print("\n--- Trades Log ---")
    trade_pair_count = 0
    for i, trade in enumerate(trades_log):
        if trade['type'] == 'BUY':
            print(f"Entry {trade_pair_count + 1}: BUY {trade['shares']:.5f} shares @ ${trade['price']:.2f} on {trade['timestamp'].strftime('%Y-%m-%d %H:%M')}. Capital after entry: ${trade['capital_after_entry']:.2f}")
        elif trade['type'] == 'SELL':
            trade_pair_count +=1 
            print(f"Exit {trade_pair_count}: SELL {trade['shares']:.5f} shares. Entry: ${trade['entry_price']:.2f}, Exit: ${trade['exit_price']:.2f} on {trade['timestamp'].strftime('%Y-%m-%d %H:%M')}. Profit: ${trade['profit']:.2f}. Capital after exit: ${trade['capital_after_exit']:.2f}")

if __name__ == "__main__":
    file_path = "../Data/BTCUSDT_1h_2.csv"
    sma_short_period = 75
    sma_long_period = 200
    initial_capital_value = 10000
    position_size_frac = 0.10 # Use 10% of current capital per trade

    # Load data
    data_df = load_and_prepare_data(file_path, lookback_days=365*2) # Using 2 years for more MA data

    if 'close' not in data_df.columns:
        print("Error: 'close' column not found.")
    else:
        # Calculate Moving Averages
        data_df_ma = calculate_moving_averages(data_df, short_window=sma_short_period, long_window=sma_long_period)
        data_df_ma.dropna(inplace=True) # Remove rows with NaN due to MA calculation

        print(f"Data loaded for MA Crossover. Shape: {data_df_ma.shape}")
        # print(data_df_ma.head())

        if data_df_ma.empty:
            print("DataFrame is empty after MA calculation, cannot backtest.")
        else:
            short_col_name = f'SMA{sma_short_period}'
            long_col_name = f'SMA{sma_long_period}'
            
            trades, final_cap = backtest_ma_crossover_strategy(data_df_ma, 
                                                               short_sma_col=short_col_name, 
                                                               long_sma_col=long_col_name,
                                                               initial_capital=initial_capital_value,
                                                               position_size_fraction=position_size_frac)
            print_ma_results(trades, final_cap, initial_capital=initial_capital_value)
