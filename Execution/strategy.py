
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    profit_points = []
    current_trade_cost = 0 # Variable to store cost of current trade

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
                leverage = investment_amount / capital
                shares_to_buy = investment_amount / current_price
                
                entry_price = current_price
                shares_held = shares_to_buy
                current_trade_cost = 0.002 * investment_amount # Calculate and store trade cost
                capital -= (investment_amount + current_trade_cost) # Deduct investment and cost
                in_position = True
                trades_log.append({
                    'type': 'BUY',
                    'price': round(current_price, 2),
                    'shares': round(shares_held, 5),
                    'capital_after_entry': round(capital, 2),
                    'timestamp': current_timestamp,
                    'leverage': leverage
                })

        # Sell Signal: Short SMA crosses below Long SMA
        elif prev_short_sma >= prev_long_sma and current_short_sma < current_long_sma:
            if in_position:
                proceeds = current_price * shares_held
                # Profit is proceeds - original investment. Cost was already deducted from capital.
                # To show profit correctly per trade, we consider (proceeds - original investment) 
                # effectively the cost reduces the net profit from (proceeds - entry_price*shares_held)
                profit_before_cost = proceeds - (entry_price * shares_held)
                net_profit = profit_before_cost # Cost is already factored into capital change
                                                # but for individual trade P&L, it was part of initial outlay implicitly.
                                                # Let's make the trade profit reflect the cost explicitly for logging.
                                                # The profit here is how much the asset value changed MINUS the cost.
                
                # The capital change due to this trade is (proceeds - (entry_price*shares_held) - current_trade_cost)
                # So profit for the log should be: (proceeds - (entry_price * shares_held)) - current_trade_cost
                # but current_trade_cost was already subtracted from capital at entry.
                # So the 'profit' for the trade log should reflect the gain/loss on the asset itself.
                # Capital already reflects the cost.

                # Let's adjust profit calculation to be more explicit about cost for the log entry.
                # Initial investment was entry_price * shares_held.
                # Total outlay for the trade was (entry_price * shares_held) + current_trade_cost.
                # Profit = proceeds - total_outlay
                profit = proceeds - (entry_price * shares_held) - current_trade_cost

                capital += proceeds # Add proceeds back to capital
                
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
                profit_points.append(profit)
                current_trade_cost = 0 # Reset cost for next trade

    return trades_log, capital, profit_points

# New strategy take a position at 00:00 candle close and  sell at the next day. Buy only if the previous day close was a green candle and sell only if the previous day close was a red candle. Close the position at the next day 00:00 candle close.

def backtest_daily_entry_strategy(df_hourly, initial_capital=10000, position_size_fraction=0.10):
    """
    Backtests a daily entry strategy:
    - Decide based on previous day's candle (green/red).
    - Enter at current day's 00:00 candle close.
    - Exit at next day's 00:00 candle close.
    - Long if previous day green, Short if previous day red.
    """
    capital = float(initial_capital)
    trades_log = []
    profit_points = []


    # 1. Create daily signals based on previous day's candle
    daily_ohlc = df_hourly.resample('D').agg(
        open_val=('open', 'first'),
        close_val=('close', 'last')
    ).dropna()

    daily_signals = daily_ohlc.copy()
    daily_signals['prev_day_open'] = daily_ohlc['open_val'].shift(1)
    daily_signals['prev_day_close'] = daily_ohlc['close_val'].shift(1)
    daily_signals.dropna(inplace=True) 

    # 2. Get trade execution prices (close of 00:00 candle for each day)
    # Ensure unique timestamps for trade_execution_prices to prevent .get() returning a Series
    trade_execution_prices_raw = df_hourly[df_hourly.index.hour == 0]['close']
    trade_execution_prices = trade_execution_prices_raw[~trade_execution_prices_raw.index.duplicated(keep='first')]

    if trade_execution_prices.empty:
        print("Warning: No 00:00 candle close prices found for daily_entry_strategy. Cannot execute trades.")
        return [], capital, []

    # 3. Iterate through days where a signal can be generated
    for decision_day_timestamp, row_data in daily_signals.iterrows():
        prev_day_open = row_data['prev_day_open']
        prev_day_close = row_data['prev_day_close']

        entry_trigger_timestamp = decision_day_timestamp 
        exit_trigger_timestamp = decision_day_timestamp + pd.Timedelta(days=1)

        entry_price = trade_execution_prices.get(entry_trigger_timestamp)
        exit_price = trade_execution_prices.get(exit_trigger_timestamp)

        # Skip if entry or exit price is not available, or if no trading is possible
        if entry_price is None or exit_price is None:
            continue
            
        if capital <= 0 or position_size_fraction == 0:
            continue

        current_capital_for_trade = capital 
        trade_type = None
        if prev_day_close > prev_day_open:
            trade_type = 'BUY'
        elif prev_day_close < prev_day_open:
            trade_type = 'SHORTSELL'
        else:
            continue 

        if trade_type == 'BUY':
            investment_amount = current_capital_for_trade * position_size_fraction
            shares = investment_amount / entry_price
            leverage = investment_amount / current_capital_for_trade
            
            trade_cost = 0.002 * investment_amount # Calculate trade cost
            cost_of_shares = shares * entry_price
            capital_after_entry = current_capital_for_trade - cost_of_shares - trade_cost # Deduct cost
            
            trades_log.append({
                'type': 'BUY',
                'price': round(entry_price, 5),
                'shares': round(shares, 5),
                'capital_after_entry': round(capital_after_entry, 2),
                'timestamp': entry_trigger_timestamp,
                'leverage': leverage
            })
            
            proceeds_from_sale = shares * exit_price
            profit = proceeds_from_sale - cost_of_shares - trade_cost # Factor cost into profit
            capital_after_exit = capital_after_entry + proceeds_from_sale
            
            trades_log.append({
                'type': 'SELL',
                'entry_price': round(entry_price, 5),
                'exit_price': round(exit_price, 5),
                'shares': round(shares, 5),
                'profit': round(profit, 2),
                'capital_after_exit': round(capital_after_exit, 2),
                'timestamp': exit_trigger_timestamp
            })
            profit_points.append(profit)
            capital = capital_after_exit

        elif trade_type == 'SHORTSELL':
            value_to_short = current_capital_for_trade * position_size_fraction
            shares = value_to_short / entry_price
            leverage = value_to_short / current_capital_for_trade
            
            trade_cost = 0.002 * value_to_short # Calculate trade cost
            proceeds_from_short_value = shares * entry_price 
            # For short, capital effectively increases by proceeds, then cost is paid from that or existing capital
            capital_after_short_opened = current_capital_for_trade + proceeds_from_short_value - trade_cost # Deduct cost

            trades_log.append({
                'type': 'SHORTSELL',
                'price': round(entry_price, 5),
                'shares': round(shares, 5),
                'capital_after_entry': round(capital_after_short_opened, 2),
                'timestamp': entry_trigger_timestamp,
                'leverage': leverage
            })

            cost_to_cover = shares * exit_price
            # Profit = (proceeds from short) - (cost to cover) - (trade cost)
            profit = proceeds_from_short_value - cost_to_cover - trade_cost # Factor cost into profit
            capital_after_exit_logged = capital_after_short_opened - cost_to_cover

            trades_log.append({
                'type': 'COVER',
                'entry_price': round(entry_price, 5),
                'exit_price': round(exit_price, 5) if exit_price is not None else None,
                'shares': round(shares, 5),
                'profit': round(profit, 2),
                'capital_after_exit': round(capital_after_exit_logged, 2),
                'timestamp': exit_trigger_timestamp
            })
            profit_points.append(profit)
            capital = capital_after_exit_logged

    return trades_log, capital, profit_points

def draw_equity_chart(profit_points):
    # Calculate cumulative profit
    cumulative_profit = np.cumsum(profit_points)
    
    # Plot equity chart
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_profit, label='Cumulative Profit')
    plt.title('Equity Chart')
    plt.xlabel('Time')
    plt.ylabel('Profit')
    plt.legend()
    plt.show()


def print_ma_results(trades_log, final_capital, initial_capital=10000):
    print("\n--- MA Crossover Strategy Backtest Results ---")
    if not trades_log:
        print("No trades were executed.")
        print(f"Final Capital: ${final_capital:.2f}")
        return

    total_profit = final_capital - initial_capital
    max_leverage = max(trade['leverage'] for trade in trades_log if 'leverage' in trade)
    positions_count = sum(1 for trade in trades_log if "profit" in trade)
    
    winning_trades = [trade['profit'] for trade in trades_log if "profit" in trade and trade['profit'] > 0]
    losing_trades = [trade['profit'] for trade in trades_log if "profit" in trade and trade['profit'] < 0]

    winrate = 0
    if positions_count > 0:
        winrate = len(winning_trades) / positions_count * 100

    average_rr_text = "N/A"
    if winning_trades and losing_trades:
        avg_win = np.average(winning_trades)
        avg_loss = np.average(losing_trades)
        average_rr = avg_win / abs(avg_loss)
        average_rr_text = f"{average_rr:.2f}"

    # Calculate Sortino Ratio
    sortino_ratio = 0
    if positions_count > 0:
        sortino_ratio = np.sqrt(252) * np.mean(profit_points) / np.std(profit_points)


    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Capital: ${final_capital:.2f}")
    print(f"Total Profit/Loss: ${total_profit:.2f}")
    print(f"Max Leverage Used: {max_leverage:.2f}")
    print(f"Winrate: {winrate:.2f}%")
    print(f"Average RR: {average_rr_text}")
    print(f"Sortino Ratio: {sortino_ratio:.2f}")

    buys = sum(1 for trade in trades_log if trade['type'] == 'BUY')
    sells = sum(1 for trade in trades_log if trade['type'] == 'SELL')
    print(f"Total Positions Opened: {buys}")
    print(f"Total Positions Closed: {sells}")

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
    position_size_frac = 0.5 # Use 10% of current capital per trade

    # Load data
    data_df = load_and_prepare_data(file_path, lookback_days=365*2)

    if 'close' not in data_df.columns:
        print("Error: 'close' column not found.")
    else:
        # Calculate Moving Averages
        data_df_ma = calculate_moving_averages(data_df, short_window=sma_short_period, long_window=sma_long_period)
        data_df_ma.dropna(inplace=True) # Remove rows with NaN due to MA calculation

        print(f"Data loaded for MA Crossover. Shape: {data_df_ma.shape}")

        if data_df_ma.empty:
            print("DataFrame is empty after MA calculation, cannot backtest.")
        else:
            short_col_name = f'SMA{sma_short_period}'
            long_col_name = f'SMA{sma_long_period}'
            
            trades, final_cap, profit_points = backtest_ma_crossover_strategy(data_df_ma, 
                                                               short_sma_col=short_col_name, 
                                                               long_sma_col=long_col_name,
                                                               initial_capital=initial_capital_value,
                                                               position_size_fraction=position_size_frac)
            
            # trades, final_cap, profit_points = backtest_daily_entry_strategy(data_df_ma, 
            #                                                    initial_capital=initial_capital_value,
            #                                                    position_size_fraction=position_size_frac)
            print_ma_results(trades, final_cap, initial_capital=initial_capital_value)

            draw_equity_chart(profit_points)


