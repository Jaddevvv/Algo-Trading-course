import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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

def backtest_ma_crossover_strategy(df, short_sma_col, long_sma_col, initial_capital, position_size_fraction, sl_amount, tp_amount):
    """
    Backtests the MA Crossover strategy with Stop Loss and Take Profit.
    - Buy when short SMA crosses above long SMA.
    - Sell when short SMA crosses below long SMA or when SL/TP is hit.
    - SL and TP are in dollar amounts from entry price.
    Only one long position at a time.
    """
    capital = initial_capital
    in_position = False
    entry_price = 0
    shares_held = 0
    trades_log = []
    profit_points = []
    current_trade_cost = 0
    sl_price = 0
    tp_price = 0

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

        # Check for SL/TP if in position
        if in_position:
            # Check if SL is hit
            if sl_amount > 0 and current_price <= sl_price:
                proceeds = sl_price * shares_held  # Use SL price for calculation
                profit = proceeds - (entry_price * shares_held) - current_trade_cost
                capital += proceeds

                trades_log.append({
                    'type': 'SELL',
                    'entry_price': round(entry_price, 2),
                    'exit_price': round(sl_price, 2),
                    'shares': round(shares_held, 5),
                    'profit': round(profit, 2),
                    'capital_after_exit': round(capital, 2),
                    'timestamp': current_timestamp,
                    'exit_reason': 'Stop Loss'
                })
                in_position = False
                entry_price = 0
                shares_held = 0
                profit_points.append(profit)
                current_trade_cost = 0
                continue

            # Check if TP is hit
            if tp_amount > 0 and current_price >= tp_price:
                proceeds = tp_price * shares_held  # Use TP price for calculation
                profit = proceeds - (entry_price * shares_held) - current_trade_cost
                capital += proceeds

                trades_log.append({
                    'type': 'SELL',
                    'entry_price': round(entry_price, 2),
                    'exit_price': round(tp_price, 2),
                    'shares': round(shares_held, 5),
                    'profit': round(profit, 2),
                    'capital_after_exit': round(capital, 2),
                    'timestamp': current_timestamp,
                    'exit_reason': 'Take Profit'
                })
                in_position = False
                entry_price = 0
                shares_held = 0
                profit_points.append(profit)
                current_trade_cost = 0
                continue

        # Buy Signal: Short SMA crosses above Long SMA
        if prev_short_sma <= prev_long_sma and current_short_sma > current_long_sma:
            if not in_position:
                if capital <= 0: continue
                investment_amount = capital * position_size_fraction
                leverage = investment_amount / capital
                shares_to_buy = investment_amount / current_price
                
                entry_price = current_price
                shares_held = shares_to_buy
                current_trade_cost = 0.00325 * investment_amount
                capital -= (investment_amount + current_trade_cost)
                in_position = True

                # Set SL and TP prices
                sl_price = entry_price - sl_amount if sl_amount > 0 else 0
                tp_price = entry_price + tp_amount if tp_amount > 0 else float('inf')

                trades_log.append({
                    'type': 'BUY',
                    'price': round(current_price, 2),
                    'shares': round(shares_held, 5),
                    'capital_after_entry': round(capital, 2),
                    'timestamp': current_timestamp,
                    'leverage': leverage,
                    'sl_price': round(sl_price, 2) if sl_amount > 0 else None,
                    'tp_price': round(tp_price, 2) if tp_amount > 0 else None
                })

        # Sell Signal: Short SMA crosses below Long SMA
        elif prev_short_sma >= prev_long_sma and current_short_sma < current_long_sma:
            if in_position:
                proceeds = current_price * shares_held
                profit = proceeds - (entry_price * shares_held) - current_trade_cost
                capital += proceeds
                
                trades_log.append({
                    'type': 'SELL',
                    'entry_price': round(entry_price, 2),
                    'exit_price': round(current_price, 2),
                    'shares': round(shares_held, 5),
                    'profit': round(profit, 2),
                    'capital_after_exit': round(capital, 2),
                    'timestamp': current_timestamp,
                    'exit_reason': 'MA Crossover'
                })
                in_position = False
                entry_price = 0
                shares_held = 0
                profit_points.append(profit)
                current_trade_cost = 0

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
            
            trade_cost = 0.00325 * investment_amount # Calculate trade cost
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
            
            trade_cost = 0.00325 * value_to_short # Calculate trade cost
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

def calculate_performance_metrics(profit_points, trades, final_capital, initial_capital):
    """Calculate various performance metrics for the strategy"""
    if not profit_points or not trades:
        return None
        
    # Convert to numpy array for calculations
    returns = np.array(profit_points)
    
    # Basic metrics
    total_return = ((final_capital - initial_capital) / initial_capital) * 100
    num_trades = len([t for t in trades if "profit" in t])
    winning_trades = [t['profit'] for t in trades if "profit" in t and t['profit'] > 0]
    losing_trades = [t['profit'] for t in trades if "profit" in t and t['profit'] < 0]
    
    # Win rate and profit metrics
    win_rate = (len(winning_trades) / num_trades * 100) if num_trades > 0 else 0
    avg_win = np.mean(winning_trades) if winning_trades else 0
    avg_loss = np.mean(losing_trades) if losing_trades else 0
    largest_win = max(winning_trades) if winning_trades else 0
    largest_loss = min(losing_trades) if losing_trades else 0
    
    # Risk metrics
    returns_std = np.std(returns) * np.sqrt(252)  # Annualized volatility
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0.0001
    
    # Sharpe & Sortino Ratios (assuming 0% risk-free rate for simplicity)
    mean_return = np.mean(returns)
    annualized_return = mean_return * 252
    sharpe_ratio = annualized_return / returns_std if returns_std != 0 else 0
    sortino_ratio = annualized_return / downside_std if downside_std != 0 else 0
    
    # Maximum Drawdown
    cumulative_returns = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = cumulative_returns - running_max
    max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0
    
    # Profit Factor
    gross_profits = sum(winning_trades) if winning_trades else 0
    gross_losses = abs(sum(losing_trades)) if losing_trades else 0.0001
    profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
    
    # Average trade metrics
    avg_trade = np.mean(returns) if len(returns) > 0 else 0
    
    # Risk-Reward Ratio
    risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    
    return {
        'total_return': total_return,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'risk_reward_ratio': risk_reward_ratio,
        'avg_trade': avg_trade,
        'annualized_return': annualized_return,
        'annualized_volatility': returns_std
    }

def print_ma_results(trades, final_capital, initial_capital=10000):
    """Print detailed backtest results including various performance metrics"""
    print("\n=== Strategy Backtest Results ===")
    if not trades:
        print("No trades were executed.")
        print(f"Final Capital: ${final_capital:.2f}")
        return

    # Calculate profit points from trades
    profit_points = [trade['profit'] for trade in trades if "profit" in trade]
    
    # Calculate all performance metrics
    metrics = calculate_performance_metrics(profit_points, trades, final_capital, initial_capital)
    
    if not metrics:
        print("Could not calculate performance metrics.")
        return
    
    # Calculate Buy and Hold returns
    # Get first and last price from the trades
    first_trade = next((t for t in trades if t['type'] == 'BUY'), None)
    last_trade = next((t for t in reversed(trades) if 'exit_price' in t), None)
    
    if first_trade and last_trade:
        entry_price = first_trade['price']
        exit_price = last_trade['exit_price']
        buy_hold_shares = initial_capital / entry_price
        buy_hold_final = buy_hold_shares * exit_price
        buy_hold_return = ((buy_hold_final - initial_capital) / initial_capital) * 100
        buy_hold_profit = buy_hold_final - initial_capital
        
        print("\n=== Strategy vs Buy & Hold Comparison ===")
        print(f"Buy & Hold Final Capital: ${buy_hold_final:,.2f}")
        print(f"Buy & Hold Total Return: {buy_hold_return:,.2f}%")
        print(f"Buy & Hold Total Profit: ${buy_hold_profit:,.2f}")
        print(f"Strategy Final Capital: ${final_capital:,.2f}")
        print(f"Strategy Total Return: {metrics['total_return']:,.2f}%")
        print(f"Strategy Total Profit: ${final_capital - initial_capital:,.2f}")
        print(f"Strategy Outperformance: {metrics['total_return'] - buy_hold_return:,.2f}%")
        
    print("\n--- Capital and Returns ---")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Capital: ${final_capital:,.2f}")
    print(f"Total Return: {metrics['total_return']:,.2f}%")
    print(f"Annualized Return: {metrics['annualized_return']*100:,.2f}%")
    print(f"Annualized Volatility: {metrics['annualized_volatility']*100:,.2f}%")
    
    print("\n--- Trade Statistics ---")
    print(f"Total Number of Trades: {metrics['num_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print(f"Average Trade: ${metrics['avg_trade']:,.2f}")
    print(f"Average Winning Trade: ${metrics['avg_win']:,.2f}")
    print(f"Average Losing Trade: ${metrics['avg_loss']:,.2f}")
    print(f"Largest Winning Trade: ${metrics['largest_win']:,.2f}")
    print(f"Largest Losing Trade: ${metrics['largest_loss']:,.2f}")
    
    print("\n--- Risk Metrics ---")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Risk-Reward Ratio: {metrics['risk_reward_ratio']:.2f}")
    
    # Print trade distribution
    print("\n--- Trade Distribution ---")
    profit_trades = len([t for t in trades if "profit" in t and t['profit'] > 0])
    loss_trades = len([t for t in trades if "profit" in t and t['profit'] < 0])
    breakeven_trades = len([t for t in trades if "profit" in t and t['profit'] == 0])
    
    print(f"Profitable Trades: {profit_trades}")
    print(f"Losing Trades: {loss_trades}")
    print(f"Breakeven Trades: {breakeven_trades}")


    # print("\n--- Trades Log ---")
    # trade_pair_count = 0
    # for i, trade in enumerate(trades_log):
    #     if trade['type'] == 'BUY':
    #         print(f"Entry {trade_pair_count + 1}: BUY {trade['shares']:.5f} shares 
    # @ ${trade['price']:.2f} on {trade['timestamp'].strftime('%Y-%m-%d %H:%M')}. 
    # Capital after entry: ${trade['capital_after_entry']:.2f}")
    #     elif trade['type'] == 'SELL':
    #         trade_pair_count +=1 
    #         print(f"Exit {trade_pair_count}: SELL {trade['shares']:.5f} shares. 
    # Entry: ${trade['entry_price']:.2f}, Exit: ${trade['exit_price']:.2f} on {trade
    # ['timestamp'].strftime('%Y-%m-%d %H:%M')}. Profit: ${trade['profit']:.2f}. 
    # Capital after exit: ${trade['capital_after_exit']:.2f}")
    

def evaluate_parameters(params, data_df, initial_capital, position_size):
    """Helper function to evaluate a single parameter combination"""
    short_period, long_period, sl, tp = params
    
    # Calculate MAs for this combination
    df_ma = calculate_moving_averages(data_df.copy(), short_window=short_period, long_window=long_period)
    df_ma.dropna(inplace=True)
    
    if df_ma.empty:
        return None
        
    short_col = f'SMA{short_period}'
    long_col = f'SMA{long_period}'
    
    trades, final_cap, profit_points = backtest_ma_crossover_strategy(
        df_ma,
        short_sma_col=short_col,
        long_sma_col=long_col,
        initial_capital=initial_capital,
        position_size_fraction=position_size,
        sl_amount=sl,
        tp_amount=tp
    )
    
    if not trades:
        return None
        
    # Calculate metrics
    positions_count = sum(1 for trade in trades if "profit" in trade)
    winning_trades = [trade['profit'] for trade in trades if "profit" in trade and trade['profit'] > 0]
    winrate = len(winning_trades) / positions_count * 100 if positions_count > 0 else 0
    
    # Calculate Sortino ratio
    sortino_ratio = 0
    if profit_points and len(profit_points) > 0:
        sortino_ratio = np.sqrt(252) * np.mean(profit_points) / np.std([p for p in profit_points if p < 0]) if any(p < 0 for p in profit_points) else np.sqrt(252) * np.mean(profit_points)
    
    return {
        'short_sma': short_period,
        'long_sma': long_period,
        'sl': sl,
        'tp': tp,
        'final_capital': final_cap,
        'sortino': sortino_ratio,
        'winrate': winrate
    }

def optimize_strategy_parameters(data_df, initial_capital=10000, position_size_frac=0.5):
    """
    Optimizes the strategy parameters by testing different combinations of:
    - Short SMA period
    - Long SMA period
    - Stop Loss amount
    - Take Profit amount
    
    Uses parallel processing to speed up the optimization.
    Returns the best performing parameters.
    """
    try:
        from pathos.multiprocessing import ProcessingPool as Pool
    except ImportError:
        print("pathos not installed. Installing pathos for parallel processing...")
        import subprocess
        subprocess.check_call(["pip", "install", "pathos"])
        from pathos.multiprocessing import ProcessingPool as Pool
    
    # Define parameter ranges to test
    short_sma_periods = range(10, 51, 5)  # 10, 15, 20, 25, 30, 35, 40, 45, 50
    long_sma_periods = range(50, 500, 25)  # 50, 75, 100, ..., 475, 500
    sl_amounts = [0, 100, 200, 300]  # Including no stop loss
    tp_amounts = [0, 200, 400, 600]  # Including no take profit
    position_size = position_size_frac
    
    # Generate all valid parameter combinations
    parameter_combinations = [
        (short, long, sl, tp)
        for short in short_sma_periods
        for long in long_sma_periods
        for sl in sl_amounts
        for tp in tp_amounts
        if long > short * 1.5  # Skip invalid combinations
    ]
    
    total_combinations = len(parameter_combinations)
    print(f"\nTesting {total_combinations} parameter combinations using parallel processing...")
    
    # Create a pool of workers
    num_cpus = os.cpu_count()
    pool = Pool(nodes=num_cpus)
    print(f"Using {num_cpus} CPU cores for parallel processing")
    
    # Evaluate all combinations in parallel
    results = []
    try:
        # Create a partial function with fixed arguments
        from functools import partial
        evaluate_partial = partial(evaluate_parameters, 
                                 data_df=data_df, 
                                 initial_capital=initial_capital,
                                 position_size=position_size)
        
        # Map the evaluation function across all parameter combinations
        results = list(filter(None, pool.map(evaluate_partial, parameter_combinations)))
    finally:
        pool.close()
        pool.join()
    
    if not results:
        print("No valid results found from parameter optimization")
        return None
    
    # Find best parameters based on Sortino ratio
    best_params = max(results, key=lambda x: x['sortino'])
    
    print("\n=== Optimization Results ===")
    print(f"Best parameters found:")
    print(f"Short SMA period: {best_params['short_sma']}")
    print(f"Long SMA period: {best_params['long_sma']}")
    print(f"Stop Loss: ${best_params['sl']}")
    print(f"Take Profit: ${best_params['tp']}")
    print(f"Final Capital: ${best_params['final_capital']:.2f}")
    print(f"Sortino Ratio: {best_params['sortino']:.2f}")
    print(f"Win Rate: {best_params['winrate']:.2f}%")
    
    # Sort results by Sortino ratio and display top 5
    print("\nTop 5 Parameter Combinations:")
    sorted_results = sorted(results, key=lambda x: x['sortino'], reverse=True)[:5]
    for i, result in enumerate(sorted_results, 1):
        print(f"\n{i}. Parameters:")
        print(f"   Short SMA: {result['short_sma']}, Long SMA: {result['long_sma']}")
        print(f"   SL: ${result['sl']}, TP: ${result['tp']}")
        print(f"   Final Capital: ${result['final_capital']:.2f}")
        print(f"   Sortino Ratio: {result['sortino']:.2f}")
        print(f"   Win Rate: {result['winrate']:.2f}%")
    
    return best_params

if __name__ == "__main__":
    file_path = "../Data/BTCUSDT_1h_2.csv"
    sma_short_period = 35
    sma_long_period = 450
    initial_capital_value = 10000
    position_size_frac = 1
    SL = 0
    TP = 0

    # Load data
    data_df = load_and_prepare_data(file_path, lookback_days=365*4)

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
            # Uncomment to optimize strategy parameters
            # optimized_params = optimize_strategy_parameters(data_df_ma, initial_capital=initial_capital_value, position_size_frac=position_size_frac)







            short_col_name = f'SMA{sma_short_period}'
            long_col_name = f'SMA{sma_long_period}'
            
            trades, final_cap, profit_points = backtest_ma_crossover_strategy(data_df_ma, 
                                                               short_sma_col=short_col_name, 
                                                               long_sma_col=long_col_name,
                                                               initial_capital=initial_capital_value,
                                                               position_size_fraction=position_size_frac,
                                                               sl_amount=SL,
                                                               tp_amount=TP)
            
            # trades, final_cap, profit_points = backtest_daily_entry_strategy(data_df_ma, 
            #                                                    initial_capital=initial_capital_value,
            #                                                    position_size_fraction=position_size_frac)
            print_ma_results(trades, final_cap, initial_capital=initial_capital_value)

            draw_equity_chart(profit_points)


