import pandas as pd
import mplfinance as mpf
import os
import numpy as np
from scipy.signal import argrelextrema


# Get user input for number of levels
nb_levels = 1
nb_touches = 3

# Read the CSV file
df = pd.read_csv("../Data/BTCUSDT_1h_2.csv")

# Drop last 500 rows 
df = df.tail(5000)

# Convert Unix timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Set timestamp as index
df.set_index('timestamp', inplace=True)

def find_support_resistance(df, window=100, num_touches=3, price_threshold=20):
    # Convert DataFrame to numpy array for faster computation
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values  # Get volume data
    
    # Calculate rolling mean of volume
    volume_ma = pd.Series(volumes).rolling(window=window, min_periods=1).mean().values
    
    # Find local maxima and minima
    high_idx = argrelextrema(highs, np.greater, order=window)[0]
    low_idx = argrelextrema(lows, np.less, order=window)[0]
    
    # Reverse indices to start from the end
    high_idx = high_idx[::-1]
    low_idx = low_idx[::-1]
    
    # Initialize lists for support and resistance levels with their touch counts and recency
    support_levels = []
    resistance_levels = []
    
    # Find resistance levels (local highs)
    for idx in high_idx:
        level = highs[idx]
        # Count touches for resistance
        touches = np.sum(
            (highs >= level * (1-price_threshold/100)) & 
            (highs <= level * (1+price_threshold/100)) &
            (volumes >= volume_ma)  # Add volume condition
        )
        if touches >= num_touches:
            # Store level with its index (for recency) and touch count
            resistance_levels.append((level, touches, idx))
    
    # Find support levels (local lows)
    for idx in low_idx:
        level = lows[idx]
        # Count touches for support
        touches = np.sum(
            (lows >= level * (1-price_threshold/100)) & 
            (lows <= level * (1+price_threshold/100)) &
            (volumes >= volume_ma)  # Add volume condition
        )
        if touches >= num_touches:
            # Store level with its index (for recency) and touch count
            support_levels.append((level, touches, idx))
    
    # Sort by recency (index) first, then by number of touches
    support_levels.sort(key=lambda x: (-x[1], x[2]))  # Sort by touches (desc) and index (asc)
    resistance_levels.sort(key=lambda x: (-x[1], x[2]))
    
    # Take top n levels and return only the price levels
    return ([x[0] for x in support_levels[:nb_levels]], 
            [x[0] for x in resistance_levels[:nb_levels]])

def find_multiple_trendlines(df, window=10, min_touches=nb_touches, min_distance=10, max_lines=nb_levels, is_support=True):
    # Get price data based on whether we're looking for support or resistance lines
    prices = df['low'].values if is_support else df['high'].values
    
    # Find local extrema
    if is_support:
        extrema_idx = argrelextrema(prices, np.less, order=window)[0]
    else:
        extrema_idx = argrelextrema(prices, np.greater, order=window)[0]
    
    # Reverse the indices to start from the end
    extrema_idx = extrema_idx[::-1]
    
    trendlines = []
    used_points = set()  # Keep track of points we've used
    valid_lines_count = 0  # Track number of valid lines separately
    
    # Start from the last point and try to find valid trendlines
    for i in range(len(extrema_idx)-1):
        x1 = extrema_idx[i]  # This is the more recent point
        
        if x1 in used_points:
            continue
            
        # Try connecting with previous points
        for j in range(i+1, len(extrema_idx)):
            x2 = extrema_idx[j]  # This is the older point
            
            if x2 in used_points:
                continue
            
            # Skip if points are too close
            if abs(x1 - x2) < min_distance:
                continue
                
            y1, y2 = prices[x1], prices[x2]
            
            # Calculate line parameters (y = mx + b)
            m = (y1 - y2) / (x1 - x2)  # Reversed because x1 is more recent
            b = y1 - m * x1
            
            # Count touches
            touches = 0
            tolerance = 0.01  # 1% tolerance
            
            # Check for touches between the two points
            for k in range(min(x1, x2), max(x1, x2)):
                expected_y = m * k + b
                if abs(prices[k] - expected_y) / expected_y < tolerance:
                    touches += 1
            
            # If we found a valid trendline, check for price crossover
            if touches >= min_touches:
                # Check if any price crosses above/below the line after x1
                valid_line = True
                for k in range(x1, len(prices)):
                    expected_y = m * k + b
                    if is_support and prices[k] < expected_y - expected_y * tolerance:
                        valid_line = False
                        break
                    elif not is_support and prices[k] > expected_y + expected_y * tolerance:
                        valid_line = False
                        break
                
                if valid_line:
                    trendlines.append(((m, b), (x2, x1)))  # Store points in chronological order
                    used_points.add(x1)
                    used_points.add(x2)
                    valid_lines_count += 1
                    break  # Move to next recent point
                
        # If we have enough valid trendlines, stop searching
        if valid_lines_count >= max_lines:
            break
    
    return trendlines

# Find support and resistance levels
support_levels, resistance_levels = find_support_resistance(df)
# support_levels = []
# resistance_levels = []

# Find both bottom and top trendlines
bottom_trendlines = find_multiple_trendlines(df, is_support=True)
top_trendlines = find_multiple_trendlines(df, is_support=False)

# Prepare plot
kwargs = dict(type='candle', volume=True, title='BTC/USDT Candlestick Chart', style='yahoo')

# Add support and resistance levels with different colors
if support_levels or resistance_levels:
    hlines = []
    colors = []
    
    # Add support levels (green)
    hlines.extend(support_levels)
    colors.extend(['g' for _ in support_levels])
    
    # Add resistance levels (red)
    hlines.extend(resistance_levels)
    colors.extend(['r' for _ in resistance_levels])
    
    if hlines:
        kwargs['hlines'] = dict(hlines=hlines, colors=colors, linestyle='--', linewidths=1)

# Add trendlines if found
alines = []
colors = []

# Get the last index for extending lines
last_idx = len(df) - 1

# Add bottom trendlines (green)
for trendline, points in bottom_trendlines:
    m, b = trendline
    x1, x2 = points
    y1 = m * x1 + b
    # Extend to the right edge of the chart
    y_end = m * last_idx + b
    alines.append([(df.index[x1], y1), (df.index[last_idx], y_end)])
    colors.append('g')

# Add top trendlines (blue)
for trendline, points in top_trendlines:
    m, b = trendline
    x1, x2 = points
    y1 = m * x1 + b
    # Extend to the right edge of the chart
    y_end = m * last_idx + b
    alines.append([(df.index[x1], y1), (df.index[last_idx], y_end)])
    colors.append('b')

if alines:
    kwargs['alines'] = dict(alines=alines, colors=colors)

# Plot the chart with all elements
mpf.plot(df, **kwargs)

