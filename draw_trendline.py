import pandas as pd
import mplfinance as mpf
import os
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt


# Get user input for number of levels
nb_levels = 5
nb_touches = 2

# Read the CSV file
df = pd.read_csv("History/BTCUSDT_1h_2.csv")

# Drop last 500 rows 
df = df.tail(5000)

# Convert Unix timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Set timestamp as index
df.set_index('timestamp', inplace=True)

def find_support_resistance(df, window=100, num_touches=10, price_threshold=20):
    # Convert DataFrame to numpy array for faster computation
    highs = df['high'].values
    lows = df['low'].values
    
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
            (highs <= level * (1+price_threshold/100))
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
            (lows <= level * (1+price_threshold/100))
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

def find_multiple_trendlines(df, window=20, min_touches=nb_touches, min_distance=100, max_lines=nb_levels, is_support=True):
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
            tolerance = 0.005  # 0.5% tolerance
            
            # Check for touches between the two points
            for k in range(min(x1, x2), max(x1, x2)):
                expected_y = m * k + b
                if abs(prices[k] - expected_y) / expected_y < tolerance:
                    touches += 1
            
            # If we found a valid trendline, add it to our list
            if touches >= min_touches:
                trendlines.append(((m, b), (x2, x1)))  # Store points in chronological order
                used_points.add(x1)
                used_points.add(x2)
                break  # Move to next recent point
                
        # If we have enough trendlines, stop searching
        if len(trendlines) >= max_lines:
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

# Add bottom trendlines (green)
for trendline, points in bottom_trendlines:
    m, b = trendline
    x1, x2 = points
    y1 = m * x1 + b
    y2 = m * x2 + b
    alines.append([(df.index[x1], y1), (df.index[x2], y2)])
    colors.append('g')

# Add top trendlines (blue)
for trendline, points in top_trendlines:
    m, b = trendline
    x1, x2 = points
    y1 = m * x1 + b
    y2 = m * x2 + b
    alines.append([(df.index[x1], y1), (df.index[x2], y2)])
    colors.append('b')

if alines:
    kwargs['alines'] = dict(alines=alines, colors=colors)

# Plot the chart with all elements
mpf.plot(df, **kwargs)

