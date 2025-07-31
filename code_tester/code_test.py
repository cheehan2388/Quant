import numpy as np
import pandas as pd
import timeit

# First function: positions_trend (NumPy/pandas-based)
def positions_trend(signal, thr):
    raw = np.where(signal > thr, 1,
                   np.where(signal < -thr, -1, 0))
    pos = pd.Series(raw, index=signal.index)
    return pos.replace(0, np.nan).ffill().fillna(0)

# Second function: trend (Python loop-based)
def trend(model_data, upper_threshold, lower_threshold):
    pos = [0]  # Initialize with 0 to avoid index errors
    long_trades = 0
    short_trades = 0
    for i in range(1, len(model_data)):
        if model_data[i] >= upper_threshold:
            pos.append(1)
            if pos[-2] != pos[-1]:
                long_trades += 1
        elif model_data[i] <= lower_threshold:
            pos.append(-1)
            if pos[-2] != pos[-1]:
                short_trades += 1
        else:
            pos.append(pos[-1])
    return pos, long_trades, short_trades

# Create a small dataset (e.g., 100 data points)
np.random.seed(42)  # For reproducibility
small_data = np.random.uniform(-1, 1, 1000)  # Random values between -1 and 1
small_series = pd.Series(small_data, index=range(1000))  # pandas Series for positions_trend

# Parameters
threshold = 0.5  # Symmetric threshold for positions_trend
upper_threshold = 0.5  # For trend
lower_threshold = -0.5  # For trend

# Test positions_trend
def test_positions_trend():
    return positions_trend(small_series, threshold)

# Test trend
def test_trend():
    return trend(small_data, upper_threshold, lower_threshold)

# Measure execution time
n_runs = 1000  # Number of runs for timing
positions_trend_time = timeit.timeit(test_positions_trend, number=n_runs)
trend_time = timeit.timeit(test_trend, number=n_runs)

# Print results
print(f"Dataset size: {len(small_data)}")
print(f"positions_trend time ({n_runs} runs): {positions_trend_time:.6f} seconds")
print(f"trend time ({n_runs} runs): {trend_time:.6f} seconds")
print(f"Speedup (trend vs positions_trend): {positions_trend_time / trend_time:.2f}x")

# Optional: Verify outputs (for correctness)
pos_trend = positions_trend(small_series, threshold)
pos_loop, long_trades, short_trades = trend(small_data, upper_threshold, lower_threshold)
print(f"\npositions_trend output (first 10): {pos_trend[:10].values}")
print(f"trend output (first 10): {pos_loop[:10]}")
print(f"Long trades (trend): {long_trades}, Short trades (trend): {short_trades}")