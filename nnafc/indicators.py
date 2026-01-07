from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int = 14) -> pd.Series:
	return series.rolling(window).mean()


def ema(series: pd.Series, span: int = 14) -> pd.Series:
	return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
	delta = series.diff()
	gain = delta.clip(lower=0.0)
	loss = -delta.clip(upper=0.0)
	avg_gain = gain.rolling(window).mean()
	avg_loss = loss.rolling(window).mean()
	rs = avg_gain / (avg_loss.replace(0, np.nan))
	val = 100 - (100 / (1 + rs))
	return val.fillna(50.0)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
	fast_ema = ema(series, span=fast)
	slow_ema = ema(series, span=slow)
	macd_line = fast_ema - slow_ema
	signal_line = ema(macd_line, span=signal)
	return macd_line - signal_line


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
	ma = series.rolling(window).mean()
	std = series.rolling(window).std()
	upper = ma + num_std * std
	lower = ma - num_std * std
	# Return bandwidth (upper - lower) / ma as a single-valued indicator
	bandwidth = (upper - lower) / ma.replace(0, np.nan)
	return bandwidth


def compute_pk_indicator(name: str, close: pd.Series, params: dict | None = None) -> pd.Series:
	params = params or {}
	name = name.lower()
	if name == "sma":
		return sma(close, window=int(params.get("window", 14)))
	elif name == "ema":
		return ema(close, span=int(params.get("span", 14)))
	elif name == "rsi":
		return rsi(close, window=int(params.get("window", 14)))
	elif name == "macd":
		return macd(close,
				 fast=int(params.get("fast", 12)),
				 slow=int(params.get("slow", 26)),
				 signal=int(params.get("signal", 9)))
	elif name in ("boll", "bollinger", "bbands"):
		return bollinger_bands(close, window=int(params.get("window", 20)), num_std=float(params.get("num_std", 2.0)))
	else:
		raise ValueError(f"Unknown indicator: {name}")


