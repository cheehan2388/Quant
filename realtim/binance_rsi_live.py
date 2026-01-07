import argparse
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd


def setup_logger(level: str = "INFO") -> logging.Logger:
	logging.basicConfig(
		level=getattr(logging, level.upper(), logging.INFO),
		format="%(asctime)s %(levelname)s %(message)s",
		handlers=[logging.StreamHandler(sys.stdout)],
	)
	return logging.getLogger("binance_rsi_live")


def compute_rsi_from_close(close_series: pd.Series, period: int = 14) -> pd.Series:
	"""Compute RSI using Wilder's smoothing (EMA with alpha=1/period)."""
	if len(close_series) < max(3, period + 1):
		return pd.Series(index=close_series.index, dtype=float)

	delta = close_series.diff()
	gain = delta.clip(lower=0)
	loss = -delta.clip(upper=0)

	# Wilder's smoothing
	avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
	avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

	rs = avg_gain / avg_loss.replace(0, np.nan)
	rsi = 100 - (100 / (1 + rs))
	return rsi.fillna(method="bfill")


def timeframe_to_seconds(timeframe: str) -> int:
	units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
	try:
		unit = timeframe[-1]
		value = int(timeframe[:-1])
		return value * units[unit]
	except Exception:
		raise ValueError(f"Invalid timeframe: {timeframe}")


def wait_until_next_candle(tf_seconds: int, buffer_seconds: int = 2) -> None:
	now = int(datetime.now(timezone.utc).timestamp())
	wait = tf_seconds - (now % tf_seconds) + buffer_seconds
	if wait > 0:
		time.sleep(wait)


def fetch_binance_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
	data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
	df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
	df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
	df.set_index("datetime", inplace=True)
	return df


def main():
	parser = argparse.ArgumentParser(description="Live Binance Close + RSI using ccxt")
	parser.add_argument("--symbol", default=os.getenv("SYMBOL", "BTC/USDT"), help="CCXT symbol, e.g., BTC/USDT")
	parser.add_argument("--timeframe", default=os.getenv("TIMEFRAME", "1m"), help="Candle timeframe, e.g., 1m, 5m, 1h")
	parser.add_argument("--period", type=int, default=int(os.getenv("RSI_PERIOD", 14)), help="RSI period")
	parser.add_argument("--limit", type=int, default=int(os.getenv("LIMIT", 300)), help="History limit for RSI calc")
	parser.add_argument("--log", default=os.getenv("LOG_LEVEL", "INFO"), help="Log level")
	parser.add_argument("--futures", action="store_true", help="Use Binance futures (default spot)")
	args = parser.parse_args()

	logger = setup_logger(args.log)
	logger.info(f"Starting live RSI for {args.symbol} @ {args.timeframe} (period={args.period})")

	options = {"defaultType": "future"} if args.futures else {}
	exchange = ccxt.binance({
		"enableRateLimit": True,
		"options": options,
	})

	# Load markets once
	exchange.load_markets()
	if args.symbol not in exchange.markets:
		raise ValueError(f"Symbol not found on Binance: {args.symbol}")

	tf_seconds = timeframe_to_seconds(args.timeframe)

	# Prime with initial history
	df = fetch_binance_ohlcv(exchange, args.symbol, args.timeframe, limit=args.limit)
	if df.empty:
		raise RuntimeError("Failed to fetch initial OHLCV data")

	while True:
		try:
			# Align to next candle close
			wait_until_next_candle(tf_seconds=tf_seconds, buffer_seconds=2)

			# Refresh recent window (limit keeps bandwidth reasonable)
			df = fetch_binance_ohlcv(exchange, args.symbol, args.timeframe, limit=args.limit)
			close = df["close"]
			rsi = compute_rsi_from_close(close, period=args.period)

			last_time = df.index[-1].strftime("%Y-%m-%d %H:%M:%S %Z")
			last_close = float(close.iloc[-1])
			last_rsi = float(rsi.iloc[-1]) if not math.isnan(rsi.iloc[-1]) else float("nan")

			state = "NEUTRAL"
			if last_rsi >= 70:
				state = "OVERBOUGHT"
			elif last_rsi <= 30:
				state = "OVERSOLD"

			logger.info(f"{last_time} | close={last_close:.2f} | rsi={last_rsi:.2f} | {state}")
		except KeyboardInterrupt:
			logger.info("Interrupted by user. Exiting...")
			break
		except Exception as e:
			logger.error(f"Loop error: {e}")
			time.sleep(5)


if __name__ == "__main__":
	main()


