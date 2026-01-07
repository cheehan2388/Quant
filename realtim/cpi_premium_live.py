import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import ccxt
import numpy as np
import pandas as pd
import requests


FRED_API_BASE = "https://api.stlouisfed.org/fred"


def setup_logger(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("cpi_premium_live")


def timeframe_to_seconds(timeframe: str) -> int:
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    return value * units[unit]


def wait_until_next_candle(tf_seconds: int, buffer_seconds: int = 2) -> None:
    now = int(datetime.now(timezone.utc).timestamp())
    wait = tf_seconds - (now % tf_seconds) + buffer_seconds
    if wait > 0:
        time.sleep(wait)


def fetch_fred_series(series_id: str, api_key: str, start: str = "2018-01-01") -> pd.Series:
    params = {
        "series_id": series_id,
        "observation_start": start,
        "file_type": "json",
        "api_key": api_key,
        "sort_order": "asc",
    }
    url = f"{FRED_API_BASE}/series/observations"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("observations", [])
    if not data:
        return pd.Series(dtype=float)
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df.set_index("date", inplace=True)
    # FRED CPI values are strings; convert to float and coerce missing to NaN
    values = pd.to_numeric(df["value"], errors="coerce")
    values.name = series_id
    return values


def load_coinbase_premium_index(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    # Expect a column named 'datetime' and 'coinbase_premium_index'
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], utc=True)
    elif "end_time" in df.columns:
        dt = pd.to_datetime(df["end_time"], unit="ms", utc=True)
    else:
        raise ValueError("CSV must contain 'datetime' or 'end_time'")
    df.index = dt
    col = "coinbase_premium_index"
    if col not in df.columns:
        raise ValueError(f"CSV missing '{col}' column")
    s = pd.to_numeric(df[col], errors="coerce")
    s.name = col
    return s.sort_index()


def fetch_binance_close(symbol: str, timeframe: str, limit: int = 500) -> pd.Series:
    ex = ccxt.binance({"enableRateLimit": True})
    ex.load_markets()
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    s = df["close"].astype(float)
    s.name = f"{symbol}_close"
    return s


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window, min_periods=max(3, window // 2)).mean()
    rolling_std = series.rolling(window, min_periods=max(3, window // 2)).std()
    z = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    return z


@dataclass
class StrategyConfig:
    cpi_series_id: str = "CPIAUCSL"  # US CPI All Urban Consumers: All Items in U.S. City Average
    cpi_window: int = 5
    premium_window: int = 10
    cpi_long_threshold: float = 1.5
    cpi_meanrev_threshold: float = -1.5
    premium_long_threshold: float = 1.5


def generate_signal(cpi_z: float, premium_z: float, current_position: Optional[str]) -> str:
    # Rule 1: if CPI z > 1.5 and premium z > 1.5 -> LONG
    if cpi_z is not None and premium_z is not None:
        if cpi_z > 1.5 and premium_z > 1.5:
            return "LONG"
    # Rule 2: if premium index < 1.5 -> CLOSE LONG and OPEN SHORT
    if premium_z is not None and premium_z < 1.5:
        # If we were long, close then short; for simplicity return SHORT
        return "SHORT"
    # Rule 3: if CPI < -1.5 mean reversion (assume LONG as mean-revert)
    if cpi_z is not None and cpi_z < -1.5:
        return "LONG"
    return "HOLD"


def main():
    parser = argparse.ArgumentParser(description="CPI + Coinbase Premium Index Z-Score strategy")
    parser.add_argument("--symbol", default="BTC/USDT", help="CCXT symbol for close")
    parser.add_argument("--timeframe", default="1h", help="Close timeframe (e.g., 1h)")
    parser.add_argument(
        "--premium_csv",
        default=os.path.join(
            "Data",
            "cryptoquant_btc",
            "Bitcoin",
            "BTC_Market_data",
            "Coinbase Premium Index_data",
            "CryptoQuant_BTC_Hour_Binance_CoinbasePremiumIndex.csv",
        ),
        help="Path to Coinbase Premium Index CSV (hourly)",
    )
    parser.add_argument("--fred_series", default="CPIAUCSL", help="FRED CPI series ID")
    parser.add_argument("--fred_api_key", default=os.getenv("FRED_API_KEY", ""), help="FRED API key")
    parser.add_argument("--log", default="INFO", help="Log level")
    args = parser.parse_args()

    logger = setup_logger(args.log)
    if not args.fred_api_key:
        logger.warning("FRED_API_KEY not set; CPI fetch will fail.")

    tf_seconds = timeframe_to_seconds(args.timeframe)

    # Static datasets
    logger.info("Loading Coinbase Premium Index CSV...")
    premium = load_coinbase_premium_index(args.premium_csv)
    premium = premium.dropna()
    premium_z = rolling_zscore(premium, window=10)

    logger.info("Fetching CPI from FRED...")
    cpi = fetch_fred_series(args.fred_series, api_key=args.fred_api_key)
    cpi = cpi.dropna()
    cpi_z = rolling_zscore(cpi, window=5)

    # Prime position
    position: Optional[str] = None

    while True:
        try:
            wait_until_next_candle(tf_seconds, buffer_seconds=2)

            # Fetch latest closes
            close = fetch_binance_close(args.symbol, args.timeframe, limit=500)

            # Align data on the close index
            idx = close.index
            # Forward-fill monthly CPI to intraday index
            cpi_aligned = cpi.reindex(idx, method="ffill")
            cpi_z_aligned = cpi_z.reindex(idx, method="ffill")
            # Align premium (already hourly); forward-fill to close index
            premium_aligned = premium.reindex(idx, method="ffill")
            premium_z_aligned = premium_z.reindex(idx, method="ffill")

            last_time = idx[-1]
            last_close = float(close.iloc[-1])
            last_cpi_z = float(cpi_z_aligned.iloc[-1]) if not np.isnan(cpi_z_aligned.iloc[-1]) else np.nan
            last_prem_z = float(premium_z_aligned.iloc[-1]) if not np.isnan(premium_z_aligned.iloc[-1]) else np.nan

            signal = generate_signal(last_cpi_z, last_prem_z, position)

            logger.info(
                f"{last_time.strftime('%Y-%m-%d %H:%M:%S %Z')} | close={last_close:.2f} | "
                f"cpi_z={last_cpi_z:.2f} | prem_z={last_prem_z:.2f} | signal={signal}"
            )

            # Position bookkeeping (no real trading here)
            if signal == "LONG" and position != "LONG":
                logger.info("-> OPEN LONG (simulated)")
                position = "LONG"
            elif signal == "SHORT" and position != "SHORT":
                if position == "LONG":
                    logger.info("-> CLOSE LONG (simulated)")
                logger.info("-> OPEN SHORT (simulated)")
                position = "SHORT"
            else:
                logger.info("-> HOLD")

        except KeyboardInterrupt:
            logger.info("Interrupted by user. Exiting...")
            break
        except Exception as e:
            logger.error(f"Loop error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()


