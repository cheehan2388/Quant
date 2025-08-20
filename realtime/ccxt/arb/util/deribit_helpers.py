import os
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd


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


def initialize_deribit(
    logger: logging.Logger,
    testnet: bool = True,
) -> ccxt.Exchange:
    client_id = os.getenv("DERIBIT_API_KEY", "")
    client_secret = os.getenv("DERIBIT_API_SECRET", "")

    exchange_class = ccxt.deribit
    exchange = exchange_class(
        {
            "apiKey": client_id,
            "secret": client_secret,
            "enableRateLimit": True,
            "options": {
                # Use testnet if requested
                "defaultType": "swap",
            },
        }
    )

    if testnet:
        exchange.urls["api"] = exchange.urls["test"]

    exchange.load_markets()
    logger.info(
        "Initialized Deribit (%s). Auth=%s",
        "testnet" if testnet else "prod",
        "yes" if client_id and client_secret else "no",
    )
    return exchange


def fetch_underlying_close_series(
    exchange: ccxt.Exchange,
    symbol: str = "BTC-PERPETUAL",
    timeframe: str = "1h",
    limit: int = 1000,
) -> pd.Series:
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(
        data, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    s = df["close"].astype(float)
    s.name = f"{symbol}_close"
    return s


def compute_realized_vol_annualized(
    close_series: pd.Series, periods_per_year: Optional[int] = None
) -> float:
    if close_series.empty or len(close_series) < 10:
        return float("nan")
    rets = np.log(close_series).diff().dropna()
    if periods_per_year is None:
        # Infer from index spacing using seconds between last two points
        if hasattr(close_series.index, "asi8") and len(close_series.index) >= 2:
            dt_seconds = (
                close_series.index[-1].to_pydatetime().timestamp()
                - close_series.index[-2].to_pydatetime().timestamp()
            )
            if dt_seconds > 0:
                periods_per_year = int(round(365 * 24 * 3600 / dt_seconds))
            else:
                periods_per_year = 365 * 24
        else:
            periods_per_year = 365 * 24
    vol = float(rets.std(ddof=1)) * float(np.sqrt(periods_per_year))
    return vol


def get_option_markets(
    exchange: ccxt.Exchange, currency: str = "BTC"
) -> List[Dict]:
    markets = []
    for symbol, m in exchange.markets.items():
        if not m.get("option") and m.get("type") != "option":
            continue
        base = m.get("base") or m.get("underlying") or m.get("info", {}).get("base_currency")
        if base != currency:
            continue
        if not m.get("active", True):
            continue
        markets.append(m)
    return markets


def extract_option_fields(market: Dict) -> Tuple[Optional[float], Optional[int], Optional[str]]:
    strike = None
    expiry_ms = None
    opt_type = None

    if "strike" in market and isinstance(market["strike"], (float, int)):
        strike = float(market["strike"])
    elif "option" in market and isinstance(market["option"], dict):
        if "strike" in market["option"]:
            strike = float(market["option"]["strike"])
        if "expiry" in market["option"]:
            expiry_ms = int(market["option"]["expiry"]) if market["option"]["expiry"] else None
        if "type" in market["option"]:
            opt_type = market["option"]["type"].lower()

    info = market.get("info", {})
    if strike is None and "strike" in info:
        try:
            strike = float(info["strike"]) if info["strike"] is not None else None
        except Exception:
            strike = None
    if expiry_ms is None:
        ts_key = None
        for k in ["expiration_timestamp", "exp_timestamp", "expiry_timestamp"]:
            if k in info:
                ts_key = k
                break
        if ts_key is not None and info.get(ts_key) is not None:
            expiry_ms = int(info[ts_key])
    if opt_type is None:
        for k in ["option_type", "type", "optionType"]:
            if isinstance(info.get(k), str):
                val = info[k].lower()
                if val in ("call", "put"):
                    opt_type = val
                    break

    return strike, expiry_ms, opt_type


def group_atm_pairs_by_expiry(
    option_markets: List[Dict], underlying_price: float
) -> Dict[int, Dict[str, Dict]]:
    groups: Dict[int, Dict[str, Dict]] = {}
    for m in option_markets:
        strike, expiry_ms, opt_type = extract_option_fields(m)
        if strike is None or expiry_ms is None or opt_type not in ("call", "put"):
            continue
        if expiry_ms not in groups:
            groups[expiry_ms] = {"call": None, "put": None}
        current = groups[expiry_ms][opt_type]
        if current is None:
            groups[expiry_ms][opt_type] = m
        else:
            cur_strike, _, _ = extract_option_fields(current)
            if abs(strike - underlying_price) < abs(float(cur_strike) - underlying_price):
                groups[expiry_ms][opt_type] = m
    # Keep only expiries where both sides exist
    return {k: v for k, v in groups.items() if v["call"] is not None and v["put"] is not None}


def pick_near_far_expiries(expiry_to_pair: Dict[int, Dict[str, Dict]]) -> Tuple[Optional[int], Optional[int]]:
    if not expiry_to_pair:
        return None, None
    expiries = sorted(expiry_to_pair.keys())
    if len(expiries) == 1:
        return expiries[0], None
    return expiries[0], expiries[min(1, len(expiries) - 1)]


def fetch_ticker_iv_and_greeks(exchange: ccxt.Exchange, symbol: str) -> Tuple[Optional[float], Dict]:
    t = exchange.fetch_ticker(symbol)
    info = t.get("info", {}) if isinstance(t, dict) else {}
    iv = None
    greeks: Dict = {}
    if isinstance(info, dict):
        if "mark_iv" in info and info["mark_iv"] is not None:
            try:
                iv = float(info["mark_iv"]) / 100.0 if float(info["mark_iv"]) > 1.0 else float(info["mark_iv"])  # deribit iv sometimes in %
            except Exception:
                iv = None
        for k in ["delta", "gamma", "vega", "theta"]:
            if k in info and info[k] is not None:
                try:
                    greeks[k] = float(info[k])
                except Exception:
                    pass
        if "greeks" in info and isinstance(info["greeks"], dict):
            for k, v in info["greeks"].items():
                if k not in greeks and v is not None:
                    try:
                        greeks[k] = float(v)
                    except Exception:
                        pass
    return iv, greeks


def place_deribit_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    amount: float,
    logger: logging.Logger,
    dry_run: bool = True,
) -> Optional[Dict]:
    if dry_run:
        logger.info("[DRY-RUN] %s %s x %.4f", side.upper(), symbol, amount)
        return None
    try:
        order = exchange.create_order(symbol, "market", side, amount)
        logger.info("Order placed: %s", order.get("id", "<no-id>"))
        return order
    except Exception as e:
        logger.error("Order failed for %s %s x %.4f: %s", side, symbol, amount, e)
        return None


