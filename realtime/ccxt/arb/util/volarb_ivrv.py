import argparse
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np

from .deribit_helpers import (
    initialize_deribit,
    fetch_underlying_close_series,
    compute_realized_vol_annualized,
    get_option_markets,
    group_atm_pairs_by_expiry,
    fetch_ticker_iv_and_greeks,
    place_deribit_order,
    timeframe_to_seconds,
    wait_until_next_candle,
)


def setup_logger(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("volarb_ivrv")


@dataclass
class IVRVConfig:
    currency: str = "BTC"
    underlying_symbol: str = "BTC-PERPETUAL"
    timeframe: str = "1h"
    lookback: int = 500
    iv_minus_rv_threshold: float = 0.10
    min_days_to_expiry: float = 2.0
    contracts_per_leg: float = 1.0
    dry_run: bool = True
    testnet: bool = True
    loop: bool = False
    loop_sleep_seconds: int = 60


def pick_best_expiry_iv(
    logger: logging.Logger,
    exchange,
    currency: str,
    underlying_price: float,
    min_days_to_expiry: float,
) -> Tuple[Optional[int], Optional[str], Optional[str], Optional[float]]:
    option_markets = get_option_markets(exchange, currency)
    expiry_to_pair = group_atm_pairs_by_expiry(option_markets, underlying_price)
    if not expiry_to_pair:
        logger.warning("No ATM option pairs found for %s", currency)
        return None, None, None, None

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    best_expiry = None
    best_iv = None
    best_call_symbol = None
    best_put_symbol = None

    for expiry_ms, pair in sorted(expiry_to_pair.items()):
        days = (expiry_ms - now_ms) / 1000 / 86400.0
        if days < min_days_to_expiry:
            continue
        call_symbol = pair["call"]["symbol"]
        put_symbol = pair["put"]["symbol"]
        call_iv, _ = fetch_ticker_iv_and_greeks(exchange, call_symbol)
        put_iv, _ = fetch_ticker_iv_and_greeks(exchange, put_symbol)
        if call_iv is None or put_iv is None:
            continue
        atm_iv = float(np.nanmean([call_iv, put_iv]))
        if math.isnan(atm_iv):
            continue
        best_expiry = expiry_ms
        best_iv = atm_iv
        best_call_symbol = call_symbol
        best_put_symbol = put_symbol
        break

    return best_expiry, best_call_symbol, best_put_symbol, best_iv


def run_once(cfg: IVRVConfig, logger: logging.Logger) -> None:
    exchange = initialize_deribit(logger, testnet=cfg.testnet)

    close = fetch_underlying_close_series(
        exchange, symbol=cfg.underlying_symbol, timeframe=cfg.timeframe, limit=cfg.lookback
    )
    if close.empty:
        logger.error("No underlying close data")
        return

    rv = compute_realized_vol_annualized(close)
    if rv is None or math.isnan(rv):
        logger.error("Failed to compute realized vol")
        return

    ticker = exchange.fetch_ticker(cfg.underlying_symbol)
    underlying_price = float(ticker.get("last"))

    expiry_ms, call_symbol, put_symbol, atm_iv = pick_best_expiry_iv(
        logger, exchange, cfg.currency, underlying_price, cfg.min_days_to_expiry
    )
    if expiry_ms is None or call_symbol is None or put_symbol is None or atm_iv is None:
        logger.error("No suitable ATM expiry found")
        return

    iv_minus_rv = atm_iv - rv
    logger.info(
        "RV=%.3f IV=%.3f IV-RV=%.3f threshold=%.3f | call=%s put=%s",
        rv,
        atm_iv,
        iv_minus_rv,
        cfg.iv_minus_rv_threshold,
        call_symbol,
        put_symbol,
    )

    if iv_minus_rv > cfg.iv_minus_rv_threshold:
        logger.info("Signal: SHORT VOL (sell straddle)")
        place_deribit_order(exchange, call_symbol, "sell", cfg.contracts_per_leg, logger, cfg.dry_run)
        place_deribit_order(exchange, put_symbol, "sell", cfg.contracts_per_leg, logger, cfg.dry_run)
    elif iv_minus_rv < -cfg.iv_minus_rv_threshold:
        logger.info("Signal: LONG VOL (buy straddle)")
        place_deribit_order(exchange, call_symbol, "buy", cfg.contracts_per_leg, logger, cfg.dry_run)
        place_deribit_order(exchange, put_symbol, "buy", cfg.contracts_per_leg, logger, cfg.dry_run)
    else:
        logger.info("No trade: IV-RV within threshold")


def main():
    parser = argparse.ArgumentParser(description="Deribit IV vs RV volatility arbitrage")
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--underlying", default="BTC-PERPETUAL")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--lookback", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--min_days", type=float, default=2.0, help="Min days to expiry")
    parser.add_argument("--contracts", type=float, default=1.0, help="Contracts per leg")
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--prod", action="store_true", help="Use production instead of testnet")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--sleep", type=int, default=60)
    parser.add_argument("--log", default="INFO")
    args = parser.parse_args()

    logger = setup_logger(args.log)
    cfg = IVRVConfig(
        currency=args.currency,
        underlying_symbol=args.underlying,
        timeframe=args.timeframe,
        lookback=args.lookback,
        iv_minus_rv_threshold=args.threshold,
        min_days_to_expiry=args.min_days,
        contracts_per_leg=args.contracts,
        dry_run=args.dry,
        testnet=not args.prod,
        loop=args.loop,
        loop_sleep_seconds=args.sleep,
    )

    if not cfg.loop:
        run_once(cfg, logger)
    else:
        tf_seconds = timeframe_to_seconds(cfg.timeframe)
        while True:
            try:
                wait_until_next_candle(tf_seconds, buffer_seconds=2)
                run_once(cfg, logger)
            except KeyboardInterrupt:
                logger.info("Interrupted. Exiting loop.")
                break
            except Exception as e:
                logger.error("Loop error: %s", e)


if __name__ == "__main__":
    main()


