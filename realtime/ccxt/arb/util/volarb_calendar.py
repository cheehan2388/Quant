import argparse
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np

from .deribit_helpers import (
    initialize_deribit,
    fetch_underlying_close_series,
    get_option_markets,
    group_atm_pairs_by_expiry,
    pick_near_far_expiries,
    fetch_ticker_iv_and_greeks,
    place_deribit_order,
)


def setup_logger(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("volarb_calendar")


@dataclass
class CalendarConfig:
    currency: str = "BTC"
    underlying_symbol: str = "BTC-PERPETUAL"
    iv_term_threshold: float = 0.10
    min_days_to_expiry: float = 2.0
    contracts_per_leg: float = 1.0
    dry_run: bool = True
    testnet: bool = True


def select_near_far_pairs(
    logger: logging.Logger,
    exchange,
    currency: str,
    underlying_price: float,
    min_days_to_expiry: float,
) -> Tuple[Optional[int], Optional[int], Optional[Dict[str, str]], Optional[Dict[str, str]]]:
    option_markets = get_option_markets(exchange, currency)
    expiry_to_pair = group_atm_pairs_by_expiry(option_markets, underlying_price)
    if not expiry_to_pair:
        logger.warning("No ATM option pairs found")
        return None, None, None, None

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    # filter by min days
    expiry_to_pair = {
        k: v
        for k, v in expiry_to_pair.items()
        if (k - now_ms) / 1000 / 86400.0 >= min_days_to_expiry
    }
    if not expiry_to_pair:
        logger.warning("No expiries beyond min days")
        return None, None, None, None

    near, far = pick_near_far_expiries(expiry_to_pair)
    if near is None or far is None:
        logger.warning("Need at least two expiries for calendar spread")
        return None, None, None, None

    near_symbols = {
        "call": expiry_to_pair[near]["call"]["symbol"],
        "put": expiry_to_pair[near]["put"]["symbol"],
    }
    far_symbols = {
        "call": expiry_to_pair[far]["call"]["symbol"],
        "put": expiry_to_pair[far]["put"]["symbol"],
    }
    return near, far, near_symbols, far_symbols


def run_once(cfg: CalendarConfig, logger: logging.Logger) -> None:
    exchange = initialize_deribit(logger, testnet=cfg.testnet)
    ticker = exchange.fetch_ticker(cfg.underlying_symbol)
    underlying_price = float(ticker.get("last"))

    near_ms, far_ms, near_syms, far_syms = select_near_far_pairs(
        logger, exchange, cfg.currency, underlying_price, cfg.min_days_to_expiry
    )
    if near_ms is None or far_ms is None or near_syms is None or far_syms is None:
        return

    near_call_iv, _ = fetch_ticker_iv_and_greeks(exchange, near_syms["call"])
    near_put_iv, _ = fetch_ticker_iv_and_greeks(exchange, near_syms["put"])
    far_call_iv, _ = fetch_ticker_iv_and_greeks(exchange, far_syms["call"])
    far_put_iv, _ = fetch_ticker_iv_and_greeks(exchange, far_syms["put"])

    if None in (near_call_iv, near_put_iv, far_call_iv, far_put_iv):
        logger.error("Missing IV for one or more legs")
        return

    near_iv = float(np.nanmean([near_call_iv, near_put_iv]))
    far_iv = float(np.nanmean([far_call_iv, far_put_iv]))
    term_diff = far_iv - near_iv

    logger.info(
        "NearIV=%.3f FarIV=%.3f Diff(Far-Near)=%.3f threshold=%.3f | near(C,P)=(%s,%s) far(C,P)=(%s,%s)",
        near_iv,
        far_iv,
        term_diff,
        cfg.iv_term_threshold,
        near_syms["call"],
        near_syms["put"],
        far_syms["call"],
        far_syms["put"],
    )

    if term_diff > cfg.iv_term_threshold:
        logger.info("Signal: SHORT CALENDAR (sell far straddle, buy near straddle)")
        place_deribit_order(exchange, far_syms["call"], "sell", cfg.contracts_per_leg, logger, cfg.dry_run)
        place_deribit_order(exchange, far_syms["put"], "sell", cfg.contracts_per_leg, logger, cfg.dry_run)
        place_deribit_order(exchange, near_syms["call"], "buy", cfg.contracts_per_leg, logger, cfg.dry_run)
        place_deribit_order(exchange, near_syms["put"], "buy", cfg.contracts_per_leg, logger, cfg.dry_run)
    elif term_diff < -cfg.iv_term_threshold:
        logger.info("Signal: LONG CALENDAR (buy far straddle, sell near straddle)")
        place_deribit_order(exchange, far_syms["call"], "buy", cfg.contracts_per_leg, logger, cfg.dry_run)
        place_deribit_order(exchange, far_syms["put"], "buy", cfg.contracts_per_leg, logger, cfg.dry_run)
        place_deribit_order(exchange, near_syms["call"], "sell", cfg.contracts_per_leg, logger, cfg.dry_run)
        place_deribit_order(exchange, near_syms["put"], "sell", cfg.contracts_per_leg, logger, cfg.dry_run)
    else:
        logger.info("No trade: term structure within threshold")


def main():
    parser = argparse.ArgumentParser(description="Deribit ATM calendar spread arbitrage")
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--underlying", default="BTC-PERPETUAL")
    parser.add_argument("--threshold", type=float, default=0.10, help="IV far-near threshold")
    parser.add_argument("--min_days", type=float, default=2.0, help="Min days to expiry")
    parser.add_argument("--contracts", type=float, default=1.0)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--prod", action="store_true")
    parser.add_argument("--log", default="INFO")
    args = parser.parse_args()

    logger = setup_logger(args.log)
    cfg = CalendarConfig(
        currency=args.currency,
        underlying_symbol=args.underlying,
        iv_term_threshold=args.threshold,
        min_days_to_expiry=args.min_days,
        contracts_per_leg=args.contracts,
        dry_run=args.dry,
        testnet=not args.prod,
    )

    run_once(cfg, logger)


if __name__ == "__main__":
    main()


