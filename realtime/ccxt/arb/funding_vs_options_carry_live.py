import argparse
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import ccxt

# Local helpers (same directory)
import deribit_helpers as dh


# ----------------------------- Logging -----------------------------
def setup_logger(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("funding_vs_options_carry")


# ----------------------------- Data models -----------------------------
@dataclass
class CarrySnapshot:
    ts: datetime
    price_usd: float
    funding_rate_8h: Optional[float]
    daily_funding_rate: Optional[float]
    daily_funding_usd: Optional[float]
    call_symbol: Optional[str]
    put_symbol: Optional[str]
    call_theta_btc_per_day: Optional[float]
    put_theta_btc_per_day: Optional[float]
    straddle_theta_btc_per_day: Optional[float]
    straddle_theta_usd_per_day: Optional[float]
    net_daily_usd_funding_minus_long_straddle: Optional[float]
    net_daily_usd_short_straddle_minus_funding: Optional[float]


# ----------------------------- Helpers -----------------------------
def timeframe_to_seconds(timeframe: str) -> int:
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    return value * units[unit]


def wait_until_next(tf_seconds: int, buffer_seconds: int = 2) -> None:
    now = int(datetime.now(timezone.utc).timestamp())
    wait = tf_seconds - (now % tf_seconds) + buffer_seconds
    if wait > 0:
        time.sleep(wait)


def initialize_perp_exchange(perp_exchange_id: str) -> ccxt.Exchange:
    if not hasattr(ccxt, perp_exchange_id):
        raise ValueError(f"Unsupported exchange id: {perp_exchange_id}")
    ex_class = getattr(ccxt, perp_exchange_id)

    # Try exchange-specific env first (e.g., BYBIT_API_KEY), then generic (PERP_API_KEY)
    env_api_key = os.getenv(f"{perp_exchange_id.upper()}_API_KEY") or os.getenv("PERP_API_KEY")
    env_api_secret = os.getenv(f"{perp_exchange_id.upper()}_API_SECRET") or os.getenv("PERP_API_SECRET")

    params = {
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    }
    if env_api_key and env_api_secret:
        params["apiKey"] = env_api_key
        params["secret"] = env_api_secret

    exchange = ex_class(params)
    exchange.load_markets()
    return exchange


def fetch_price(exchange: ccxt.Exchange, symbol: str) -> Optional[float]:
    try:
        t = exchange.fetch_ticker(symbol)
        price = t.get("last") or t.get("info", {}).get("last_price") or t.get("info", {}).get("index_price")
        return float(price) if price is not None else None
    except Exception:
        return None


def fetch_funding_rate_daily(exchange: ccxt.Exchange, symbol: str, default_interval_hours: int = 8) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (funding_rate_8h, daily_rate), both as fractions (e.g., 0.01 = 1%).
    Falls back to default 8h if interval unknown.
    """
    try:
        fr = exchange.fetch_funding_rate(symbol)
        # Prefer nextFundingRate if available, else current fundingRate
        rate = None
        for k in ["nextFundingRate", "fundingRate", "info"]:
            if k in fr and fr[k] is not None:
                if k == "info":
                    # common info locations
                    info = fr["info"]
                    for key in ["fundingRate", "predictedFundingRate", "nextFundingRate", "lastFundingRate"]:
                        if isinstance(info, dict) and info.get(key) is not None:
                            try:
                                rate = float(info[key])
                                break
                            except Exception:
                                pass
                    if rate is not None:
                        break
                else:
                    try:
                        rate = float(fr[k])
                        break
                    except Exception:
                        pass

        if rate is None:
            return None, None

        # Normalize to daily
        interval_hours = default_interval_hours
        if isinstance(fr.get("fundingIntervalHours"), (int, float)) and fr["fundingIntervalHours"] > 0:
            interval_hours = float(fr["fundingIntervalHours"])
        daily_rate = rate * (24.0 / float(interval_hours))
        return rate, daily_rate
    except Exception:
        return None, None


def find_atm_straddle_symbols(deribit: ccxt.Exchange, underlying_price: float, currency: str = "BTC") -> Tuple[Optional[str], Optional[str]]:
    option_markets = dh.get_option_markets(deribit, currency=currency)
    expiry_pairs = dh.group_atm_pairs_by_expiry(option_markets, underlying_price)
    near, _ = dh.pick_near_far_expiries(expiry_pairs)
    if near is None:
        return None, None
    pair = expiry_pairs[near]
    call_m = pair.get("call")
    put_m = pair.get("put")
    call_symbol = call_m.get("symbol") if isinstance(call_m, dict) else None
    put_symbol = put_m.get("symbol") if isinstance(put_m, dict) else None
    return call_symbol, put_symbol


def fetch_straddle_theta_usd_per_day(deribit: ccxt.Exchange, call_symbol: str, put_symbol: str, underlying_price: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns (theta_call_btc_per_day, theta_put_btc_per_day, theta_straddle_usd_per_day)
    Theta is expected per-day decay of option price. On Deribit it is quoted in base currency (BTC).
    For USD value, multiply by underlying USD price.
    """
    iv_c, greeks_c = dh.fetch_ticker_iv_and_greeks(deribit, call_symbol)
    iv_p, greeks_p = dh.fetch_ticker_iv_and_greeks(deribit, put_symbol)
    theta_c = float(greeks_c.get("theta")) if isinstance(greeks_c.get("theta"), (int, float)) else None
    theta_p = float(greeks_p.get("theta")) if isinstance(greeks_p.get("theta"), (int, float)) else None
    if theta_c is None or theta_p is None:
        return theta_c, theta_p, None
    theta_straddle_btc = theta_c + theta_p
    theta_straddle_usd = theta_straddle_btc * float(underlying_price)
    return theta_c, theta_p, theta_straddle_usd


def compute_snapshot(
    logger: logging.Logger,
    perp: ccxt.Exchange,
    perp_symbol: str,
    deribit: ccxt.Exchange,
    size_btc: float,
) -> CarrySnapshot:
    # Price from perp venue for consistency
    price = fetch_price(perp, perp_symbol)
    if price is None:
        # Fallback to Deribit perp
        price = fetch_price(deribit, "BTC-PERPETUAL")
    if price is None:
        raise RuntimeError("Could not fetch underlying price")

    # Funding
    fr_8h, fr_daily = fetch_funding_rate_daily(perp, perp_symbol)
    daily_funding_usd = None
    if fr_daily is not None:
        daily_funding_usd = float(price) * float(fr_daily) * float(size_btc)

    # Options: ATM straddle (near expiry)
    call_sym, put_sym = find_atm_straddle_symbols(deribit, underlying_price=float(price), currency="BTC")
    theta_c_btc, theta_p_btc, theta_straddle_usd = (None, None, None)
    if call_sym and put_sym:
        theta_c_btc, theta_p_btc, theta_straddle_usd = fetch_straddle_theta_usd_per_day(
            deribit, call_sym, put_sym, float(price)
        )

    # Net comparisons (positive = attractive)
    net_funding_minus_long_straddle = None
    net_short_straddle_minus_funding = None
    if daily_funding_usd is not None and theta_straddle_usd is not None:
        # Long straddle has negative theta (theta_straddle_usd typically < 0). Cost = -theta.
        cost_long_straddle_usd = -theta_straddle_usd * float(size_btc)
        net_funding_minus_long_straddle = daily_funding_usd - cost_long_straddle_usd

        # Short straddle earns -theta; hedging with perp may incur/payout funding intermittently.
        earn_short_straddle_usd = -theta_straddle_usd * float(size_btc)
        # Approximate net as theta income minus absolute funding magnitude for the same notional
        net_short_straddle_minus_funding = earn_short_straddle_usd - max(0.0, daily_funding_usd)

    return CarrySnapshot(
        ts=datetime.now(timezone.utc),
        price_usd=float(price),
        funding_rate_8h=fr_8h,
        daily_funding_rate=fr_daily,
        daily_funding_usd=daily_funding_usd,
        call_symbol=call_sym,
        put_symbol=put_sym,
        call_theta_btc_per_day=theta_c_btc,
        put_theta_btc_per_day=theta_p_btc,
        straddle_theta_btc_per_day=(None if (theta_c_btc is None or theta_p_btc is None) else theta_c_btc + theta_p_btc),
        straddle_theta_usd_per_day=(None if theta_straddle_usd is None else float(theta_straddle_usd)),
        net_daily_usd_funding_minus_long_straddle=net_funding_minus_long_straddle,
        net_daily_usd_short_straddle_minus_funding=net_short_straddle_minus_funding,
    )


def format_snapshot(s: CarrySnapshot, size_btc: float) -> str:
    def fmt(x: Optional[float], pct: bool = False) -> str:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "na"
        return (f"{x*100:.4f}%" if pct else f"{x:.6f}")

    lines = []
    lines.append(f"ts={s.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} | px={s.price_usd:.2f} | size={size_btc}")
    lines.append(
        f"funding_8h={fmt(s.funding_rate_8h, pct=True)} | daily_rate={fmt(s.daily_funding_rate, pct=True)} | daily_funding_usd={fmt(s.daily_funding_usd)}"
    )
    lines.append(
        f"ATM call={s.call_symbol or 'na'} put={s.put_symbol or 'na'} | theta_c_btc/d={fmt(s.call_theta_btc_per_day)} | theta_p_btc/d={fmt(s.put_theta_btc_per_day)}"
    )
    lines.append(
        f"straddle_theta_btc/d={fmt(s.straddle_theta_btc_per_day)} | straddle_theta_usd/d={fmt(s.straddle_theta_usd_per_day)}"
    )
    lines.append(
        f"net: funding - long_straddle = {fmt(s.net_daily_usd_funding_minus_long_straddle)} | short_straddle - funding = {fmt(s.net_daily_usd_short_straddle_minus_funding)}"
    )
    return " | ".join(lines)


# ----------------------------- Execution (optional) -----------------------------
def place_perp_order(exchange: ccxt.Exchange, symbol: str, side: str, amount: float, logger: logging.Logger, dry_run: bool = True) -> Optional[Dict]:
    if dry_run:
        logger.info("[DRY-RUN] PERP %s %s x %.4f", side.upper(), symbol, amount)
        return None
    try:
        order = exchange.create_order(symbol, "market", side, amount)
        logger.info("Perp order placed: %s", order.get("id", "<no-id>"))
        return order
    except Exception as e:
        logger.error("Perp order failed: %s", e)
        return None


def open_structure(
    logger: logging.Logger,
    side: str,
    perp: ccxt.Exchange,
    perp_symbol: str,
    deribit: ccxt.Exchange,
    call_symbol: str,
    put_symbol: str,
    size_btc: float,
    dry_run: bool,
) -> None:
    """
    side in {"funding_minus_long_straddle", "short_straddle_minus_funding"}
    This is a simplified execution: opens core legs only (no continuous re-hedging here).
    """
    if side == "funding_minus_long_straddle":
        # Short perp to receive funding, buy straddle (delta ~ 0 near-ATM)
        place_perp_order(perp, perp_symbol, "sell", size_btc, logger, dry_run=dry_run)
        dh.place_deribit_order(deribit, call_symbol, "buy", size_btc, logger, dry_run=dry_run)
        dh.place_deribit_order(deribit, put_symbol, "buy", size_btc, logger, dry_run=dry_run)
    elif side == "short_straddle_minus_funding":
        # Sell straddle to earn theta; hedging via perp not automated here
        dh.place_deribit_order(deribit, call_symbol, "sell", size_btc, logger, dry_run=dry_run)
        dh.place_deribit_order(deribit, put_symbol, "sell", size_btc, logger, dry_run=dry_run)
    else:
        logger.warning("Unknown structure side: %s", side)


# ----------------------------- Main loop -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Funding vs Options Delta-Hedged Carry (monitor + optional execution)")
    parser.add_argument("--perp_exchange", default=os.getenv("PERP_EXCHANGE", "bybit"), help="Perpetual venue id in ccxt (e.g., bybit, binance)")
    parser.add_argument("--perp_symbol", default=os.getenv("PERP_SYMBOL", "BTC/USDT:USDT"), help="Perp symbol (e.g., BTC/USDT:USDT for Bybit)")
    parser.add_argument("--size_btc", type=float, default=float(os.getenv("SIZE_BTC", 0.1)), help="Position size in BTC notional")
    parser.add_argument("--deribit_testnet", action="store_true", help="Use Deribit testnet")
    parser.add_argument("--no-deribit_testnet", dest="deribit_testnet", action="store_false")
    parser.set_defaults(deribit_testnet=True)
    parser.add_argument("--interval", default=os.getenv("INTERVAL", "5m"), help="Refresh cadence (e.g., 1m, 5m, 1h)")
    parser.add_argument("--log", default=os.getenv("LOG_LEVEL", "INFO"), help="Log level")
    parser.add_argument("--dry_run", action="store_true", help="Do not place real orders")
    parser.add_argument("--execute", action="store_true", help="Attempt to execute the better carry structure when above threshold")
    parser.add_argument("--threshold_usd_per_day", type=float, default=float(os.getenv("THRESHOLD_USD_PER_DAY", 10.0)), help="Min absolute net daily carry to act")
    args = parser.parse_args()

    logger = setup_logger(args.log)
    logger.info("Starting Funding vs Options Carry | perp=%s %s | size=%.4f BTC | testnet=%s | dry_run=%s",
                args.perp_exchange, args.perp_symbol, args.size_btc, args.deribit_testnet, args.dry_run)

    # Initialize exchanges
    perp = initialize_perp_exchange(args.perp_exchange)
    deribit = dh.initialize_deribit(logger=logger, testnet=args.deribit_testnet)

    tf_seconds = timeframe_to_seconds(args.interval)

    try:
        while True:
            try:
                s = compute_snapshot(logger, perp, args.perp_symbol, deribit, args.size_btc)
                logger.info(format_snapshot(s, args.size_btc))

                # Decide side
                best_side = None
                best_value = -float("inf")
                if s.net_daily_usd_funding_minus_long_straddle is not None:
                    if s.net_daily_usd_funding_minus_long_straddle > best_value:
                        best_value = s.net_daily_usd_funding_minus_long_straddle
                        best_side = "funding_minus_long_straddle"
                if s.net_daily_usd_short_straddle_minus_funding is not None:
                    if s.net_daily_usd_short_straddle_minus_funding > best_value:
                        best_value = s.net_daily_usd_short_straddle_minus_funding
                        best_side = "short_straddle_minus_funding"

                if best_side and best_value >= args.threshold_usd_per_day and args.execute and s.call_symbol and s.put_symbol:
                    logger.info("Executing structure=%s (net_daily_usd=%.4f >= threshold=%.4f)", best_side, best_value, args.threshold_usd_per_day)
                    open_structure(
                        logger=logger,
                        side=best_side,
                        perp=perp,
                        perp_symbol=args.perp_symbol,
                        deribit=deribit,
                        call_symbol=s.call_symbol,
                        put_symbol=s.put_symbol,
                        size_btc=args.size_btc,
                        dry_run=args.dry_run or not args.execute,
                    )
                else:
                    if best_side is None:
                        logger.info("No comparable signals yet (missing data). Holding.")
                    else:
                        logger.info("Best structure=%s but below threshold (%.4f < %.4f) or execution disabled.", best_side, best_value, args.threshold_usd_per_day)

            except KeyboardInterrupt:
                logger.info("Interrupted by user. Exiting...")
                break
            except Exception as e:
                logger.error("Loop error: %s", e)
                time.sleep(5)

            wait_until_next(tf_seconds=tf_seconds, buffer_seconds=1)
    finally:
        try:
            perp.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()


