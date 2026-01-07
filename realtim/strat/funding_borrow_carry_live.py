import argparse
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone

import ccxt


def setup_logger(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("funding_borrow_carry")


class PositionManager:
    def __init__(self) -> None:
        self.mode = None  # "stable_borrow" or "asset_borrow"
        self.base_qty = 0.0
        self.entry_price = 0.0
        self.open_time = None

    def is_open(self) -> bool:
        return bool(self.mode)

    def open(self, mode: str, base_qty: float, entry_price: float) -> None:
        self.mode = mode
        self.base_qty = float(base_qty)
        self.entry_price = float(entry_price)
        self.open_time = int(time.time())

    def close(self) -> None:
        self.mode = None
        self.base_qty = 0.0
        self.entry_price = 0.0
        self.open_time = None


def initialize_binance_clients(api_key: str, api_secret: str, sandbox: bool = False):
    spot = ccxt.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",
            "adjustForTimeDifference": True,
        },
        "sandbox": sandbox,
    })

    futures = ccxt.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {
            # Binance USD-M perpetuals
            "defaultType": "future",
            "adjustForTimeDifference": True,
        },
        "sandbox": sandbox,
    })

    spot.load_markets()
    futures.load_markets()
    return spot, futures


def get_binance_pair(base: str, quote: str) -> str:
    return f"{base}{quote}"


def fetch_price(futures: ccxt.Exchange, symbol_ccxt: str) -> float:
    ticker = futures.fetch_ticker(symbol_ccxt)
    last = ticker.get("last") or ticker.get("close")
    if last is None:
        raise RuntimeError("No price in ticker")
    return float(last)


def fetch_funding_rate_usdm_raw(futures: ccxt.Exchange, pair_binance: str) -> float:
    # Uses Binance USD-M Premium Index endpoint; returns last 8h funding rate (fraction per 8h)
    data = futures.fapiPublicGetPremiumIndex({"symbol": pair_binance})
    # API returns either a list or a dict depending on symbol param; handle both
    if isinstance(data, list):
        data = data[0] if data else None
    if not data:
        raise RuntimeError("premiumIndex returned empty response")
    last_fr = float(data.get("lastFundingRate", 0.0))
    return last_fr


def fetch_borrow_rate_daily_binance(spot: ccxt.Exchange, asset: str, logger: logging.Logger) -> float:
    # Try ccxt unified if available
    try:
        if hasattr(spot, "fetch_borrow_rate"):
            result = spot.fetch_borrow_rate(asset)
            # Some exchanges return annual rate; Binance returns dailyInterestRate
            # Normalize to daily fraction
            daily = result.get("rate") or result.get("dailyRate") or result.get("dailyInterestRate")
            if daily is not None:
                return float(daily)
    except Exception as e:
        logger.debug(f"fetch_borrow_rate unified failed for {asset}: {e}")

    # Raw SAPI fallback
    try:
        history = spot.sapiGetMarginInterestRateHistory({
            "asset": asset,
            "vipLevel": 0,
            "limit": 1,
        })
        if history:
            return float(history[-1]["dailyInterestRate"])  # already a fraction per day
    except Exception as e:
        logger.warning(f"Failed to fetch borrow rate for {asset}, falling back to default. Error: {e}")

    # Conservative default daily rates if API not available
    defaults = {"USDT": 0.00025, "USDC": 0.00025}
    return defaults.get(asset.upper(), 0.00035)


def estimate_carry(
    funding_per_8h: float,
    price: float,
    notional_usd: float,
    borrow_daily_quote: float,
    borrow_daily_base: float,
    spot_taker: float,
    futures_taker: float,
    hold_days: float,
):
    # Convert funding per-8h to per-day assuming 3 intervals/day
    funding_per_day = funding_per_8h * 3.0

    # Fees: open + close
    one_off_fees = (spot_taker + futures_taker) * 2.0 * notional_usd

    # Stable-borrow mode (funding positive): long spot (with borrowed quote), short perp
    net_quote_mode_daily = funding_per_day * notional_usd - borrow_daily_quote * notional_usd

    # Asset-borrow mode (funding negative): short spot (borrow base), long perp
    base_qty = notional_usd / price if price > 0 else 0.0
    net_base_mode_daily = (-funding_per_day) * notional_usd - (borrow_daily_base * price * base_qty)

    # Spread one-off fees across holding window
    per_day_fee = one_off_fees / max(1.0, hold_days)

    # Net per-day after fees
    net_quote_mode_daily_after = net_quote_mode_daily - per_day_fee
    net_base_mode_daily_after = net_base_mode_daily - per_day_fee

    # APY approximations
    apy_quote = 365.0 * net_quote_mode_daily_after / notional_usd
    apy_base = 365.0 * net_base_mode_daily_after / notional_usd

    return {
        "funding_per_day": funding_per_day,
        "net_quote_usd_day": net_quote_mode_daily_after,
        "net_base_usd_day": net_base_mode_daily_after,
        "apy_quote": apy_quote,
        "apy_base": apy_base,
        "base_qty": base_qty,
        "one_off_fees": one_off_fees,
    }


def open_stable_borrow_carry(
    spot: ccxt.Exchange,
    futures: ccxt.Exchange,
    symbol_ccxt_spot: str,
    symbol_ccxt_fut: str,
    base_qty: float,
    notional_usd: float,
    dry_run: bool,
    logger: logging.Logger,
):
    if dry_run:
        logger.info(
            f"DRY-RUN open stable-borrow carry: borrow USDT {notional_usd:.2f}, buy {base_qty:.6f} {symbol_ccxt_spot.split('/')[0]}, short perp {base_qty:.6f}"
        )
        return True

    # 1) Borrow USDT
    try:
        _ = spot.sapiPostMarginLoan({"asset": symbol_ccxt_spot.split("/")[1], "amount": str(notional_usd)})
    except Exception as e:
        logger.error(f"Borrow USDT failed: {e}")
        return False

    # 2) Buy spot using quote order quantity (Binance supports quoteOrderQty)
    try:
        order = spot.create_order(symbol_ccxt_spot, "market", "buy", None, None, {"quoteOrderQty": notional_usd})
        logger.info(f"Spot buy executed: {order.get('id')}")
    except Exception as e:
        logger.error(f"Spot buy failed: {e}")
        return False

    # 3) Short perpetual equal base_qty
    try:
        order = futures.create_market_sell_order(symbol_ccxt_fut, base_qty)
        logger.info(f"Perp short executed: {order.get('id')}")
    except Exception as e:
        logger.error(f"Perp short failed: {e}")
        return False

    return True


def open_asset_borrow_carry(
    spot: ccxt.Exchange,
    futures: ccxt.Exchange,
    symbol_ccxt_spot: str,
    symbol_ccxt_fut: str,
    base_qty: float,
    dry_run: bool,
    logger: logging.Logger,
):
    base, quote = symbol_ccxt_spot.split("/")
    if dry_run:
        logger.info(
            f"DRY-RUN open asset-borrow carry: borrow {base_qty:.6f} {base}, sell spot, long perp {base_qty:.6f}"
        )
        return True

    # 1) Borrow base asset
    try:
        _ = spot.sapiPostMarginLoan({"asset": base, "amount": str(base_qty)})
    except Exception as e:
        logger.error(f"Borrow {base} failed: {e}")
        return False

    # 2) Sell spot base_qty
    try:
        order = spot.create_market_sell_order(symbol_ccxt_spot, base_qty)
        logger.info(f"Spot sell executed: {order.get('id')}")
    except Exception as e:
        logger.error(f"Spot sell failed: {e}")
        return False

    # 3) Long perpetual equal base_qty
    try:
        order = futures.create_market_buy_order(symbol_ccxt_fut, base_qty)
        logger.info(f"Perp long executed: {order.get('id')}")
    except Exception as e:
        logger.error(f"Perp long failed: {e}")
        return False

    return True


def close_position(
    pos: PositionManager,
    spot: ccxt.Exchange,
    futures: ccxt.Exchange,
    symbol_ccxt_spot: str,
    symbol_ccxt_fut: str,
    dry_run: bool,
    logger: logging.Logger,
):
    if not pos.is_open():
        logger.info("No open position to close")
        return True

    base, quote = symbol_ccxt_spot.split("/")

    if dry_run:
        logger.info(
            f"DRY-RUN close position ({pos.mode}): reverse perp {pos.base_qty:.6f}, reverse spot {pos.base_qty:.6f}, repay borrowed {base if pos.mode=='asset_borrow' else quote}"
        )
        pos.close()
        return True

    try:
        if pos.mode == "stable_borrow":
            # Close perp short -> buy
            futures.create_market_buy_order(symbol_ccxt_fut, pos.base_qty)
            # Sell spot base
            spot.create_market_sell_order(symbol_ccxt_spot, pos.base_qty)
            # Repay quote (USDT) — repayment amount should cover principal+interest
            # User may need to compute exact outstanding; here we attempt to repay a large amount up to balance
            quote_balance = spot.fetch_balance()["total"].get(quote, 0)
            repay_amt = min(quote_balance, math.inf)  # placeholder; exchange enforces max to outstanding
            spot.sapiPostMarginRepay({"asset": quote, "amount": str(repay_amt)})
        else:
            # Close perp long -> sell
            futures.create_market_sell_order(symbol_ccxt_fut, pos.base_qty)
            # Buy spot base
            spot.create_market_buy_order(symbol_ccxt_spot, pos.base_qty)
            # Repay base
            base_balance = spot.fetch_balance()["total"].get(base, 0)
            repay_amt = min(base_balance, math.inf)
            spot.sapiPostMarginRepay({"asset": base, "amount": str(repay_amt)})

        logger.info("Position closed and repay attempted")
        pos.close()
        return True
    except Exception as e:
        logger.error(f"Failed to close position: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Borrow-Lend vs Funding Carry (Binance, ccxt)")
    parser.add_argument("--base", default=os.getenv("BASE", "BTC"), help="Base asset, e.g., BTC")
    parser.add_argument("--quote", default=os.getenv("QUOTE", "USDT"), help="Quote asset, e.g., USDT")
    parser.add_argument("--notional", type=float, default=float(os.getenv("NOTIONAL_USD", 1000)), help="Target notional in USD")
    parser.add_argument("--min_apy", type=float, default=float(os.getenv("MIN_APY", 0.10)), help="Minimum APY to open position (e.g., 0.15 for 15%)")
    parser.add_argument("--hold_days", type=float, default=float(os.getenv("HOLD_DAYS", 3.0)), help="Assumed holding period to spread one-off fees")
    parser.add_argument("--spot_taker", type=float, default=float(os.getenv("SPOT_TAKER", 0.0010)), help="Spot taker fee (fraction)")
    parser.add_argument("--futures_taker", type=float, default=float(os.getenv("FUTURES_TAKER", 0.0004)), help="Futures taker fee (fraction)")
    parser.add_argument("--poll", type=int, default=int(os.getenv("POLL_SECONDS", 60)), help="Polling interval seconds")
    parser.add_argument("--dry_run", action="store_true", help="Do not place real orders or borrow")
    parser.add_argument("--log", default=os.getenv("LOG_LEVEL", "INFO"), help="Log level")
    parser.add_argument("--sandbox", action="store_true", help="Use sandbox endpoints if supported")
    args = parser.parse_args()

    logger = setup_logger(args.log)

    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    if not api_key or not api_secret:
        raise ValueError("Missing BINANCE_API_KEY/BINANCE_API_SECRET env vars")

    spot, futures = initialize_binance_clients(api_key, api_secret, sandbox=args.sandbox)

    base = args.base.upper()
    quote = args.quote.upper()
    pair_binance = get_binance_pair(base, quote)  # e.g., BTCUSDT for fapi endpoints
    symbol_spot = f"{base}/{quote}"
    # CCXT unified futures symbol on binance USDM is also base/quote
    symbol_futures = f"{base}/{quote}"

    # Sanity check markets
    if symbol_spot not in spot.markets:
        raise ValueError(f"Spot symbol not found: {symbol_spot}")
    if symbol_futures not in futures.markets:
        raise ValueError(f"Futures symbol not found: {symbol_futures}")

    pos = PositionManager()

    logger.info(
        f"Starting funding-carry monitor for {base}/{quote}, notional ${args.notional:.2f}, min APY {args.min_apy:.2%}, dry_run={args.dry_run}"
    )

    while True:
        try:
            price = fetch_price(futures, symbol_futures)
            fr_8h = fetch_funding_rate_usdm_raw(futures, pair_binance)
            borrow_quote_daily = fetch_borrow_rate_daily_binance(spot, quote, logger)
            borrow_base_daily = fetch_borrow_rate_daily_binance(spot, base, logger)

            est = estimate_carry(
                funding_per_8h=fr_8h,
                price=price,
                notional_usd=args.notional,
                borrow_daily_quote=borrow_quote_daily,
                borrow_daily_base=borrow_base_daily,
                spot_taker=args.spot_taker,
                futures_taker=args.futures_taker,
                hold_days=args.hold_days,
            )

            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
            logger.info(
                f"{ts} | px={price:.2f} | fr(8h)={fr_8h:.6f} (~{est['funding_per_day']:.6f}/day) | "
                f"borrow {quote}={borrow_quote_daily:.6f}/day, {base}={borrow_base_daily:.6f}/day | "
                f"APY quote-mode={est['apy_quote']:.2%}, base-mode={est['apy_base']:.2%}"
            )

            should_open_quote = est["apy_quote"] >= args.min_apy
            should_open_base = est["apy_base"] >= args.min_apy

            # Manage position state
            if not pos.is_open():
                if should_open_quote and est["apy_quote"] >= est["apy_base"] and est["net_quote_usd_day"] > 0:
                    base_qty = args.notional / price
                    ok = open_stable_borrow_carry(
                        spot,
                        futures,
                        symbol_spot,
                        symbol_futures,
                        base_qty,
                        args.notional,
                        args.dry_run,
                        logger,
                    )
                    if ok:
                        pos.open("stable_borrow", base_qty, price)
                elif should_open_base and est["net_base_usd_day"] > 0:
                    base_qty = args.notional / price
                    ok = open_asset_borrow_carry(
                        spot,
                        futures,
                        symbol_spot,
                        symbol_futures,
                        base_qty,
                        args.dry_run,
                        logger,
                    )
                    if ok:
                        pos.open("asset_borrow", base_qty, price)
            else:
                # Exit if APY for current mode falls below threshold or flips sign
                if pos.mode == "stable_borrow":
                    if est["apy_quote"] < args.min_apy or est["net_quote_usd_day"] <= 0:
                        close_position(pos, spot, futures, symbol_spot, symbol_futures, args.dry_run, logger)
                elif pos.mode == "asset_borrow":
                    if est["apy_base"] < args.min_apy or est["net_base_usd_day"] <= 0:
                        close_position(pos, spot, futures, symbol_spot, symbol_futures, args.dry_run, logger)

        except KeyboardInterrupt:
            logger.info("Interrupted by user, attempting to close position in dry-run mode")
            try:
                close_position(pos, spot, futures, symbol_spot, symbol_futures, True, logger)
            finally:
                break
        except Exception as e:
            logger.error(f"Loop error: {e}")

        time.sleep(max(10, int(args.poll)))


if __name__ == "__main__":
    main()


