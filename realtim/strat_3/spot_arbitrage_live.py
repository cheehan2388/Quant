"""
Realtime cross-exchange spot arbitrage scanner using ccxt (async).

Scans multiple centralized exchanges concurrently, fetches best bid/ask
for a set of symbols, estimates net spread after taker fees, and prints
the top-N opportunities.

Notes:
- Public market data only (no API keys required)
- Taker fee assumptions are approximate defaults; adjust if you have VIP tiers
- Does NOT execute transfers or trades; this is a discovery/monitoring tool

Usage examples (PowerShell):
  python Quant/realtim/spot_arbitrage_live.py --symbols BTC/USDT,ETH/USDT,SOL/USDT --top 5 --interval 5
  python Quant/realtim/spot_arbitrage_live.py --exchanges binance,bybit,okx,kucoin,gateio,kraken --top 5
"""

import asyncio
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import ccxt.async_support as ccxt_async


# ----------------------------- Configuration -----------------------------

DEFAULT_EXCHANGES = [
    "binance",
    "bybit",
    "okx",
    "kucoin",
    "gateio",
    "kraken",
]

DEFAULT_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "LINK/USDT",
    "AVAX/USDT",
    "TON/USDT",
]

# Approximate taker fee rates per exchange (as fraction, e.g., 0.001 = 0.1%)
# Adjust based on your account tier. Missing exchanges default to 0.0015 (0.15%).
TAKER_FEE_MAP: Dict[str, float] = {
    "binance": 0.0010,
    "bybit": 0.0010,
    "okx": 0.0010,
    "kucoin": 0.0010,
    "gateio": 0.0020,
    "kraken": 0.0026,
    "mexc": 0.0020,
}

ORDERBOOK_DEPTH = 5
PRINT_TOP = 5
REFRESH_INTERVAL_SECONDS = 5
MIN_NET_SPREAD_PCT = 0.02  # Only show opportunities above this net percentage
SLIPPAGE_BUFFER_PCT = 0.01  # Extra safety buffer in percentage points


# ----------------------------- Data Models -----------------------------

@dataclass
class Quote:
    price: float
    amount: float


@dataclass
class VenueQuote:
    exchange_id: str
    symbol: str
    best_bid: Optional[Quote]
    best_ask: Optional[Quote]
    ts: float


@dataclass
class Opportunity:
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    size: float
    gross_spread_pct: float
    net_spread_pct: float


# ----------------------------- Helpers -----------------------------

def _now_ts() -> float:
    return time.time()


def get_taker_fee(exchange_id: str) -> float:
    return TAKER_FEE_MAP.get(exchange_id, 0.0015)


async def safe_load_markets(exchange) -> bool:
    try:
        await exchange.load_markets()
        return True
    except Exception:
        return False


async def fetch_best_quotes(exchange_id: str, exchange, symbol: str) -> Optional[VenueQuote]:
    try:
        orderbook = await exchange.fetch_order_book(symbol, ORDERBOOK_DEPTH)
        bids = orderbook.get("bids") or []
        asks = orderbook.get("asks") or []

        best_bid = Quote(bids[0][0], bids[0][1]) if bids else None
        best_ask = Quote(asks[0][0], asks[0][1]) if asks else None

        return VenueQuote(
            exchange_id=exchange_id,
            symbol=symbol,
            best_bid=best_bid,
            best_ask=best_ask,
            ts=_now_ts(),
        )
    except Exception:
        return None


def compute_opportunity(symbol: str, buyer: VenueQuote, seller: VenueQuote) -> Optional[Opportunity]:
    if not buyer or not seller or not buyer.best_ask or not seller.best_bid:
        return None

    if buyer.best_ask.price <= 0 or seller.best_bid.price <= 0:
        return None

    raw_spread = (seller.best_bid.price - buyer.best_ask.price) / buyer.best_ask.price
    gross_spread_pct = raw_spread * 100.0

    # Taker fees on both legs
    fee_buy = get_taker_fee(buyer.exchange_id)
    fee_sell = get_taker_fee(seller.exchange_id)
    total_fee_pct = (fee_buy + fee_sell) * 100.0

    # Safety buffer for slippage and volatility
    net_spread_pct = gross_spread_pct - total_fee_pct - SLIPPAGE_BUFFER_PCT

    if net_spread_pct <= 0:
        return None

    trade_size = min(buyer.best_ask.amount, seller.best_bid.amount)
    if trade_size <= 0:
        return None

    return Opportunity(
        symbol=symbol,
        buy_exchange=buyer.exchange_id,
        sell_exchange=seller.exchange_id,
        buy_price=buyer.best_ask.price,
        sell_price=seller.best_bid.price,
        size=trade_size,
        gross_spread_pct=gross_spread_pct,
        net_spread_pct=net_spread_pct,
    )


async def initialize_exchanges(exchange_ids: List[str]) -> Dict[str, ccxt_async.Exchange]:
    instances: Dict[str, ccxt_async.Exchange] = {}
    for ex_id in exchange_ids:
        if not hasattr(ccxt_async, ex_id):
            continue
        ex_class = getattr(ccxt_async, ex_id)
        instances[ex_id] = ex_class({
            "enableRateLimit": True,
            "timeout": 20000,
            # Public only; add apiKey/secret if you want to use private endpoints later
        })

    # Load markets concurrently
    tasks = [safe_load_markets(ex) for ex in instances.values()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Drop exchanges that failed to load markets
    for i, ex_id in enumerate(list(instances.keys())):
        ok = results[i] is True
        if not ok:
            try:
                await instances[ex_id].close()
            except Exception:
                pass
            del instances[ex_id]

    return instances


def filter_supported_symbols(exchanges: Dict[str, ccxt_async.Exchange], symbols: List[str]) -> Dict[str, List[str]]:
    supported: Dict[str, List[str]] = {}
    for ex_id, ex in exchanges.items():
        ex_symbols = []
        try:
            markets = ex.markets or {}
            for s in symbols:
                if s in markets:
                    ex_symbols.append(s)
        except Exception:
            pass
        if ex_symbols:
            supported[ex_id] = ex_symbols
    return supported


async def scan_once(exchanges: Dict[str, ccxt_async.Exchange], symbols: List[str], top_n: int) -> List[Opportunity]:
    # Build tasks: one fetch per (exchange, symbol)
    tasks: List[asyncio.Task] = []
    key_list: List[Tuple[str, str]] = []
    for ex_id, ex in exchanges.items():
        for s in symbols:
            tasks.append(asyncio.create_task(fetch_best_quotes(ex_id, ex, s)))
            key_list.append((ex_id, s))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Aggregate quotes per symbol
    symbol_to_quotes: Dict[str, List[VenueQuote]] = {}
    for res in results:
        if isinstance(res, VenueQuote):
            symbol_to_quotes.setdefault(res.symbol, []).append(res)

    # Compute best opportunities per symbol
    all_opps: List[Opportunity] = []
    for s, quotes in symbol_to_quotes.items():
        if len(quotes) < 2:
            continue

        # For each pair of venues, compute opportunity (buy on A, sell on B)
        for i in range(len(quotes)):
            for j in range(len(quotes)):
                if i == j:
                    continue
                opp = compute_opportunity(s, buyer=quotes[i], seller=quotes[j])
                if opp and opp.net_spread_pct >= MIN_NET_SPREAD_PCT:
                    all_opps.append(opp)

    # Rank by net spread desc
    all_opps.sort(key=lambda o: o.net_spread_pct, reverse=True)
    return all_opps[:top_n]


def format_opportunity_row(idx: int, o: Opportunity) -> str:
    return (
        f"#{idx:02d} {o.symbol:<10} Buy {o.buy_exchange:<8} @ {o.buy_price:>12.6f}  "
        f"Sell {o.sell_exchange:<8} @ {o.sell_price:>12.6f}  Size {o.size:>10.6f}  "
        f"Gross {o.gross_spread_pct:>6.3f}%  Net {o.net_spread_pct:>6.3f}%"
    )


async def run_scanner(exchange_ids: List[str], symbols: List[str], top_n: int, interval_seconds: int):
    # Windows event loop compatibility (esp. for older Python versions)
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass

    exchanges = await initialize_exchanges(exchange_ids)
    if not exchanges:
        print("No exchanges initialized. Exiting.")
        return

    supported = filter_supported_symbols(exchanges, symbols)
    universe: List[str] = sorted({s for lst in supported.values() for s in lst})
    if not universe:
        print("No requested symbols are supported on the initialized exchanges. Exiting.")
        await asyncio.gather(*[ex.close() for ex in exchanges.values() if hasattr(ex, "close")], return_exceptions=True)
        return

    ex_list = ", ".join(sorted(exchanges.keys()))
    sym_list = ", ".join(universe)
    print(f"Initialized exchanges: {ex_list}")
    print(f"Scanning symbols: {sym_list}")
    print(f"Showing top {top_n} opportunities every {interval_seconds}s (min net spread {MIN_NET_SPREAD_PCT:.3f}%)")

    try:
        while True:
            start = _now_ts()
            opps = await scan_once(exchanges, universe, top_n)

            # Clear-like separation
            print("\n" + "=" * 100)
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "- Top Opportunities")
            if not opps:
                print("No opportunities above threshold.")
            else:
                for idx, o in enumerate(opps, start=1):
                    print(format_opportunity_row(idx, o))

            elapsed = max(0.0, interval_seconds - (time.time() - start))
            await asyncio.sleep(elapsed)
    except KeyboardInterrupt:
        print("Interrupted. Shutting down...")
    finally:
        await asyncio.gather(*[ex.close() for ex in exchanges.values() if hasattr(ex, "close")], return_exceptions=True)


def parse_cli_args() -> Tuple[List[str], List[str], int, int]:
    import argparse

    parser = argparse.ArgumentParser(description="Realtime cross-exchange spot arbitrage scanner")
    parser.add_argument("--exchanges", type=str, default=",".join(DEFAULT_EXCHANGES), help="Comma-separated list of exchanges")
    parser.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS), help="Comma-separated list of symbols")
    parser.add_argument("--top", type=int, default=PRINT_TOP, help="Top N opportunities to show")
    parser.add_argument("--interval", type=int, default=REFRESH_INTERVAL_SECONDS, help="Refresh interval in seconds")

    args = parser.parse_args()
    exchange_ids = [x.strip() for x in args.exchanges.split(",") if x.strip()]
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    return exchange_ids, symbols, int(args.top), int(args.interval)


def main():
    exchange_ids, symbols, top_n, interval_seconds = parse_cli_args()
    asyncio.run(run_scanner(exchange_ids, symbols, top_n, interval_seconds))


if __name__ == "__main__":
    main()


