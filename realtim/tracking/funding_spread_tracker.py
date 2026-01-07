import os
import time
import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import ccxt.async_support as ccxt


# --------------------------- Config ---------------------------
DEFAULT_EXCHANGES = [
    "binance",
    "bybit",
    "okx",
    "bitget",
    "kucoinfutures",
    "bingx",
    "gate",
    "huobi",
    "mexc",
]

# Exchanges that require password/extra opts
EXTRA_PASSWORD = {
    "okx": os.getenv("OKX_API_PASSWORD", ""),
}


def setup_logger(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return logging.getLogger("funding_spread_tracker")


@dataclass
class MarketFunding:
    exchange: str
    symbol: str
    base: str
    quote: str
    funding_rate_8h: Optional[float]
    funding_rate_daily: Optional[float]


def normalize_symbol_for_swap(exchange_id: str, market: Dict) -> Optional[str]:
    # Use ccxt unified symbol when type is swap/future
    if not market.get("active", True):
        return None
    if market.get("type") not in ("swap", "future") and not market.get("swap", False):
        return None
    symbol = market.get("symbol")
    if not symbol:
        return None
    return symbol


def normalize_quote(quote: str) -> str:
    # Remove any contract suffix like USDT:USDT, keep primary currency
    q = (quote or "").split(":")[0].upper()
    return q


def is_linear_stable_usd_market(market: Dict) -> bool:
    # Select USDT/USDC linear swaps only for apples-to-apples comparison
    if not (market.get("swap", False) or market.get("type") == "swap"):
        return False
    if market.get("inverse"):
        return False
    quote = normalize_quote(market.get("quote") or market.get("settle") or "")
    if quote not in {"USDT", "USDC"}:
        return False
    # Prefer explicit linear flag if provided
    if market.get("linear") is False:
        return False
    return True


def normalize_base(base: str) -> str:
    b = (base or "").upper()
    mapping = {
        "XBT": "BTC",
        "BCHABC": "BCH",
        "BCHSV": "BSV",
    }
    return mapping.get(b, b)


async def build_exchange(exchange_id: str, enable_private: bool = False):
    klass = getattr(ccxt, exchange_id)
    params = {"enableRateLimit": True, "options": {"defaultType": "swap"}}
    if enable_private:
        ak = os.getenv(f"{exchange_id.upper()}_API_KEY", "")
        sk = os.getenv(f"{exchange_id.upper()}_API_SECRET", "")
        if ak and sk:
            params.update({"apiKey": ak, "secret": sk})
        if exchange_id in EXTRA_PASSWORD and EXTRA_PASSWORD[exchange_id]:
            params["password"] = EXTRA_PASSWORD[exchange_id]
    ex = klass(params)
    await ex.load_markets()
    return ex


async def close_exchange_safe(ex) -> None:
    try:
        await ex.close()
    except Exception:
        pass


async def pick_top_perps_by_tickers(ex, top_n: int, linear_stable_only: bool = True, min_24h_volume_usd: float = 0.0) -> List[str]:
    try:
        tickers = await ex.fetch_tickers()
    except Exception:
        return []

    markets = getattr(ex, "markets", {}) or {}
    entries: List[Tuple[str, float]] = []
    for sym, t in tickers.items():
        m = markets.get(sym)
        if not m:
            continue
        if linear_stable_only and not is_linear_stable_usd_market(m):
            continue
        vol_usd = 0.0
        # Prefer quoteVolume if available
        for k in ["quoteVolume", "volumeQuote", "baseVolume"]:
            try:
                v = t.get(k)
                if v is not None:
                    v = float(v)
                    # baseVolume needs conversion; attempt with last price
                    if k == "baseVolume":
                        px = t.get("last") or t.get("close")
                        if px is not None:
                            v = float(v) * float(px)
                    vol_usd = max(vol_usd, float(v))
            except Exception:
                pass
        # Info fallbacks
        info = t.get("info", {}) or {}
        for k in ["turnoverUsd24h", "quote_volume_24h", "volumeUsd24h", "turnover24h"]:
            try:
                v = info.get(k)
                if v is not None:
                    vol_usd = max(vol_usd, float(v))
            except Exception:
                pass
        if vol_usd >= min_24h_volume_usd:
            entries.append((sym, vol_usd))

    if not entries:
        return []
    entries.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, _ in entries[:top_n]]


def pick_top_perps_from_markets(ex, top_n: int, linear_stable_only: bool = True) -> List[str]:
    # Heuristic: choose top by quote volume or list order
    markets = getattr(ex, "markets", {}) or {}
    entries: List[Tuple[str, float]] = []
    for m in markets.values():
        if linear_stable_only and not is_linear_stable_usd_market(m):
            continue
        sym = normalize_symbol_for_swap(ex.id, m)
        if not sym:
            continue
        quote_vol = 0.0
        info = m.get("info", {}) or {}
        for k in ["quoteVolume", "turnoverUsd24h", "quote_volume_24h", "volumeUsd24h"]:
            try:
                v = float(info.get(k)) if info.get(k) is not None else None
                if isinstance(v, (float, int)):
                    quote_vol = max(quote_vol, float(v))
            except Exception:
                pass
        # Fallback by precision tick size as a weak proxy to filter illiquid nonsense
        if quote_vol == 0.0:
            prec = m.get("precision", {}) or {}
            if prec.get("price") is not None:
                quote_vol = 1.0
        entries.append((sym, quote_vol))
    entries.sort(key=lambda x: x[1], reverse=True)
    top = [sym for sym, _ in entries[:top_n]]
    return top


async def fetch_funding_rate_daily(ex, symbol: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        fr = await ex.fetch_funding_rate(symbol)
        rate = None
        for k in ("nextFundingRate", "fundingRate"):
            if fr.get(k) is not None:
                try:
                    rate = float(fr[k])
                    break
                except Exception:
                    pass
        if rate is None and isinstance(fr.get("info"), dict):
            info = fr["info"]
            for k in ("fundingRate", "predictedFundingRate", "nextFundingRate", "lastFundingRate"):
                if info.get(k) is not None:
                    try:
                        rate = float(info[k])
                        break
                    except Exception:
                        pass
        if rate is None:
            return None, None
        interval_hours = 8.0
        if isinstance(fr.get("fundingIntervalHours"), (int, float)) and fr["fundingIntervalHours"]:
            interval_hours = float(fr["fundingIntervalHours"])
        return rate, rate * (24.0 / interval_hours)
    except Exception:
        return None, None


async def gather_exchange_funding(
    ex,
    exchange_id: str,
    linear_stable_only: bool = True,
    top_n: int = 20,
    min_24h_volume_usd: float = 0.0,
    use_tickers_for_ranking: bool = True,
) -> List[MarketFunding]:
    symbols: List[str] = []
    if use_tickers_for_ranking:
        symbols = await pick_top_perps_by_tickers(
            ex, top_n=top_n, linear_stable_only=linear_stable_only, min_24h_volume_usd=min_24h_volume_usd
        )
    if not symbols:
        symbols = pick_top_perps_from_markets(ex, top_n=top_n, linear_stable_only=linear_stable_only)
    results: List[MarketFunding] = []
    tasks = [fetch_funding_rate_daily(ex, s) for s in symbols]
    fetched = await asyncio.gather(*tasks, return_exceptions=True)
    for s, res in zip(symbols, fetched):
        if isinstance(res, Exception):
            fr8, frd = None, None
        else:
            fr8, frd = res
        try:
            parts = s.split("/")
            base = normalize_base(parts[0])
            quote = normalize_quote(parts[1]) if len(parts) > 1 else "USD"
        except Exception:
            base, quote = s, "USD"
        results.append(MarketFunding(exchange=exchange_id, symbol=s, base=base, quote=quote, funding_rate_8h=fr8, funding_rate_daily=frd))
    return results


def best_cross_exchange_spreads(data: List[MarketFunding], min_abs_daily_spread: float) -> List[Tuple[MarketFunding, MarketFunding, float]]:
    # key by base/quote standardized to compare same contract notionals
    by_pair: Dict[Tuple[str, str], List[MarketFunding]] = {}
    for d in data:
        if d.funding_rate_daily is None:
            continue
        key = (normalize_base(d.base), normalize_quote(d.quote))
        by_pair.setdefault(key, []).append(d)

    signals: List[Tuple[MarketFunding, MarketFunding, float]] = []
    for pair, items in by_pair.items():
        items = [x for x in items if x.funding_rate_daily is not None]
        if len(items) < 2:
            continue
        items.sort(key=lambda x: x.funding_rate_daily)  # ascending
        low = items[0]
        high = items[-1]
        spread = float(high.funding_rate_daily) - float(low.funding_rate_daily)
        if abs(spread) >= min_abs_daily_spread:
            # Return as (receiver, payer, spread) where receiver is short on higher funding (receives),
            # and payer is long on lower funding (pays or receives less)
            signals.append((high, low, spread))
    # Sort by absolute spread desc
    signals.sort(key=lambda x: abs(x[2]), reverse=True)
    return signals


async def place_hedged_trades(
    logger: logging.Logger,
    exchanges: Dict[str, any],
    receiver: MarketFunding,
    payer: MarketFunding,
    notional_usd: float,
    dry_run: bool = True,
) -> None:
    # Simplified: assume amount = notional / price; fetch price from each venue ticker
    async def fetch_px(ex, sym: str) -> Optional[float]:
        try:
            t = await ex.fetch_ticker(sym)
            p = t.get("last") or t.get("close") or t.get("info", {}).get("index_price")
            return float(p) if p is not None else None
        except Exception:
            return None

    ex_r = exchanges.get(receiver.exchange)
    ex_p = exchanges.get(payer.exchange)
    if not ex_r or not ex_p:
        logger.warning("Missing exchange instances for trade")
        return

    pr, pp = await asyncio.gather(fetch_px(ex_r, receiver.symbol), fetch_px(ex_p, payer.symbol))
    if not pr or not pp or pr <= 0 or pp <= 0:
        logger.warning("Could not fetch prices for hedged trade")
        return
    amt_r = notional_usd / pr
    amt_p = notional_usd / pp

    if dry_run:
        logger.info(
            "[DRY-RUN] Hedge trade: SHORT %s %s x %.6f (receiver) | LONG %s %s x %.6f (payer)",
            receiver.exchange,
            receiver.symbol,
            amt_r,
            payer.exchange,
            payer.symbol,
            amt_p,
        )
        return

    try:
        sell_task = ex_r.create_order(receiver.symbol, "market", "sell", amt_r)
        buy_task = ex_p.create_order(payer.symbol, "market", "buy", amt_p)
        await asyncio.gather(sell_task, buy_task)
        logger.info("Executed hedged trades: short %s %s, long %s %s", receiver.exchange, receiver.symbol, payer.exchange, payer.symbol)
    except Exception as e:
        logger.error("Hedged trade failed: %s", e)


async def run_tracker(
    exchanges: Optional[List[str]] = None,
    min_abs_daily_spread: float = 0.03,  # 3% per day spread
    poll_seconds: int = 60,
    notional_usd: float = 1000.0,
    execute: bool = False,
    dry_run: bool = True,
    log_level: str = "INFO",
    cooldown_seconds: int = 900,
    linear_stable_only: bool = True,
    top_n_per_exchange: int = 20,
    min_24h_volume_usd: float = 0.0,
    use_tickers_for_ranking: bool = True,
):
    logger = setup_logger(log_level)
    ex_ids = exchanges or DEFAULT_EXCHANGES

    logger.info("Starting funding spread tracker on: %s", ", ".join(ex_ids))

    # Build all exchanges concurrently
    build_tasks = [build_exchange(eid, enable_private=execute and not dry_run) for eid in ex_ids]
    instances = await asyncio.gather(*build_tasks, return_exceptions=True)
    id_to_ex = {}
    for eid, ex in zip(ex_ids, instances):
        if isinstance(ex, Exception):
            logger.warning("Failed to init %s: %s", eid, ex)
            continue
        id_to_ex[eid] = ex

    last_trade_ts: Dict[Tuple[str, str, str, str, str], float] = {}

    try:
        while True:
            try:
                gather_tasks = [
                    gather_exchange_funding(
                        ex,
                        eid,
                        linear_stable_only=linear_stable_only,
                        top_n=top_n_per_exchange,
                        min_24h_volume_usd=min_24h_volume_usd,
                        use_tickers_for_ranking=use_tickers_for_ranking,
                    )
                    for eid, ex in id_to_ex.items()
                ]
                per_ex = await asyncio.gather(*gather_tasks, return_exceptions=True)
                all_data: List[MarketFunding] = []
                for res in per_ex:
                    if isinstance(res, Exception):
                        continue
                    all_data.extend(res)

                # Compute spreads
                signals = best_cross_exchange_spreads(all_data, min_abs_daily_spread)
                if not signals:
                    logger.info("No spreads >= %.4f daily found across venues", min_abs_daily_spread)
                else:
                    top = signals[:5]
                    for receiver, payer, spread in top:
                        logger.info(
                            "Spread %.4f daily | %s %s (%.4f) vs %s %s (%.4f)",
                            spread,
                            receiver.exchange,
                            receiver.symbol,
                            receiver.funding_rate_daily or float("nan"),
                            payer.exchange,
                            payer.symbol,
                            payer.funding_rate_daily or float("nan"),
                        )
                        if execute:
                            # Per-pair cooldown key (base+quote normalized + sorted legs)
                            qn = normalize_quote(receiver.quote)
                            pair_key = (
                                receiver.base.upper(),
                                qn,
                                min((receiver.exchange, receiver.symbol), (payer.exchange, payer.symbol)),
                                max((receiver.exchange, receiver.symbol), (payer.exchange, payer.symbol)),
                                "hedge",
                            )
                            now = time.time()
                            if now - last_trade_ts.get(pair_key, 0.0) < max(60, cooldown_seconds):
                                logger.info("Cooldown active for pair %s/%s between %s and %s", receiver.base.upper(), qn, receiver.exchange, payer.exchange)
                                continue
                            await place_hedged_trades(
                                logger=logger,
                                exchanges=id_to_ex,
                                receiver=receiver,
                                payer=payer,
                                notional_usd=notional_usd,
                                dry_run=dry_run,
                            )
                            last_trade_ts[pair_key] = now
            except Exception as e:
                logger.error("Tracker loop error: %s", e)

            await asyncio.sleep(max(5, int(poll_seconds)))
    finally:
        await asyncio.gather(*[close_exchange_safe(ex) for ex in id_to_ex.values()])


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Realtime funding spread tracker across exchanges (ccxt)")
    parser.add_argument("--exchanges", type=str, default=",".join(DEFAULT_EXCHANGES), help="Comma-separated ccxt ids")
    parser.add_argument("--threshold_daily", type=float, default=float(os.getenv("SPREAD_DAILY", 0.03)), help="Min daily spread to signal/execute (e.g., 0.03 = 3%)")
    parser.add_argument("--poll", type=int, default=int(os.getenv("POLL_SECONDS", 60)), help="Polling interval seconds")
    parser.add_argument("--notional", type=float, default=float(os.getenv("NOTIONAL_USD", 1000)), help="Hedge notional per leg in USD")
    parser.add_argument("--execute", action="store_true", help="Place hedged trades when threshold met")
    parser.add_argument("--dry_run", action="store_true", help="Simulate orders even when execute is set")
    parser.add_argument("--cooldown", type=int, default=int(os.getenv("COOLDOWN_SECONDS", 900)), help="Cooldown seconds per pair to avoid refiring")
    parser.add_argument("--linear_stable_only", action="store_true", help="Restrict to USDT/USDC linear swaps only (recommended)")
    parser.add_argument("--no-linear_stable_only", dest="linear_stable_only", action="store_false")
    parser.set_defaults(linear_stable_only=True)
    parser.add_argument("--top_n", type=int, default=int(os.getenv("TOP_N_PER_EXCHANGE", 20)), help="Top N perps per exchange to track")
    parser.add_argument("--min_vol_usd_24h", type=float, default=float(os.getenv("MIN_24H_VOL_USD", 0.0)), help="Minimum 24h USD volume to include (ranking by tickers)")
    parser.add_argument("--rank_by_tickers", action="store_true", help="Use fetch_tickers for ranking top markets (more precise, heavier)")
    parser.add_argument("--no-rank_by_tickers", dest="rank_by_tickers", action="store_false")
    parser.set_defaults(rank_by_tickers=True)
    parser.add_argument("--log", type=str, default=os.getenv("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    ex_list = [x.strip() for x in args.exchanges.split(",") if x.strip()]
    asyncio.run(
        run_tracker(
            exchanges=ex_list,
            min_abs_daily_spread=args.threshold_daily,
            poll_seconds=args.poll,
            notional_usd=args.notional,
            execute=args.execute,
            dry_run=args.dry_run or (not args.execute),
            log_level=args.log,
            cooldown_seconds=args.cooldown,
            linear_stable_only=args.linear_stable_only,
            top_n_per_exchange=args.top_n,
            min_24h_volume_usd=args.min_vol_usd_24h,
            use_tickers_for_ranking=args.rank_by_tickers,
        )
    )


if __name__ == "__main__":
    main()


