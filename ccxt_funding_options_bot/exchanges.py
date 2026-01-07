from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import ccxt  # type: ignore

from .utils import get_logger, round_to_increment
from .greeks import BlackScholesInputs, call_delta, put_delta


logger = get_logger("exchanges")


@dataclass
class MarketMeta:
    symbol: str
    base: str
    quote: str
    type: str
    expiry: Optional[str] = None
    strike: Optional[float] = None
    option_type: Optional[str] = None  # "call" or "put"
    tick_size: Optional[float] = None
    lot_size: Optional[float] = None


class ExchangeWrapper:
    def __init__(self, exchange_id: str, api_key: Optional[str], secret: Optional[str], password: Optional[str] = None, testnet: bool = False, dry_run: bool = True) -> None:
        self.exchange_id = exchange_id
        self.dry_run = dry_run
        self.client = self._build_ccxt_client(exchange_id, api_key, secret, password, testnet)
        self.markets = self._load_all_markets()

    def _build_ccxt_client(self, exchange_id: str, api_key: Optional[str], secret: Optional[str], password: Optional[str], testnet: bool) -> Any:
        klass = getattr(ccxt, exchange_id)
        kwargs: Dict[str, Any] = {
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {},
        }
        if exchange_id == "deribit":
            kwargs["options"] = {"defaultType": "swap"}
            if testnet:
                kwargs["urls"] = {"api": {"public": "https://test.deribit.com/api/v2", "private": "https://test.deribit.com/api/v2"}}
        if exchange_id == "okx":
            kwargs["password"] = password
            kwargs["options"] = {"defaultType": "swap"}
            if testnet:
                kwargs["urls"] = {"api": {"public": "https://www.okx.com", "private": "https://www.okx.com"}}
        return klass(kwargs)

    def _load_all_markets(self) -> Dict[str, Any]:
        """Load both swap and option markets when available and merge by symbol."""
        merged: Dict[str, Any] = {}
        # Load swap/futures first
        try:
            mkts = self.client.fetch_markets({"type": "swap"})
            for m in mkts:
                merged[m["symbol"]] = m
        except Exception:
            pass
        # Load options next
        try:
            mkts = self.client.fetch_markets({"type": "option"})
            for m in mkts:
                merged[m["symbol"]] = m
        except Exception:
            pass
        # Fallback to default load_markets if still empty
        if not merged:
            try:
                self.client.load_markets()
                merged = dict(self.client.markets)
            except Exception:
                logger.warning("Failed to load markets for %s", self.exchange_id)
        return merged

    # --- Market data helpers ---
    def fetch_ticker_price(self, symbol: str) -> float:
        ticker = self.client.fetch_ticker(symbol)
        last = ticker.get("last") or ticker.get("close")
        if last is None:
            raise RuntimeError(f"No price in ticker for {symbol}")
        return float(last)

    def fetch_funding_rate(self, symbol: str) -> Optional[float]:
        try:
            fr = self.client.fetch_funding_rate(symbol)
            rate = fr.get("fundingRate")
            return float(rate) if rate is not None else None
        except Exception:
            try:
                history = self.client.fetch_funding_rate_history(symbol, limit=1)
                if history:
                    return float(history[0].get("fundingRate"))
            except Exception:
                return None
        return None

    def find_option_markets(self, base: str) -> List[MarketMeta]:
        metas: List[MarketMeta] = []
        for mk in self.markets.values():
            is_option = bool(mk.get("option")) or (mk.get("type") == "option")
            if not is_option:
                continue
            if mk.get("base") != base:
                continue
            metas.append(
                MarketMeta(
                    symbol=mk["symbol"],
                    base=mk.get("base"),
                    quote=mk.get("quote"),
                    type="option",
                    expiry=mk.get("expiry"),
                    strike=mk.get("strike"),
                    option_type=(mk.get("optionType") or mk.get("type")),
                    tick_size=(mk.get("precision") or {}).get("price"),
                    lot_size=(mk.get("precision") or {}).get("amount"),
                )
            )
        return metas

    # --- Trading helpers ---
    def create_order(self, symbol: str, side: str, amount: float, order_type: str = "market", price: Optional[float] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params or {}
        if self.dry_run:
            logger.info(f"DRY RUN create_order {symbol} {side} {amount} {order_type} price={price} params={params}")
            return {"id": "dry-run", "status": "simulated"}
        order = self.client.create_order(symbol, order_type, side, amount, price, params)
        return order

    def ensure_precisions(self, symbol: str, amount: float, price: Optional[float]) -> Tuple[float, Optional[float]]:
        market = self.markets.get(symbol)
        if not market:
            return amount, price
        amount_increment = None
        price_increment = None
        if market.get("precision"):
            amount_increment = market["precision"].get("amount")
            price_increment = market["precision"].get("price")
        adj_amount = round_to_increment(amount, amount_increment or 0.0)
        adj_price = round_to_increment(price, price_increment or 0.0) if price is not None else None
        return adj_amount, adj_price


def select_simple_delta_option(exchange: ExchangeWrapper, base: str, target_delta_sign: int, spot: float, guess_vol: float = 0.8, t_years: float = 30.0 / 365.0) -> Optional[MarketMeta]:
    """Pick a near-the-money option roughly matching a +/-0.25 delta in the needed sign.

    This is a heuristic selection; adjust for your venue and needs.
    """
    options = exchange.find_option_markets(base)
    if not options:
        return None
    desired_is_call = target_delta_sign > 0
    candidates = [m for m in options if (m.option_type or "").lower().startswith("c" if desired_is_call else "p")]
    if not candidates:
        return None
    best: Tuple[Optional[MarketMeta], float] = (None, 1e9)
    for m in candidates:
        if not m.strike:
            continue
        inputs = BlackScholesInputs(spot=spot, strike=float(m.strike), vol=guess_vol, t_years=t_years)
        dlt = call_delta(inputs) if desired_is_call else put_delta(inputs)
        err = abs(abs(dlt) - 0.25)
        if err < best[1]:
            best = (m, err)
    return best[0]


