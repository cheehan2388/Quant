from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

from .config import CoinConfig, StrategyConfig
from .exchanges import ExchangeWrapper, select_simple_delta_option
from .greeks import BlackScholesInputs, call_delta, put_delta, gamma as bs_gamma
from .risk import check_gamma_bound
from .utils import get_logger


logger = get_logger("strategy")


@dataclass
class HedgePlan:
    option_symbol: str
    side: str  # buy/sell
    contracts: float
    expected_option_delta_per_contract: float


def decide_perp_direction(funding_rate: Optional[float], min_bps: float) -> Optional[str]:
    if funding_rate is None:
        return None
    threshold = min_bps / 10000.0
    if funding_rate >= threshold:
        return "short"
    if funding_rate <= -threshold:
        return "long"
    return None


def plan_delta_hedge(exchange: ExchangeWrapper, coin: CoinConfig, spot: float, perp_amount_base: float) -> Optional[HedgePlan]:
    # Target: reduce absolute delta to coin.target_delta_abs fraction of perp delta
    desired_residual = abs(perp_amount_base) * coin.target_delta_abs
    to_hedge = max(0.0, abs(perp_amount_base) - desired_residual)
    if to_hedge <= 0.0:
        return None

    need_positive_delta = perp_amount_base < 0  # short perp -> need +delta
    target_sign = +1 if need_positive_delta else -1

    opt = select_simple_delta_option(exchange, coin.option_base, target_delta_sign=target_sign, spot=spot)
    if not opt:
        logger.warning("No suitable option market found for %s", coin.option_base)
        return None

    # Approximate Greeks to size contracts
    inputs = BlackScholesInputs(spot=spot, strike=float(opt.strike or spot), vol=0.8, t_years=30.0 / 365.0)
    dlt = call_delta(inputs) if target_sign > 0 else abs(put_delta(inputs))
    dlt = max(1e-3, min(0.9, dlt))  # clamp
    contracts = to_hedge / dlt
    side = "buy"  # buy calls for +delta or buy puts for -delta (both purchases)
    logger.info("Planned hedge: %s %s x %.4f (delta/ct=%.3f)", side, opt.symbol, contracts, dlt)
    return HedgePlan(option_symbol=opt.symbol, side=side, contracts=contracts, expected_option_delta_per_contract=dlt)


def run_cycle(exchange: ExchangeWrapper, coin: CoinConfig, strat: StrategyConfig) -> None:
    perp_symbol = coin.perp_symbol
    spot = exchange.fetch_ticker_price(perp_symbol)
    funding = exchange.fetch_funding_rate(perp_symbol)
    logger.info("%s price=%.4f funding=%.6f", perp_symbol, spot, funding or float("nan"))

    direction = decide_perp_direction(funding, coin.min_funding_bps_to_trade)
    if not direction:
        logger.info("No trade: funding within threshold")
        return

    usd_notional = coin.usd_notional_per_trade
    perp_amount = usd_notional / max(1e-8, spot)
    side = "sell" if direction == "short" else "buy"
    amt, _ = exchange.ensure_precisions(perp_symbol, perp_amount, None)
    logger.info("Perp %s %s amount=%.6f (~$%.2f)", side, perp_symbol, amt, amt * spot)
    exchange.create_order(perp_symbol, side, amt, order_type="market")

    # Hedge using options
    hedge = plan_delta_hedge(exchange, coin, spot, perp_amount_base=(-amt if side == "sell" else amt))
    if hedge:
        # Risk check: gamma guardrail
        inputs = BlackScholesInputs(spot=spot, strike=spot, vol=0.8, t_years=30.0 / 365.0)
        gam = bs_gamma(inputs)
        gcheck = check_gamma_bound(gam * hedge.contracts, spot)
        if not gcheck.ok:
            logger.warning("Skip options hedge due to gamma risk: %s", gcheck.message)
        else:
            c_amt, _ = exchange.ensure_precisions(hedge.option_symbol, hedge.contracts, None)
            exchange.create_order(hedge.option_symbol, hedge.side, c_amt, order_type="market")


def run_loop(exchange: ExchangeWrapper, coin: CoinConfig, strat: StrategyConfig) -> None:
    while True:
        try:
            run_cycle(exchange, coin, strat)
        except Exception as e:
            logger.exception("Cycle error: %s", e)
        time.sleep(max(5, strat.poll_seconds))


