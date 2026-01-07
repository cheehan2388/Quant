from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import config, utils


@dataclass
class ExecutionConfig:
    notional_per_vega_unit: float = 1.0  # maps vega_target units to PnL scale
    transaction_cost_bps: float = 1.0  # applied on changes in vega exposure
    delta_hedge_cost_bps: float = 0.5  # cost per hedge turnover unit (simplified)


def simulate_variance_swap_like_pnl(
    vega_target: pd.Series,
    realized_daily_variance: pd.Series,
    implied_annualized_vol: pd.Series,
    cfg: ExecutionConfig = ExecutionConfig(),
) -> pd.DataFrame:
    """
    Simplified PnL proxy:
    - Exposure proportional to vega_target (dimensionless), scaled by notional_per_vega_unit
    - PnL approximated by (IV^2 - RV_annualized) * exposure per day
    - Transaction cost on changes in exposure
    This is not a full option PnL model but aligns with variance swap intuition.
    """
    vt, rv_daily = utils.safe_align(vega_target, realized_daily_variance)
    vt, iv = utils.safe_align(vt, implied_annualized_vol)

    rv_ann = utils.annualize_daily_variance(rv_daily)
    iv_var = iv ** 2

    exposure = vt * cfg.notional_per_vega_unit
    pnl_core = (iv_var - rv_ann) * exposure / config.TRADING_DAYS_PER_YEAR

    exposure_change = exposure.diff().abs().fillna(0.0)
    tc = - (cfg.transaction_cost_bps * 1e-4) * exposure_change

    # crude proxy for delta-hedge cost proportional to absolute exposure times daily vol
    daily_vol = np.sqrt(rv_daily.clip(lower=0.0))
    hedge_tc = - (cfg.delta_hedge_cost_bps * 1e-4) * (exposure.abs() * daily_vol)

    pnl = (pnl_core + tc + hedge_tc).rename("pnl")
    equity = pnl.cumsum().rename("equity")
    out = pd.concat([pnl, equity, exposure.rename("exposure")], axis=1)
    return out


