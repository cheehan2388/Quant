import numpy as np
import pandas as pd
from typing import Tuple


def variance_swap_proxy_pnl(var_imp: pd.Series, realized_var: pd.Series) -> pd.Series:
    """Variance swap proxy daily PnL for unit notional on variance.

    PnL_t ≈ (K_var - RV_realized_future) / N, but in practice we need horizon alignment.
    For a simple daily mark-to-model proxy, we use changes in implied variance and realized variance accrual.
    Here we approximate VRP carry by: carry_t = var_imp_t - realized_var_{t+1}.
    This is a stylized proxy suitable for signal evaluation; a production sim should model term structure and settlement.
    """
    # Align realized var to next day to avoid look-ahead
    rv_next = realized_var.shift(-1)
    pnl = var_imp - rv_next
    pnl = pnl.dropna()
    return pnl


def apply_weights(pnl: pd.Series, weights: pd.Series, tc_bps: float = 0.0) -> pd.Series:
    """Apply strategy weights to unit-notional PnL and subtract linear transaction costs.

    tc_bps: round-trip basis points cost per weight change. We approximate daily cost as tc_bps * |Δw|.
    """
    w = weights.reindex(pnl.index).fillna(0.0)
    strat = w * pnl
    dw = w.diff().abs().fillna(0.0)
    tc = (tc_bps / 1e4) * dw
    return strat - tc


def performance_stats(returns: pd.Series, periods_per_year: int = 252) -> dict:
    mu = returns.mean() * periods_per_year
    sigma = returns.std(ddof=0) * np.sqrt(periods_per_year)
    sharpe = mu / sigma if sigma > 0 else 0.0
    dd = (returns.cumsum() - returns.cumsum().cummax())
    max_dd = dd.min()
    return {
        'ann_return': float(mu),
        'ann_vol': float(sigma),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
    }


def join_inputs(iv_df: pd.DataFrame, rv_forecast: pd.Series, rv_realized: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(index=iv_df.index.union(rv_forecast.index).union(rv_realized.index))
    if 'var_ann' in iv_df.columns:
        df['var_imp'] = iv_df['var_ann']
    else:
        df['var_imp'] = iv_df['iv_ann'] ** 2
    df['var_exp'] = rv_forecast
    df['var_real'] = rv_realized
    return df.dropna()


