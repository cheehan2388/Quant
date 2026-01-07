import pandas as pd
import numpy as np


def merge_iv_and_rv(iv_daily: pd.DataFrame, rv_forecast_daily: pd.Series) -> pd.DataFrame:
    """Merge implied variance and RV forecast to compute VRP.

    iv_daily: DataFrame indexed by date with columns:
      - 'iv_ann': annualized implied volatility (e.g., ATM weekly IV)
      - or 'var_ann': annualized implied variance. If 'iv_ann' is present, we square it.
    rv_forecast_daily: Series indexed by date with annualized variance forecast.
    """
    iv = iv_daily.copy()
    if 'var_ann' in iv.columns:
        var_imp = iv['var_ann']
    elif 'iv_ann' in iv.columns:
        var_imp = iv['iv_ann'] ** 2
    else:
        raise ValueError("iv_daily must contain 'iv_ann' or 'var_ann'")

    df = pd.DataFrame({
        'var_imp': var_imp,
        'var_exp': rv_forecast_daily,
    }).dropna()

    df['vrp'] = df['var_imp'] - df['var_exp']
    return df


def position_sizing_vmm(vrp: pd.Series, vol_of_signal: pd.Series, scale: float = 1.0, clip: float = 1.5) -> pd.Series:
    """Volatility-managed sizing: w_t = scale * (vrp_t / vol_of_signal_t), clipped.

    vol_of_signal is a rolling stdev of vrp or returns proxy. clip limits leverage.
    Positive weight implies short variance (sell vol), negative implies long variance.
    """
    w = scale * (vrp / (vol_of_signal.replace(0.0, np.nan)))
    w = w.fillna(0.0)
    w = w.clip(lower=-clip, upper=clip)
    return w


