from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from . import config


def compute_log_returns_from_prices(prices: pd.Series) -> pd.Series:
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices index must be a DatetimeIndex")
    returns = np.log(prices).diff().dropna()
    returns.name = "log_ret"
    return returns


def ensure_datetime_index(series: pd.Series) -> pd.Series:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series must have a DatetimeIndex")
    return series.sort_index()


def annualize_daily_variance(daily_variance: float | np.ndarray | pd.Series) -> float | np.ndarray | pd.Series:
    return daily_variance * config.TRADING_DAYS_PER_YEAR


def annualized_vol_from_daily_variance(daily_variance: float | np.ndarray | pd.Series) -> float | np.ndarray | pd.Series:
    return np.sqrt(annualize_daily_variance(daily_variance))


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def safe_align(lhs: pd.Series, rhs: pd.Series) -> tuple[pd.Series, pd.Series]:
    lhs = ensure_datetime_index(lhs)
    rhs = ensure_datetime_index(rhs)
    joined = pd.concat([lhs, rhs], axis=1, join="inner").dropna()
    return joined.iloc[:, 0], joined.iloc[:, 1]


def as_daily(series: pd.Series, how: str = "sum") -> pd.Series:
    series = ensure_datetime_index(series)
    if how == "sum":
        return series.resample("1D").sum().dropna()
    if how == "mean":
        return series.resample("1D").mean().dropna()
    if how == "last":
        return series.resample("1D").last().dropna()
    raise ValueError("Unsupported aggregation method: %s" % how)


def realized_variance_over_window(daily_variance: pd.Series, start_inclusive: pd.Timestamp, end_inclusive: pd.Timestamp) -> float:
    window = daily_variance.loc[start_inclusive:end_inclusive]
    if window.empty:
        raise ValueError("No daily variance in the requested window")
    mean_daily_var = window.mean()
    return annualize_daily_variance(mean_daily_var)


