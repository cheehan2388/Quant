from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from . import config, utils


@dataclass
class HarVarianceModel:
    intercept: float
    beta_daily: float
    beta_weekly: float
    beta_monthly: float

    def predict_next_daily_variance(self, features: Tuple[float, float, float]) -> float:
        x_d, x_w, x_m = features
        return (
            self.intercept
            + self.beta_daily * x_d
            + self.beta_weekly * x_w
            + self.beta_monthly * x_m
        )


def compute_daily_variance_from_intraday(log_returns: pd.Series) -> pd.Series:
    """
    Compute daily realized variance from intraday log returns by summing squared returns per day.
    Returns a daily series of variance (not annualized).
    """
    log_returns = utils.ensure_datetime_index(log_returns).dropna()
    daily_var = (log_returns ** 2).resample("1D").sum().dropna()
    daily_var.name = "daily_var"
    return daily_var


def compute_daily_variance_from_daily(log_returns_daily: pd.Series) -> pd.Series:
    log_returns_daily = utils.ensure_datetime_index(log_returns_daily).dropna()
    daily_var = (log_returns_daily ** 2).asfreq("1D").dropna()
    daily_var.name = "daily_var"
    return daily_var


def _har_design_matrix(daily_variance: pd.Series) -> pd.DataFrame:
    dv = daily_variance.copy()
    x_d = dv.shift(1)
    x_w = dv.rolling(window=config.HAR_WEEK_LENGTH, min_periods=1).mean().shift(1)
    x_m = dv.rolling(window=config.HAR_MONTH_LENGTH, min_periods=1).mean().shift(1)
    design = pd.concat(
        [
            pd.Series(1.0, index=dv.index, name="intercept"),
            x_d.rename("x_d"),
            x_w.rename("x_w"),
            x_m.rename("x_m"),
        ],
        axis=1,
    ).dropna()
    y = dv.reindex(design.index)
    return design, y


def fit_har_variance(daily_variance: pd.Series) -> HarVarianceModel:
    daily_variance = utils.ensure_datetime_index(daily_variance).dropna()
    daily_variance = daily_variance.tail(config.HAR_LOOKBACK_DAYS)
    design, y = _har_design_matrix(daily_variance)
    if len(design) < config.HAR_MIN_DAYS_FOR_FIT:
        raise ValueError("Not enough data to fit HAR model")
    x = design.values
    beta, *_ = np.linalg.lstsq(x, y.values, rcond=None)
    return HarVarianceModel(
        intercept=float(beta[0]),
        beta_daily=float(beta[1]),
        beta_weekly=float(beta[2]),
        beta_monthly=float(beta[3]),
    )


def forecast_next_day_variance(daily_variance: pd.Series, model: Optional[HarVarianceModel] = None) -> pd.Series:
    dv = utils.ensure_datetime_index(daily_variance).dropna()
    if model is None:
        model = fit_har_variance(dv)
    x_d = dv.shift(1)
    x_w = dv.rolling(window=config.HAR_WEEK_LENGTH, min_periods=1).mean().shift(1)
    x_m = dv.rolling(window=config.HAR_MONTH_LENGTH, min_periods=1).mean().shift(1)
    design = pd.concat([x_d.rename("x_d"), x_w.rename("x_w"), x_m.rename("x_m")], axis=1).dropna()
    preds = (
        model.intercept
        + model.beta_daily * design["x_d"]
        + model.beta_weekly * design["x_w"]
        + model.beta_monthly * design["x_m"]
    )
    preds.name = "forecast_var_next_day"
    return preds


def forecast_horizon_annualized_vol(daily_variance: pd.Series, horizon_days: int, model: Optional[HarVarianceModel] = None) -> pd.Series:
    """
    Approximate horizon forecast by using next-day variance forecast as a proxy for
    the mean daily variance over the horizon, then annualize and take square root.
    """
    next_day_var = forecast_next_day_variance(daily_variance, model=model)
    annualized_var = utils.annualize_daily_variance(next_day_var)
    ann_vol = np.sqrt(annualized_var)
    ann_vol.name = f"forecast_vol_{horizon_days}d"
    return ann_vol


