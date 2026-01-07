from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config
from . import utils
from . import har_rv
from . import signals
from . import execution


@dataclass
class BacktestInputs:
    prices_csv: Path  # CSV with datetime, price columns
    iv_csv: Optional[Path] = None  # CSV with datetime, iv columns (annualized vol)
    datetime_col: str = "datetime"
    price_col: str = "price"
    iv_col: str = "iv"
    freq: str = "5min"  # frequency of prices


def load_prices(csv_path: Path, datetime_col: str, price_col: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    ts = pd.to_datetime(df[datetime_col])
    price = pd.Series(df[price_col].values, index=ts, name="price").sort_index()
    return price


def load_iv(csv_path: Path, datetime_col: str, iv_col: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    ts = pd.to_datetime(df[datetime_col])
    iv = pd.Series(df[iv_col].values, index=ts, name="iv").sort_index()
    return iv


def compute_intraday_returns(price: pd.Series) -> pd.Series:
    return utils.compute_log_returns_from_prices(price)


def daily_rv_from_intraday(log_returns: pd.Series) -> pd.Series:
    return har_rv.compute_daily_variance_from_intraday(log_returns)


def forward_fill_daily(series: pd.Series) -> pd.Series:
    daily = utils.as_daily(series, how="last")
    return daily.ffill()


def run_backtest(inputs: BacktestInputs) -> pd.DataFrame:
    price = load_prices(inputs.prices_csv, inputs.datetime_col, inputs.price_col)
    log_ret = compute_intraday_returns(price)
    daily_var = daily_rv_from_intraday(log_ret)

    forecast_vol_30d = har_rv.forecast_horizon_annualized_vol(daily_var, horizon_days=30)

    if inputs.iv_csv is not None:
        iv_series = load_iv(inputs.iv_csv, inputs.datetime_col, inputs.iv_col)
        iv_daily = forward_fill_daily(iv_series)
        iv_daily = iv_daily.reindex(forecast_vol_30d.index).ffill().dropna()
    else:
        # If no IV provided, build a naive proxy from realized vol plus a premium
        naive_iv = np.sqrt(utils.annualize_daily_variance(daily_var)) * 1.1
        iv_daily = naive_iv.reindex(forecast_vol_30d.index).ffill().dropna()

    sig_df = signals.build_vrp_signal(
        forecast_ann_vol=forecast_vol_30d,
        implied_ann_vol=iv_daily,
    )

    pnl_df = execution.simulate_variance_swap_like_pnl(
        vega_target=sig_df["vega_target"],
        realized_daily_variance=daily_var.reindex(sig_df.index).dropna(),
        implied_annualized_vol=iv_daily.reindex(sig_df.index).dropna(),
    )

    result = sig_df.join(pnl_df, how="inner")
    return result


def main():
    parser = argparse.ArgumentParser(description="VRP vol-arb backtest scaffold")
    parser.add_argument("--prices_csv", type=str, required=True, help="CSV with datetime, price")
    parser.add_argument("--iv_csv", type=str, required=False, default=None, help="CSV with datetime, iv (annualized)")
    parser.add_argument("--datetime_col", type=str, default="datetime")
    parser.add_argument("--price_col", type=str, default="price")
    parser.add_argument("--iv_col", type=str, default="iv")
    parser.add_argument("--out_csv", type=str, default=None, help="Write results to CSV")
    args = parser.parse_args()

    inputs = BacktestInputs(
        prices_csv=Path(args.prices_csv),
        iv_csv=Path(args.iv_csv) if args.iv_csv else None,
        datetime_col=args.datetime_col,
        price_col=args.price_col,
        iv_col=args.iv_col,
    )

    res = run_backtest(inputs)
    if args.out_csv:
        res.to_csv(args.out_csv, index=True)
    else:
        print(res.tail(10).to_string())


if __name__ == "__main__":
    main()


