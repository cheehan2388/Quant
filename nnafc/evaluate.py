from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_daily_spearman_ic(factor_df: pd.DataFrame, return_df: pd.DataFrame) -> pd.DataFrame:
	"""Both DataFrames have a MultiIndex (datetime, symbol) and a column named 'value'.

	Returns a DataFrame indexed by datetime with columns ['ic'].
	"""
	# Align indices
	merged = factor_df.join(return_df, lsuffix="_factor", rsuffix="_ret", how="inner")
	merged = merged.reset_index()
	ics: List[Tuple[pd.Timestamp, float]] = []
	for dt, group in merged.groupby("datetime"):
		if group.shape[0] < 3:
			continue
		corr, _ = spearmanr(group["value_factor"], group["value_ret"])  # type: ignore
		if np.isnan(corr):
			continue
		ics.append((pd.Timestamp(dt), float(corr)))
	return pd.DataFrame(ics, columns=["datetime", "ic"]).set_index("datetime").sort_index()


def ic_summary(ic_series: pd.Series) -> Dict[str, float]:
	mu = float(ic_series.mean()) if len(ic_series) else float("nan")
	sd = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else float("nan")
	ir = mu / sd if sd and sd == sd and mu == mu else float("nan")
	return {"mean": mu, "std": sd, "ir": ir}


def factor_correlation(factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
	"""Compute correlation matrix across factors. Each df is MultiIndex (datetime, symbol) with 'value'."""
	# Pivot each factor to align on (datetime, symbol)
	aligned = []
	for name, df in factors.items():
		w = df.copy()
		w = w.rename(columns={"value": name})
		aligned.append(w)
	merged = aligned[0]
	for w in aligned[1:]:
		merged = merged.join(w, how="inner")
	return merged.corr()


def quintile_backtest(factor_df: pd.DataFrame, return_df: pd.DataFrame, q: int = 5) -> pd.DataFrame:
	"""Long top 1/q, short bottom 1/q. Returns cumulative return series of L-S portfolio."""
	merged = factor_df.join(return_df, lsuffix="_factor", rsuffix="_ret", how="inner").reset_index()
	rets: List[Tuple[pd.Timestamp, float]] = []
	for dt, group in merged.groupby("datetime"):
		if group.shape[0] < q:
			continue
		g = group.sort_values("value_factor")
		bottom = g.head(len(g)//q)["value_ret"].mean()
		top = g.tail(len(g)//q)["value_ret"].mean()
		rets.append((pd.Timestamp(dt), float(top - bottom)))
	ret_df = pd.DataFrame(rets, columns=["datetime", "ls_return"]).set_index("datetime").sort_index()
	ret_df["cum_return"] = (1 + ret_df["ls_return"]).cumprod() - 1
	return ret_df


