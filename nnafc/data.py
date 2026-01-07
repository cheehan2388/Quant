from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# File loading and standardization
# -----------------------------

SYMBOL_PATTERN = re.compile(r"(?i)(BTC|ETH|BNB|SOL|XRP|DOGE|ADA|AVAX|BCH|LINK|SUI)[USDT]*")


def _canon(name: str) -> str:
	"""Canonicalize a column name for fuzzy matching: lowercase alnum only."""
	return re.sub(r"[^a-z0-9]", "", name.lower())


def _infer_symbol_from_filename(filename: str) -> Optional[str]:
	base = os.path.basename(filename)
	match = SYMBOL_PATTERN.search(base)
	return match.group(1).upper() + "USDT" if match else None


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
	cols = {c.lower(): c for c in df.columns}
	# Find datetime-like column
	for key in ["datetime", "date", "timestamp", "time"]:
		if key in cols:
			dt_col = cols[key]
			break
	else:
		raise ValueError("No datetime-like column found in data")
	
	df = df.copy()
	df[dt_col] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
	df = df.dropna(subset=[dt_col])
	df = df.sort_values(dt_col)
	df = df.set_index(dt_col)

	# Map OHLC and extended Binance columns
	# Build canonical map for fuzzy matching
	canon_map = {_canon(c): c for c in df.columns}

	def find_any(candidates: List[str]) -> Optional[str]:
		for cand in candidates:
			c = cand.lower()
			if c in cols:
				return cols[c]
			cc = _canon(cand)
			if cc in canon_map:
				return canon_map[cc]
		return None

	name_map = {
		"open": find_any(["open", "o"]),
		"high": find_any(["high", "h"]),
		"low": find_any(["low", "l"]),
		"close": find_any(["close", "c", "price"]),
		# Base asset volume
		"volume": find_any(["volume", "base asset volume", "v"]),
		# Additional Binance fields
		"quote_volume": find_any(["quote asset volume", "quote volume", "quote_volume", "quote asset vol"]),
		"num_trades": find_any(["number of trades", "num_trades", "trades"]),
		"taker_buy_base": find_any(["taker buy base asset volume", "takerbuybaseassetvolume"]),
		"taker_buy_quote": find_any(["taker buy quote asset volume", "takerbuyquoteassetvolume"]),
	}

	for needed in ["open", "high", "low", "close"]:
		if not name_map[needed]:
			raise ValueError(f"Missing required column: {needed}")

	std = pd.DataFrame(index=df.index)
	for target in ["open", "high", "low", "close"]:
		std[target] = pd.to_numeric(df[name_map[target]], errors="coerce")
	# Optional extended columns
	std["volume"] = pd.to_numeric(df.get(name_map.get("volume") or "volume", 0.0), errors="coerce").fillna(0.0)
	for extra in ["quote_volume", "num_trades", "taker_buy_base", "taker_buy_quote"]:
		src = name_map.get(extra)
		if src:
			std[extra] = pd.to_numeric(df[src], errors="coerce")
		else:
			std[extra] = np.nan

	std = std.dropna(subset=["open", "high", "low", "close"])  # allow others to be NaN
	return std


def load_symbol_csvs(data_dir: str, symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
	"""Load standardized OHLCV per symbol from a directory of Binance 1h CSVs.

	Returns a dict of symbol -> DataFrame with index 'datetime' (UTC) and columns [open, high, low, close, volume].
	"""
	files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
	data: Dict[str, pd.DataFrame] = {}
	for path in files:
		try:
			sym = _infer_symbol_from_filename(path)
			if sym is None:
				continue
			if symbols is not None and sym not in symbols:
				continue
			df = pd.read_csv(path)
			df = _standardize_columns(df)
			data[sym] = df
		except Exception:
			# Skip files that don't meet the schema
			continue
	return data


# -----------------------------
# Windowing, labels, normalization
# -----------------------------


def compute_forward_return(close: pd.Series, horizon: int, use_log: bool = True) -> pd.Series:
	future = close.shift(-horizon)
	if use_log:
		ret = np.log(future / close)
	else:
		ret = (future / close) - 1.0
	return ret


@dataclass
class TimeSeriesNormalizer:
	"""Feature-wise (per-channel) normalizer fitted on training windows only."""
	mean: Optional[np.ndarray] = None
	std: Optional[np.ndarray] = None
	eps: float = 1e-8

	def fit(self, windows: np.ndarray) -> None:
		# windows: [num_samples, num_features, lookback]
		# Compute per-feature mean/std across samples and time
		num_features = windows.shape[1]
		reduced = windows.reshape(windows.shape[0], num_features, -1)
		self.mean = reduced.mean(axis=(0, 2)).astype(np.float32)
		self.std = reduced.std(axis=(0, 2)).astype(np.float32)
		self.std[self.std < self.eps] = 1.0

	def transform(self, windows: np.ndarray) -> np.ndarray:
		if self.mean is None or self.std is None:
			raise RuntimeError("Normalizer must be fitted before calling transform().")
		return ((windows - self.mean[None, :, None]) / self.std[None, :, None]).astype(np.float32)


DEFAULT_FEATURES = [
	"open",
	"high",
	"low",
	"close",
	"volume",
	"quote_volume",
	"num_trades",
	"taker_buy_base",
	"taker_buy_quote",
]


class CrossSectionalBatcher:
	"""Construct cross-sectional batches over a unified hourly timeline.

	- Features: OHLCV → 5 channels
	- Window: lookback m hours ending at t
	- Label: forward return over horizon k hours: (close_{t+k}/close_t - 1)
	- Batches: for each eligible t, gather all symbols with full window and label
	"""

	def __init__(
		self,
		data: Dict[str, pd.DataFrame],
		lookback: int,
		horizon: int,
		train_range: Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]],
		val_range: Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]],
		test_range: Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]],
		feature_columns: Optional[List[str]] = None,
		use_log_return: bool = True,
	):
		self.data = data
		self.symbols = sorted(data.keys())
		self.lookback = lookback
		self.horizon = horizon
		self.use_log_return = use_log_return

		# Determine feature columns available across all symbols
		if feature_columns is None:
			available = set(DEFAULT_FEATURES)
			for df in data.values():
				available = available.intersection(set([c for c in DEFAULT_FEATURES if c in df.columns]))
			self.feature_columns = [c for c in DEFAULT_FEATURES if c in available]
		else:
			self.feature_columns = [c for c in feature_columns if c in DEFAULT_FEATURES]
		if len(self.feature_columns) == 0:
			raise ValueError("No valid feature columns available in data.")

		# Build a unified timeline (intersection of all symbols) to simplify batching
		indices = [df.index for df in data.values()]
		if not indices:
			raise ValueError("No data loaded for any symbol.")
		common_index = indices[0]
		for idx in indices[1:]:
			common_index = common_index.intersection(idx)
		common_index = common_index.sort_values()
		# Eligible timestamps with full lookback and forward label
		self.timeline = common_index[self.lookback - 1 : -self.horizon]

		# Precompute forward returns per symbol
		self.forward_returns: Dict[str, pd.Series] = {
			s: compute_forward_return(df["close"], horizon, use_log=self.use_log_return) for s, df in data.items()
		}

		# Split timeline
		self.train_ts = self._slice_timeline(self.timeline, *train_range)
		self.val_ts = self._slice_timeline(self.timeline, *val_range)
		self.test_ts = self._slice_timeline(self.timeline, *test_range)

		# Fit normalizer on training windows only
		train_windows = self._collect_windows(self.train_ts)
		self.normalizer = TimeSeriesNormalizer()
		self.normalizer.fit(train_windows)

	def _slice_timeline(
		self,
		timeline: pd.DatetimeIndex,
		start: Optional[pd.Timestamp],
		end: Optional[pd.Timestamp],
	) -> pd.DatetimeIndex:
		def to_utc(ts: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
			if ts is None:
				return None
			obj = pd.Timestamp(ts)
			if obj.tzinfo is None:
				return obj.tz_localize("UTC")
			return obj.tz_convert("UTC")

		start_utc = to_utc(start)
		end_utc = to_utc(end)
		if start_utc is not None:
			timeline = timeline[timeline >= start_utc]
		if end_utc is not None:
			timeline = timeline[timeline <= end_utc]
		return timeline

	def _collect_windows(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
		"""Collect all windows for given timestamps across symbols without normalization.

		Returns array of shape [num_samples_total, num_features, lookback]
		"""
		windows: List[np.ndarray] = []
		for t in timestamps:
			for s in self.symbols:
				df = self.data[s]
				# Ensure we use the last 'lookback' rows ending at t
				try:
					loc = df.index.get_loc(t)
				except KeyError:
					continue
				start = loc - (self.lookback - 1)
				if start < 0 or (loc + self.horizon) >= len(df):
					continue
				slice_df = df.iloc[start : loc + 1]
				feat = slice_df[self.feature_columns].to_numpy().T  # [F, m]
				windows.append(feat)
		F = len(self.feature_columns)
		return np.stack(windows) if windows else np.empty((0, F, self.lookback), dtype=np.float32)

	def _batch_at(self, t: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray, List[str]]:
		X_list: List[np.ndarray] = []
		Y_list: List[float] = []
		syms: List[str] = []
		for s in self.symbols:
			df = self.data[s]
			try:
				loc = df.index.get_loc(t)
			except KeyError:
				continue
			start = loc - (self.lookback - 1)
			if start < 0 or (loc + self.horizon) >= len(df):
				continue
			slice_df = df.iloc[start : loc + 1]
			feat = slice_df[self.feature_columns].to_numpy().T
			ret = float(self.forward_returns[s].iloc[loc])
			if np.isnan(ret):
				continue
			X_list.append(feat)
			Y_list.append(ret)
			syms.append(s)
		if not X_list:
			F = len(self.feature_columns)
			return np.empty((0, F, self.lookback), dtype=np.float32), np.empty((0, 1), dtype=np.float32), []
		X = np.stack(X_list).astype(np.float32)
		X = self.normalizer.transform(X)
		Y = np.asarray(Y_list, dtype=np.float32)[:, None]
		return X, Y, syms

	def iter_split(self, split: str) -> Iterable[Tuple[pd.Timestamp, np.ndarray, np.ndarray, List[str]]]:
		if split == "train":
			timestamps = self.train_ts
		elif split == "val":
			timestamps = self.val_ts
		elif split == "test":
			timestamps = self.test_ts
		else:
			raise ValueError("split must be one of: train, val, test")
		for t in timestamps:
			X, Y, syms = self._batch_at(t)
			if X.shape[0] == 0:
				continue
			yield t, X, Y, syms


