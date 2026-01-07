from __future__ import annotations

from datetime import datetime
from io import StringIO

import pandas as pd
import requests

from core.base import ProviderBase
from core.utils import ensure_timestamp_index, to_utc_timestamp


class ECBFxDaily(ProviderBase):
    # Historical daily FX against EUR
    URL = 'https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.csv'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        symbols = self.params.get('symbols', ['USD'])
        if isinstance(symbols, str):
            symbols = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        resp = requests.get(self.URL, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        # Normalize columns: strip and upper-case currency codes
        df.columns = [str(c).strip() for c in df.columns]
        # Handle date column variations
        date_col = 'Date' if 'Date' in df.columns else ('DATE' if 'DATE' in df.columns else None)
        if date_col is None:
            return pd.DataFrame()
        df = df.rename(columns={date_col: 'date'})
        # Ensure UTC without passing tz parameter alongside tz-aware values
        ts = pd.to_datetime(df['date'])
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize('UTC')
        else:
            ts = ts.dt.tz_convert('UTC')
        df['timestamp'] = ts
        # Some CSVs may have currency columns with inconsistent casing; align to upper for currencies only
        df = df.rename(columns={c: (c.upper() if str(c).lower() not in ('date', 'timestamp') else c) for c in df.columns})
        symbols = [s.upper() for s in symbols]
        # Compute currency columns after normalization
        currency_cols_all = [c for c in df.columns if c not in ('date', 'timestamp')]
        keep_cols = ['timestamp'] + [s for s in symbols if s in currency_cols_all]
        # Fallback: if requested symbols missing, keep all currency columns
        if len(keep_cols) == 1:
            currency_cols = currency_cols_all
            keep_cols = ['timestamp'] + currency_cols
        df = df[keep_cols]
        start_utc = to_utc_timestamp(start)
        end_utc = to_utc_timestamp(end)
        df = df[(df['timestamp'] >= start_utc) & (df['timestamp'] <= end_utc)]
        # Values are quoted as string like '1.2345' or empty
        for s in symbols:
            if s in df.columns:
                df[s] = pd.to_numeric(df[s], errors='coerce')
        return ensure_timestamp_index(df)


