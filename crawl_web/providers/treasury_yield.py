from __future__ import annotations

from datetime import datetime
from io import StringIO

import pandas as pd
import requests

from core.base import ProviderBase
from core.utils import ensure_timestamp_index, to_utc_timestamp


class USTreasuryDaily(ProviderBase):
    URL = 'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        resp = requests.get(self.URL, timeout=60)
        if resp.status_code != 200:
            return pd.DataFrame()
        df = pd.read_csv(StringIO(resp.text))
        # Standardize
        date_col = None
        for c in df.columns:
            if str(c).lower().startswith('date'):
                date_col = c
                break
        if date_col is None:
            return pd.DataFrame()
        df['timestamp'] = pd.to_datetime(df[date_col], utc=True)
        # Keep a subset of maturities if present
        keep = [c for c in ['1 Mo', '3 Mo', '6 Mo', '1 Yr', '2 Yr', '5 Yr', '10 Yr', '30 Yr'] if c in df.columns]
        if not keep:
            return pd.DataFrame()
        out = df[['timestamp'] + keep]
        start_utc = to_utc_timestamp(start)
        end_utc = to_utc_timestamp(end)
        out = out[(out['timestamp'] >= start_utc) & (out['timestamp'] <= end_utc)]
        for c in keep:
            out[c] = pd.to_numeric(out[c], errors='coerce')
        return ensure_timestamp_index(out)


