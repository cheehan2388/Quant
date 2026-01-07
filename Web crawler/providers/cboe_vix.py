from __future__ import annotations

from datetime import datetime
from io import StringIO

import pandas as pd
import requests

from core.base import ProviderBase
from core.utils import ensure_timestamp_index, to_utc_timestamp


class CBOEVIXDaily(ProviderBase):
    URL = 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        resp = requests.get(self.URL, timeout=60)
        if resp.status_code != 200:
            return pd.DataFrame()
        df = pd.read_csv(StringIO(resp.text))
        # Expect columns: DATE, OPEN, HIGH, LOW, CLOSE
        date_col = 'DATE' if 'DATE' in df.columns else 'Date'
        if date_col not in df.columns:
            return pd.DataFrame()
        df['timestamp'] = pd.to_datetime(df[date_col], utc=True)
        df = df.rename(columns={'CLOSE': 'close', 'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low'})
        keep = [c for c in ['open', 'high', 'low', 'close'] if c in df.columns]
        out = df[['timestamp'] + keep]
        start_utc = to_utc_timestamp(start)
        end_utc = to_utc_timestamp(end)
        out = out[(out['timestamp'] >= start_utc) & (out['timestamp'] <= end_utc)]
        for c in keep:
            out[c] = pd.to_numeric(out[c], errors='coerce')
        return ensure_timestamp_index(out)


