from __future__ import annotations

from datetime import datetime

import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index, to_utc_timestamp


class FearGreedIndex(ProviderBase):
    URL = 'https://api.alternative.me/fng/'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        params = {'limit': 0, 'format': 'json'}
        resp = requests.get(self.URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get('data', [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s', utc=True)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df[['timestamp', 'value']]
        start_utc = to_utc_timestamp(start)
        end_utc = to_utc_timestamp(end)
        df = df[(df['timestamp'] >= start_utc) & (df['timestamp'] <= end_utc)]
        # daily frequency
        df = ensure_timestamp_index(df).resample('1D').last()
        return df


