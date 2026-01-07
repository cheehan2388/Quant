from __future__ import annotations

from datetime import datetime

import os
import pandas as pd
import requests

from core.base import ProviderBase
from core.utils import ensure_timestamp_index, to_utc_timestamp


class AlphaVantageFXDaily(ProviderBase):
    BASE_URL = 'https://www.alphavantage.co/query'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        api_key = self.params.get('api_key') or os.getenv('ALPHAVANTAGE_API_KEY')
        if not api_key:
            return pd.DataFrame()
        from_symbol = self.params.get('from_symbol', 'USD')
        to_symbol = self.params.get('to_symbol', 'JPY')
        params = {
            'function': 'FX_DAILY',
            'from_symbol': from_symbol,
            'to_symbol': to_symbol,
            'outputsize': 'full',
            'apikey': api_key,
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get('Time Series FX (Daily)', {})
        if not data:
            return pd.DataFrame()
        rows = []
        for ds, mapping in data.items():
            ts = pd.to_datetime(ds, utc=True)
            if ts < to_utc_timestamp(start) or ts > to_utc_timestamp(end):
                continue
            rows.append({
                'timestamp': ts,
                'open': float(mapping['1. open']),
                'high': float(mapping['2. high']),
                'low': float(mapping['3. low']),
                'close': float(mapping['4. close']),
            })
        df = pd.DataFrame(rows)
        return ensure_timestamp_index(df)


