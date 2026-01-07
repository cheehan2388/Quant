from __future__ import annotations

from datetime import datetime

import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index


class BitstampOHLC(ProviderBase):
    BASE_URL = 'https://www.bitstamp.net/api/v2/ohlc/{pair}/'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        pair = self.params.get('pair', 'btcusd')
        step = int(self.params.get('step', 3600))
        params = {
            'step': step,
            'limit': 1000,
            'start': int(start.timestamp()),
            'end': int(end.timestamp()),
        }
        url = self.BASE_URL.format(pair=pair)
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get('data', {}).get('ohlc', [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s', utc=True)
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        return ensure_timestamp_index(df)


