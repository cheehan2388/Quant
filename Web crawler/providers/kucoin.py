from __future__ import annotations

from datetime import datetime

import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index


class KucoinCandles(ProviderBase):
    BASE_URL = 'https://api.kucoin.com/api/v1/market/candles'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        symbol = self.params.get('symbol', 'BTC-USDT')
        typ = self.params.get('type', '1hour')
        params = {
            'type': typ,
            'symbol': symbol,
            'startAt': int(start.timestamp()),
            'endAt': int(end.timestamp()),
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get('data', [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=['time', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
        df['timestamp'] = pd.to_datetime(df['time'].astype(int), unit='s', utc=True)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return ensure_timestamp_index(df)


