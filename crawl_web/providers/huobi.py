from __future__ import annotations

from datetime import datetime

import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index


class HuobiKlines(ProviderBase):
    BASE_URL = 'https://api.huobi.pro/market/history/kline'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        symbol = self.params.get('symbol', 'btcusdt')
        period = self.params.get('period', '60min')
        params = {'symbol': symbol, 'period': period, 'size': 2000}
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get('data', [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['id'].astype(int), unit='s', utc=True)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'vol']].rename(columns={'vol': 'volume'})
        return ensure_timestamp_index(df)


