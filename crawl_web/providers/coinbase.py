from __future__ import annotations

from datetime import datetime

import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index


class CoinbaseCandles(ProviderBase):
    BASE_URL = 'https://api.exchange.coinbase.com/products/{product_id}/candles'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        product_id = self.params.get('product_id', 'BTC-USD')
        granularity = int(self.params.get('granularity', 3600))
        params = {'granularity': granularity, 'start': start.isoformat(), 'end': end.isoformat()}
        url = self.BASE_URL.format(product_id=product_id)
        resp = requests.get(url, params=params, timeout=30, headers={'User-Agent': 'crawler'})
        resp.raise_for_status()
        data = resp.json()
        if not data or isinstance(data, dict):
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        return ensure_timestamp_index(df)


