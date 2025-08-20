from __future__ import annotations

from datetime import datetime

import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index


class BitfinexCandles(ProviderBase):
    BASE_URL = 'https://api-pub.bitfinex.com/v2/candles/trade:{timeframe}:{symbol}/hist'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        symbol = self.params.get('symbol', 'tBTCUSD')
        timeframe = self.params.get('timeframe', '1h')
        params = {
            'start': int(start.timestamp() * 1000),
            'end': int(end.timestamp() * 1000),
            'sort': 1,
            'limit': 1000,
        }
        url = self.BASE_URL.format(timeframe=timeframe, symbol=symbol)
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=['ts', 'open', 'close', 'high', 'low', 'volume'])
        df['timestamp'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        return ensure_timestamp_index(df)


