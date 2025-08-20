from __future__ import annotations

from datetime import datetime

import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index, to_utc_timestamp


class OkxCandles(ProviderBase):
    BASE_URL = 'https://www.okx.com/api/v5/market/history-candles'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        inst_id = self.params.get('instId', 'BTC-USDT')
        bar = self.params.get('bar', '1H')
        params = {
            'instId': inst_id,
            'bar': bar,
            'after': int(end.timestamp() * 1000),
            'limit': 300,
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get('data', [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'volCcy', 'volCcyQuote', 'confirm'])
        # Avoid potential int overflow by coercing to int64 via to_numeric
        df['timestamp'] = pd.to_datetime(pd.to_numeric(df['ts'], errors='coerce').astype('int64'), unit='ms', utc=True)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'vol']].rename(columns={'vol': 'volume'})
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        start_utc = to_utc_timestamp(start)
        end_utc = to_utc_timestamp(end)
        df = df[(df['timestamp'] >= start_utc) & (df['timestamp'] <= end_utc)]
        return ensure_timestamp_index(df)


