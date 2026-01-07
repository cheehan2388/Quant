from __future__ import annotations

from datetime import datetime
from typing import Dict, Any

import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index, to_utc_timestamp


class BybitKlines(ProviderBase):
    BASE_URL = 'https://api.bybit.com/v5/market/kline'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        symbol = self.params.get('symbol', 'BTCUSDT')
        category = self.params.get('category', 'linear')  # 'linear' for USDT perp, 'spot' for spot
        interval = self.params.get('interval', '60')  # minutes: '60' for 1h, 'D' for 1d
        params: Dict[str, Any] = {
            'category': category,
            'symbol': symbol,
            'interval': interval,
            'start': int(start.timestamp() * 1000),
            'end': int(end.timestamp() * 1000),
            'limit': 1000,
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        result = resp.json().get('result', {})
        rows = result.get('list', [])
        if not rows:
            return pd.DataFrame()
        # Each row: [start, open, high, low, close, volume, turnover]
        cols = ['ts', 'open', 'high', 'low', 'close', 'volume', 'turnover']
        df = pd.DataFrame(rows, columns=cols)
        # Convert robustly to int64 to avoid overflow
        df['timestamp'] = pd.to_datetime(pd.to_numeric(df['ts'], errors='coerce').astype('int64'), unit='ms', utc=True)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return ensure_timestamp_index(df)


