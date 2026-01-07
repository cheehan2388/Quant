from __future__ import annotations

from datetime import datetime

import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index, to_utc_timestamp


class BinanceOpenInterest(ProviderBase):
    BASE_URL = 'https://fapi.binance.com/futures/data/openInterestHist'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        symbol = self.params.get('symbol', 'BTCUSDT')
        period = self.params.get('period', '1h')
        limit = 500
        params = {
            'symbol': symbol,
            'period': period,
            'limit': limit,
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms', utc=True)
        df['sumOpenInterest'] = pd.to_numeric(df['sumOpenInterest'], errors='coerce')
        df = df[['timestamp', 'sumOpenInterest']].rename(columns={'sumOpenInterest': 'open_interest'})
        # Slice by requested time window
        start_utc = to_utc_timestamp(start)
        end_utc = to_utc_timestamp(end)
        df = df[(df['timestamp'] >= start_utc) & (df['timestamp'] <= end_utc)]
        return ensure_timestamp_index(df)


