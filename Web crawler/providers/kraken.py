from __future__ import annotations

from datetime import datetime

import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index, to_utc_timestamp


class KrakenOHLC(ProviderBase):
    BASE_URL = 'https://api.kraken.com/0/public/OHLC'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        pair = self.params.get('pair', 'XBTUSD')
        interval_minutes = int(self.params.get('interval_minutes', 60))
        params = {'pair': pair, 'interval': interval_minutes, 'since': int(start.timestamp())}
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get('result', {})
        series = None
        for k, v in data.items():
            if isinstance(v, list):
                series = v
                break
        if not series:
            return pd.DataFrame()
        df = pd.DataFrame(series, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
        df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)
        start_utc = to_utc_timestamp(start)
        end_utc = to_utc_timestamp(end)
        df = df[(df['timestamp'] >= start_utc) & (df['timestamp'] <= end_utc)]
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return ensure_timestamp_index(df)


