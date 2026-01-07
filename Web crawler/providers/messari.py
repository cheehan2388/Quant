from __future__ import annotations

from datetime import datetime

import os
import pandas as pd
import requests

from core.base import ProviderBase
from core.utils import ensure_timestamp_index


class MessariAssetTimeseries(ProviderBase):
    BASE_URL = 'https://data.messari.io/api/v1/assets/{asset}/metrics/price/time-series'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        api_key = self.params.get('api_key') or os.getenv('MESSARI_API_KEY')
        asset = self.params.get('asset', 'btc')
        interval = self.params.get('interval', '1d')
        url = self.BASE_URL.format(asset=asset)
        params = {
            'start': start.isoformat(),
            'end': end.isoformat(),
            'interval': interval,
        }
        headers = {'x-messari-api-key': api_key} if api_key else {}
        resp = requests.get(url, params=params, headers=headers, timeout=60)
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json().get('data', {}).get('values', [])
        if not data:
            return pd.DataFrame()
        # values are [time, open, high, low, close, volume, marketcap]
        df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close', 'volume', 'marketcap'])
        df['timestamp'] = pd.to_datetime(df['ts'], utc=True, unit='ms')
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'marketcap']]
        return ensure_timestamp_index(df)


