from __future__ import annotations

from datetime import datetime

import os
import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index


class GlassnodeMetric(ProviderBase):
    BASE_URL = 'https://api.glassnode.com/v1/metrics'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        api_key = self.params.get('api_key') or os.getenv('GLASSNODE_API_KEY')
        if not api_key:
            return pd.DataFrame()
        asset = self.params.get('asset', 'BTC')
        metric = self.params.get('metric', 'market/price_usd_close')
        interval = self.params.get('interval', '24h')
        url = f"{self.BASE_URL}/{metric}"
        params = {
            'a': asset,
            's': int(start.timestamp()),
            'u': int(end.timestamp()),
            'i': interval,
            'api_key': api_key,
        }
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 403:
            return pd.DataFrame()
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['t'], unit='s', utc=True)
        df = df.rename(columns={'v': 'value'})
        df = df[['timestamp', 'value']]
        return ensure_timestamp_index(df)


