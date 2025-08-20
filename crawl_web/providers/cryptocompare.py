from __future__ import annotations

from datetime import datetime

import os
import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index


class CryptoCompareHourly(ProviderBase):
    BASE_URL = 'https://min-api.cryptocompare.com/data/v2/histohour'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        api_key = self.params.get('api_key') or os.getenv('CRYPTOCOMPARE_API_KEY')
        fsym = self.params.get('fsym', 'BTC')
        tsym = self.params.get('tsym', 'USD')
        limit = min(2000, int((end - start).total_seconds() // 3600))
        params = {
            'fsym': fsym,
            'tsym': tsym,
            'limit': max(1, limit),
            'toTs': int(end.timestamp()),
        }
        headers = {'authorization': f'Apikey {api_key}'} if api_key else {}
        resp = requests.get(self.BASE_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json().get('Data', {}).get('Data', [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volumefrom']].rename(columns={'volumefrom': 'volume'})
        return ensure_timestamp_index(df)


