from __future__ import annotations

from datetime import datetime

import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index


class CoinpaprikaHistorical(ProviderBase):
    BASE_URL = 'https://api.coinpaprika.com/v1/tickers/{coin_id}/historical'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        coin_id = self.params.get('coin_id', 'btc-bitcoin')
        params = {
            'start': start.isoformat(),
            'end': end.isoformat(),
            'interval': '1d',
        }
        url = self.BASE_URL.format(coin_id=coin_id)
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df[['timestamp', 'price', 'volume_24h']]
        return ensure_timestamp_index(df)


