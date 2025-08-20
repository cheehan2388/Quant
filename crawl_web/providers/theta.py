from __future__ import annotations

from datetime import datetime

import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index


class ThetaNetworkPrice(ProviderBase):
    # Example third-party endpoint (replace with official when available)
    URL = 'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        coin_id = self.params.get('coin_id', 'theta-token')
        params = {
            'vs_currency': self.params.get('vs_currency', 'usd'),
            'from': int(start.timestamp()),
            'to': int(end.timestamp()),
        }
        url = self.URL.format(coin_id=coin_id)
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 404:
                # Invalid coin id or unsupported; treat as empty
                return pd.DataFrame()
            resp.raise_for_status()
        except requests.RequestException:
            return pd.DataFrame()
        prices = resp.json().get('prices', [])
        if not prices:
            return pd.DataFrame()
        df = pd.DataFrame(prices, columns=['ts', 'price'])
        df['timestamp'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
        df = df[['timestamp', 'price']]
        return ensure_timestamp_index(df)


