from __future__ import annotations

from datetime import datetime

import os
import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index


class CoinAPITrades(ProviderBase):
    BASE_URL = 'https://rest.coinapi.io/v1/ohlcv/{symbol}/history'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        api_key = self.params.get('api_key') or os.getenv('COINAPI_KEY')
        if not api_key:
            return pd.DataFrame()
        symbol = self.params.get('symbol', 'BITSTAMP_SPOT_BTC_USD')
        period_id = self.params.get('period_id', '1HRS')
        params = {
            'time_start': start.isoformat(),
            'time_end': end.isoformat(),
            'limit': 1000,
        }
        url = self.BASE_URL.format(symbol=symbol)
        headers = {'X-CoinAPI-Key': api_key}
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if 'time_period_start' not in df.columns or 'price_close' not in df.columns:
            return pd.DataFrame()
        df['timestamp'] = pd.to_datetime(df['time_period_start'], utc=True)
        df = df.rename(columns={'price_open': 'open', 'price_high': 'high', 'price_low': 'low', 'price_close': 'close', 'volume_traded': 'volume'})
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        return ensure_timestamp_index(df)


