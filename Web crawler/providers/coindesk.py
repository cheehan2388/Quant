from __future__ import annotations

from datetime import datetime

import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index


class CoindeskBPI(ProviderBase):
    BASE_URL = 'https://api.coindesk.com/v1/bpi/historical/close.json'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        currency = self.params.get('currency', 'USD')
        params = {'start': start.date().isoformat(), 'end': end.date().isoformat(), 'currency': currency}
        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
        except Exception:
            # Network DNS failure or temporary outage: skip quietly
            return pd.DataFrame()
        data = resp.json().get('bpi', {})
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(list(data.items()), columns=['date', 'price'])
        df['timestamp'] = pd.to_datetime(df['date'], utc=True)
        df = df[['timestamp', 'price']]
        return ensure_timestamp_index(df)


