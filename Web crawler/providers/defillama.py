from __future__ import annotations

from datetime import datetime

import pandas as pd
import requests

from core.base import ProviderBase
from core.utils import ensure_timestamp_index, to_utc_timestamp


class DefiLlamaTVL(ProviderBase):
    BASE_URL = 'https://api.llama.fi/overview/fees'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        resp = requests.get(self.BASE_URL, timeout=60)
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json().get('totalDataChart', [])
        if not data:
            return pd.DataFrame()
        # array of [timestamp, value]
        df = pd.DataFrame(data, columns=['ts', 'value'])
        df['timestamp'] = pd.to_datetime(df['ts'], unit='s', utc=True)
        start_utc = to_utc_timestamp(start)
        end_utc = to_utc_timestamp(end)
        df = df[(df['timestamp'] >= start_utc) & (df['timestamp'] <= end_utc)]
        df = df[['timestamp', 'value']]
        return ensure_timestamp_index(df)


