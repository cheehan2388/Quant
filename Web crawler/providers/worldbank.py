from __future__ import annotations

from datetime import datetime

import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index


class WorldBankIndicator(ProviderBase):
    BASE_URL = 'https://api.worldbank.org/v2/country/{country}/indicator/{indicator}'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        country = self.params.get('country', 'US')
        indicator = self.params.get('indicator', 'SP.POP.TOTL')
        url = self.BASE_URL.format(country=country, indicator=indicator)
        params = {'format': 'json', 'date': f"{start.year}:{end.year}"}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or len(data) < 2:
            return pd.DataFrame()
        rows = data[1]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df[['date', 'value']].rename(columns={'date': 'year'})
        df['timestamp'] = pd.to_datetime(df['year'], format='%Y', utc=True)
        df = df[['timestamp', 'value']]
        return ensure_timestamp_index(df)


