from __future__ import annotations

from datetime import datetime

import os
import pandas as pd
import requests

from core.base import ProviderBase
from core.utils import ensure_timestamp_index


class FREDSeries(ProviderBase):
    BASE_URL = 'https://api.stlouisfed.org/fred/series/observations'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        api_key = self.params.get('api_key') or os.getenv('FRED_API_KEY')
        if not api_key:
            return pd.DataFrame()
        series_id = self.params.get('series_id', 'DGS10')
        params = {
            'series_id': series_id,
            'api_key': api_key,
            'file_type': 'json',
            'observation_start': start.date().isoformat(),
            'observation_end': end.date().isoformat(),
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        observations = resp.json().get('observations', [])
        if not observations:
            return pd.DataFrame()
        df = pd.DataFrame(observations)
        df['timestamp'] = pd.to_datetime(df['date'], utc=True)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df[['timestamp', 'value']]
        return ensure_timestamp_index(df)


