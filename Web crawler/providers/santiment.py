from __future__ import annotations

from datetime import datetime

import os
import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index


class SantimentMetric(ProviderBase):
    URL = 'https://api.santiment.net/graphql'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        api_key = self.params.get('api_key') or os.getenv('SANTIMENT_API_KEY')
        if not api_key:
            return pd.DataFrame()
        slug = self.params.get('slug', 'bitcoin')
        metric = self.params.get('metric', 'transaction_volume')
        interval = self.params.get('interval', '1d')
        query = """
        query ($slug: String!, $from: DateTime!, $to: DateTime!, $interval: String!) {
          getMetric(metric: $metric){
            timeseriesData(selector: {slug: $slug}, from: $from, to: $to, interval: $interval){
              datetime
              value
            }
          }
        }
        """
        payload = {
            'query': query.replace('$metric', metric),
            'variables': {'slug': slug, 'from': start.isoformat(), 'to': end.isoformat(), 'interval': interval}
        }
        headers = {'Authorization': f'Apikey {api_key}'}
        resp = requests.post(self.URL, json=payload, headers=headers, timeout=60)
        if resp.status_code != 200:
            return pd.DataFrame()
        d = resp.json()
        series = (
            d.get('data', {})
             .get('getMetric', {})
             .get('timeseriesData', [])
        )
        if not series:
            return pd.DataFrame()
        df = pd.DataFrame(series)
        df['timestamp'] = pd.to_datetime(df['datetime'], utc=True)
        df = df[['timestamp', 'value']]
        return ensure_timestamp_index(df)


