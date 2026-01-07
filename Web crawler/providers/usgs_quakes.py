from __future__ import annotations

from datetime import datetime

import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index


class USGSEarthquakes(ProviderBase):
    BASE_URL = 'https://earthquake.usgs.gov/fdsnws/event/1/query'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        params = {
            'format': 'geojson',
            'starttime': start.isoformat(),
            'endtime': end.isoformat(),
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        features = data.get('features', [])
        if not features:
            return pd.DataFrame()
        rows = []
        for f in features:
            props = f.get('properties', {})
            ts_ms = props.get('time')
            if ts_ms is None:
                continue
            rows.append({
                'timestamp': pd.to_datetime(ts_ms, unit='ms', utc=True),
                'mag': props.get('mag'),
                'place': props.get('place'),
                'depth': f.get('geometry', {}).get('coordinates', [None, None, None])[2],
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df[['timestamp', 'mag', 'depth']]
        return ensure_timestamp_index(df)


