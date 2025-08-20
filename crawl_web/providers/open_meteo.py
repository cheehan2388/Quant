from __future__ import annotations

from datetime import datetime

import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index, to_utc_timestamp


class OpenMeteoHourly(ProviderBase):
    BASE_URL = 'https://api.open-meteo.com/v1/forecast'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        latitude = float(self.params.get('latitude', 25.0375))
        longitude = float(self.params.get('longitude', 121.5637))
        hourly = self.params.get('hourly', 'temperature_2m')
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'hourly': hourly,
            'start_hour': start.strftime('%Y-%m-%dT%H:%M'),
            'end_hour': end.strftime('%Y-%m-%dT%H:%M'),
            'timezone': 'UTC',
        }
        # Open-Meteo uses date range via start_date/end_date. We'll fetch larger set and slice.
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'hourly': hourly,
            'timezone': 'UTC',
            'past_days': 90,
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        hourly_data = data.get('hourly', {})
        times = hourly_data.get('time', [])
        values = hourly_data.get(hourly, [])
        if not times or not values:
            return pd.DataFrame()
        df = pd.DataFrame({'timestamp': pd.to_datetime(times, utc=True), hourly: values})
        start_utc = to_utc_timestamp(start)
        end_utc = to_utc_timestamp(end)
        df = df[(df['timestamp'] >= start_utc) & (df['timestamp'] <= end_utc)]
        return ensure_timestamp_index(df)


