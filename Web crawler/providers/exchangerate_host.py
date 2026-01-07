from __future__ import annotations

from datetime import datetime

import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index, to_utc_timestamp


class ExchangeRateHostTimeseries(ProviderBase):
    BASE_URL = 'https://api.exchangerate.host/timeseries'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        base = self.params.get('base', 'USD')
        symbols = self.params.get('symbols', 'EUR')
        params = {
            'base': base,
            'symbols': symbols,
            'start_date': start.date().isoformat(),
            'end_date': end.date().isoformat(),
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        if resp.status_code == 404:
            return pd.DataFrame()
        resp.raise_for_status()
        data = resp.json()
        # API returns {'success': bool, 'timeseries': bool, 'rates': {...}}
        rates = data.get('rates', {}) if isinstance(data, dict) else {}
        if not rates:
            return pd.DataFrame()
        rows = []
        for day, mapping in rates.items():
            ts = to_utc_timestamp(pd.to_datetime(day))
            for sym, val in mapping.items():
                rows.append({'timestamp': ts, sym: float(val)})
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.groupby('timestamp').first().reset_index()
        return ensure_timestamp_index(df)


