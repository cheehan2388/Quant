from __future__ import annotations

from datetime import datetime
from typing import List

import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index


class CoinMetricsCommunity(ProviderBase):
    BASE_URL = 'https://community-api.coinmetrics.io/v4/timeseries/asset-metrics'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        assets: List[str] = self.params.get('assets', ['btc'])
        metrics: List[str] = self.params.get('metrics', ['PriceUSD'])
        frequency = self.params.get('frequency', '1D').upper()
        params = {
            'assets': ','.join(assets),
            'metrics': ','.join(metrics),
            'start_time': start.date().isoformat(),
            'end_time': end.date().isoformat(),
            'frequency': '1d' if frequency.endswith('D') else '1h',
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=60)
        if resp.status_code == 401:
            # Community endpoint may require API key; skip silently
            return pd.DataFrame()
        resp.raise_for_status()
        data = resp.json().get('data', [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['time'], utc=True)
        # Flatten metrics columns
        value_cols = {}
        for m in metrics:
            col = m
            if m not in df.columns:
                continue
            value_cols[m] = m
        keep = ['timestamp', 'asset'] + list(value_cols.values())
        df = df[keep]
        pivoted = df.pivot(index='timestamp', columns='asset', values=list(value_cols.values()))
        # Flatten multi-index columns: (metric, asset) -> f"{asset}_{metric}"
        pivoted.columns = [f"{a}_{m}" for m, a in pivoted.columns]
        pivoted = pivoted.reset_index()
        return ensure_timestamp_index(pivoted)


