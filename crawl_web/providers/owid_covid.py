from __future__ import annotations

from datetime import datetime

import pandas as pd
import requests
from io import StringIO

from core.base import ProviderBase
from core.utils import ensure_timestamp_index, to_utc_timestamp


class OWIDCovidCountry(ProviderBase):
    URL = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        iso_code = self.params.get('iso_code', 'USA')
        # Fetch full dataset; filter by date-range locally. Dataset is not huge.
        try:
            resp = requests.get(self.URL, timeout=60)
            resp.raise_for_status()
        except requests.RequestException:
            return pd.DataFrame()
        df = pd.read_csv(StringIO(resp.text))
        if df.empty:
            return df
        df = df[df['iso_code'] == iso_code]
        if df.empty:
            return pd.DataFrame()
        df['timestamp'] = pd.to_datetime(df['date'], utc=True)
        # select a subset of columns
        cols = [c for c in ['new_cases', 'new_deaths', 'new_vaccinations', 'stringency_index'] if c in df.columns]
        out = df[['timestamp'] + cols]
        start_utc = to_utc_timestamp(start)
        end_utc = to_utc_timestamp(end)
        out = out[(out['timestamp'] >= start_utc) & (out['timestamp'] <= end_utc)]
        return ensure_timestamp_index(out)


