from __future__ import annotations

from datetime import datetime

import pandas as pd
import requests

from core.base import ProviderBase
from core.utils import ensure_timestamp_index, to_utc_timestamp


class COTFutures(ProviderBase):
    # CFTC commitments of traders (via Quandl replacement: unofficial CSV mirror)
    URL = 'https://www.cftc.gov/dea/newcot/f_disagg_txt_2020.csv'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        resp = requests.get(self.URL, timeout=60)
        if resp.status_code != 200:
            return pd.DataFrame()
        df = pd.read_csv(pd.compat.StringIO(resp.text)) if hasattr(pd, 'compat') else pd.read_csv(self.URL)
        # Normalize columns; dataset varies, keep date and a few positioning columns if available
        if 'Report_Date_as_YYYY-MM-DD' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Report_Date_as_YYYY-MM-DD'], utc=True)
        elif 'Report_Date_as_YYYYMMDD' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Report_Date_as_YYYYMMDD'], format='%Y%m%d', utc=True)
        else:
            return pd.DataFrame()
        keep_candidates = [
            'Prod_Merc_Positions_Long_All',
            'Prod_Merc_Positions_Short_All',
            'Swap_Positions_Long_All',
            'Swap_Positions_Short_All',
            'M_Money_Positions_Long_All',
            'M_Money_Positions_Short_All',
        ]
        cols = [c for c in keep_candidates if c in df.columns]
        if not cols:
            return pd.DataFrame()
        out = df[['timestamp'] + cols]
        start_utc = to_utc_timestamp(start)
        end_utc = to_utc_timestamp(end)
        out = out[(out['timestamp'] >= start_utc) & (out['timestamp'] <= end_utc)]
        for c in cols:
            out[c] = pd.to_numeric(out[c], errors='coerce')
        return ensure_timestamp_index(out)


