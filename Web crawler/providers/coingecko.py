from __future__ import annotations

from datetime import datetime
import time

import requests
from requests.adapters import HTTPAdapter, Retry
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index


class CoingeckoMarketChart(ProviderBase):
    BASE_URL = 'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range'

    def _session(self) -> requests.Session:
        s = requests.Session()
        retries = Retry(
            total=6,
            connect=6,
            read=6,
            backoff_factor=0.8,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        s.mount('https://', HTTPAdapter(max_retries=retries))
        return s

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        coin_id = self.params.get('coin_id', 'bitcoin')
        vs_currency = self.params.get('vs_currency', 'usd')
        url = self.BASE_URL.format(coin_id=coin_id)
        params = {
            'vs_currency': vs_currency,
            'from': int(start.timestamp()),
            'to': int(end.timestamp()),
        }
        sess = self._session()
        headers = {'User-Agent': 'cndatest-data-collector/1.0', 'Accept': 'application/json'}
        for attempt in range(6):
            try:
                resp = sess.get(url, params=params, timeout=30, headers=headers)
                if resp.status_code in (401, 403):
                    return pd.DataFrame()
                if resp.status_code == 429 and attempt < 5:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                break
            except requests.RequestException:
                if attempt >= 5:
                    return pd.DataFrame()
                time.sleep(2 ** attempt)
        data = resp.json()
        prices = data.get('prices', [])
        vols = data.get('total_volumes', [])
        if not prices:
            return pd.DataFrame()
        df_price = pd.DataFrame(prices, columns=['ts', 'price'])
        df_price['timestamp'] = pd.to_datetime(df_price['ts'], unit='ms', utc=True)
        df_price = df_price[['timestamp', 'price']]
        df_vol = pd.DataFrame(vols, columns=['ts', 'volume']) if vols else pd.DataFrame(columns=['timestamp', 'volume'])
        if not df_vol.empty:
            df_vol['timestamp'] = pd.to_datetime(df_vol['ts'], unit='ms', utc=True)
            df_vol = df_vol[['timestamp', 'volume']]
            df = pd.merge(df_price, df_vol, on='timestamp', how='outer')
        else:
            df = df_price
        df = ensure_timestamp_index(df)
        # resample to target frequency if specified
        freq = self.settings.frequency.upper()
        if freq.endswith('H') or freq.endswith('D'):
            df = df.resample(freq).last().dropna(how='all')
        return df


