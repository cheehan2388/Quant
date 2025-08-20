from __future__ import annotations

from datetime import datetime

import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index, to_utc_timestamp


class CoinbaseBestBidAsk(ProviderBase):
    URL = 'https://api.exchange.coinbase.com/products/{product_id}/ticker'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        # Ticker endpoint returns latest; we snapshot at end timestamp only
        product_id = self.params.get('product_id', 'BTC-USD')
        url = self.URL.format(product_id=product_id)
        resp = requests.get(url, timeout=15, headers={'User-Agent': 'crawler'})
        resp.raise_for_status()
        d = resp.json()
        ts = to_utc_timestamp(pd.Timestamp.utcnow())
        df = pd.DataFrame([{
            'timestamp': ts,
            'bid': float(d.get('bid', 'nan')),
            'ask': float(d.get('ask', 'nan')),
            'price': float(d.get('price', 'nan')),
            'volume': float(d.get('volume', 'nan')),
        }])
        return ensure_timestamp_index(df)


