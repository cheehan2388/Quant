from __future__ import annotations

from datetime import datetime

import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index


class YahooPrice(ProviderBase):
    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        try:
            import yfinance as yf
        except Exception as exc:
            raise RuntimeError("yfinance is required for YahooPrice") from exc

        ticker = self.params.get('ticker', 'SPY')
        interval = self.params.get('interval', '1d')
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'
        })
        df['timestamp'] = pd.to_datetime(df.index, utc=True)
        df = df.reset_index(drop=True)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]
        return ensure_timestamp_index(df)


