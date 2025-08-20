from __future__ import annotations

from datetime import datetime
from typing import Dict, Any

import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index


_INTERVAL_MAP = {
    '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
    '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
    '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M',
}


class BinanceKlines(ProviderBase):
    BASE_URL = 'https://api.binance.com/api/v3/klines'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        symbol = self.params.get('symbol', 'BTCUSDT')
        interval = _INTERVAL_MAP.get(self.params.get('interval', '1h'), '1h')
        # Binance returns up to 1000 klines per call
        params: Dict[str, Any] = {
            'symbol': symbol,
            'interval': interval,
            'startTime': int(start.timestamp() * 1000),
            'endTime': int(end.timestamp() * 1000),
            'limit': 1000,
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # If Binance returns an error object like {code: -1121, msg: '...'} guard against it
        if isinstance(data, dict):
            return pd.DataFrame()
        if not data:
            return pd.DataFrame()
        cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume', 'ignore']
        # Sometimes Binance returns fewer columns; build robustly
        try:
            df = pd.DataFrame(data, columns=cols)
        except Exception:
            df = pd.DataFrame(data)
            # Map positional columns if present
            rename_map = {0: 'open_time', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'}
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # If no open_time column could be created, skip this chunk
        if 'open_time' not in df.columns:
            return pd.DataFrame()

        # Build timestamp safely
        try:
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        except Exception:
            return pd.DataFrame()

        # Keep only expected columns if present
        keep_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        present = [c for c in keep_cols if c in df.columns]
        if 'timestamp' not in present:
            return pd.DataFrame()
        df = df[present]
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return ensure_timestamp_index(df)


