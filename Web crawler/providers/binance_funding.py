from __future__ import annotations
from datetime import datetime, timezone
from typing import List, Dict, Any
import time
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

from core.base import ProviderBase
from core.utils import ensure_timestamp_index, to_utc_timestamp

class BinanceFundingRate(ProviderBase):
    BASE_URL = "https://fapi.binance.com/fapi/v1/fundingRate"

    def _session(self) -> requests.Session:
        s = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        s.mount("https://", HTTPAdapter(max_retries=retries))
        return s

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        if end <= start:
            return pd.DataFrame()

        symbol = self.params.get("symbol", "BTCUSDT")
        limit = int(self.params.get("limit", 1000))
        delay = float(self.params.get("rate_delay_sec", 0.0))
        max_pages = int(self.params.get("max_pages", 200))

        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        sess = self._session()
        all_rows: List[Dict[str, Any]] = []
        cursor = start_ms

        for _ in range(max_pages):
            params = {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": limit}
            resp = sess.get(self.BASE_URL, params=params, timeout=30)
            if resp.status_code != 200:
                break
            page = resp.json()
            if not page:
                break

            all_rows.extend(page)
            next_cursor = int(page[-1].get("fundingTime", cursor)) + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor

            if len(page) < limit:
                break
            if delay > 0:
                time.sleep(delay)

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        if "fundingTime" not in df.columns:
            return pd.DataFrame()

        df = df.rename(columns={"fundingTime": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")

        start_utc = to_utc_timestamp(start)
        end_utc = to_utc_timestamp(end)
        df = df[(df["timestamp"] >= start_utc) & (df["timestamp"] < end_utc)]
        df = df[["timestamp", "fundingRate"]].drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        return ensure_timestamp_index(df)
