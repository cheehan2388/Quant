from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Iterable

import pandas as pd


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_timestamp(value) -> datetime:
    if isinstance(value, datetime):
        return to_utc(value)
    return pd.to_datetime(value, utc=True).to_pydatetime()


def floor_dt(dt: datetime, freq: str) -> datetime:
    # pandas-based floor for convenience
    return pd.Timestamp(dt).floor(freq, ambiguous=False).to_pydatetime()


def ceil_dt(dt: datetime, freq: str) -> datetime:
    return pd.Timestamp(dt).ceil(freq, ambiguous=False).to_pydatetime()


def iter_time_chunks(start: datetime, end: datetime, step: timedelta) -> Iterable[tuple[datetime, datetime]]:
    cursor = start
    while cursor < end:
        nxt = min(cursor + step, end)
        yield cursor, nxt
        cursor = nxt


def ensure_timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    # Accept either a 'timestamp' column or a DatetimeIndex
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], utc=True)
    elif isinstance(df.index, pd.DatetimeIndex):
        ts = pd.to_datetime(df.index, utc=True)
    else:
        raise ValueError("DataFrame must contain 'timestamp' column or a DatetimeIndex")
    df['timestamp'] = ts
    df = df.set_index('timestamp')
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='last')]
    return df


def to_utc_timestamp(value) -> pd.Timestamp:
    """Return a tz-aware pandas Timestamp in UTC for any datetime-like input."""
    ts = pd.Timestamp(value)
    if ts.tz is None:
        return ts.tz_localize('UTC')
    return ts.tz_convert('UTC')


