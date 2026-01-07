from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .utils import ensure_timestamp_index


def read_timeseries_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(file_path)
    if df.empty:
        return df
    if 'timestamp' not in df.columns:
        # Attempt to recover from common alternatives
        for alt in ['time', 'date', 'datetime']:
            if alt in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df[alt], utc=True)
                except Exception:
                    pass
                break
        # If still missing, treat as no usable history (legacy/corrupt file)
        if 'timestamp' not in df.columns:
            return pd.DataFrame()
    return ensure_timestamp_index(df)


def write_timeseries_csv(path: str | Path, df: pd.DataFrame) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        # create file with headers for consistency
        pd.DataFrame(columns=['timestamp']).to_csv(file_path, index=False)
        return
    df_to_write = df.reset_index()
    df_to_write.to_csv(file_path, index=False)


def merge_timeseries(existing: pd.DataFrame, new: pd.DataFrame, how: str = 'outer') -> pd.DataFrame:
    if existing is None or existing.empty:
        return new
    if new is None or new.empty:
        return existing
    # Outer join to preserve full history
    merged = existing.combine_first(new)
    # Prefer new values where overlaps exist
    overlapping = existing.index.intersection(new.index)
    if len(overlapping) > 0:
        merged.loc[overlapping] = new.loc[overlapping]
    merged = merged.sort_index()
    merged = merged[~merged.index.duplicated(keep='last')]
    return merged


