from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd

from sqlalchemy import select

from db import create_session
from db.schema import Dataset, TimeseriesPoint
from .storage import read_timeseries_csv
from .utils import utc_now


def ensure_dataset(session, name: str, provider_type: str, frequency: str | None, extra: Dict[str, Any] | None) -> int:
    ds = session.execute(select(Dataset).where(Dataset.name == name)).scalar_one_or_none()
    if ds is None:
        ds = Dataset(name=name, provider_type=provider_type, frequency=frequency, extra=extra)
        session.add(ds)
        session.flush()
    return ds.id


def df_to_timeseries_points(dataset_id: int, df: pd.DataFrame) -> list[Dict[str, Any]]:
    # df must be indexed by timestamp, columns = any numeric/values
    records: list[Dict[str, Any]] = []
    if df is None or df.empty:
        return records
    for ts, row in df.iterrows():
        values = {}
        for col, val in row.items():
            if pd.isna(val):
                continue
            # Basic JSON serializable casting
            if isinstance(val, (int, float, str, bool)):
                values[col] = val
            else:
                try:
                    values[col] = float(val)
                except Exception:
                    continue
        if not values:
            continue
        records.append({'dataset_id': dataset_id, 'timestamp': ts.to_pydatetime(), 'values': values, 'is_final': True})
    return records


def _clean_timeseries(df: pd.DataFrame, *, frequency: str | None, missing_row_threshold: float = 0.3, forward_fill_limit: int = 1) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    cleaned = df.copy()
    # Ensure sorted, unique index
    if not isinstance(cleaned.index, pd.DatetimeIndex):
        # caller is expected to pass time-indexed df; guard regardless
        if 'timestamp' in cleaned.columns:
            cleaned['timestamp'] = pd.to_datetime(cleaned['timestamp'], utc=True)
            cleaned = cleaned.set_index('timestamp')
        else:
            return pd.DataFrame()
    cleaned = cleaned.sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep='last')]

    # Drop rows in the future beyond a small tolerance
    now = utc_now()
    cleaned = cleaned[cleaned.index <= (now)]

    # Replace non-finite values with NaN
    cleaned = cleaned.replace([float('inf'), float('-inf')], pd.NA)

    # If a target frequency is provided, align to it to surface gaps
    if frequency:
        try:
            aligned_index = pd.date_range(start=cleaned.index.min(), end=cleaned.index.max(), freq=frequency)
            cleaned = cleaned.reindex(aligned_index)
        except Exception:
            # If frequency is invalid, skip alignment
            pass

    # Forward-fill small gaps per column to backfill missing data
    if forward_fill_limit and forward_fill_limit > 0:
        cleaned = cleaned.ffill(limit=forward_fill_limit)

    # Drop rows where more than the threshold of columns are missing
    if cleaned.shape[1] > 0:
        frac_missing = cleaned.isna().mean(axis=1)
        cleaned = cleaned[frac_missing <= missing_row_threshold]

    # Finally, drop rows that are entirely NaN
    cleaned = cleaned.dropna(how='all')
    return cleaned


def ingest_csv_to_db(database_url: str, dataset_name: str, provider_type: str, frequency: str | None, csv_path: str, extra: Dict[str, Any] | None = None) -> int:
    session = create_session(database_url)
    try:
        raw_df = read_timeseries_csv(csv_path)
        df = _clean_timeseries(raw_df, frequency=frequency, missing_row_threshold=0.7, forward_fill_limit=1)
        dataset_id = ensure_dataset(session, dataset_name, provider_type, frequency, extra)
        records = df_to_timeseries_points(dataset_id, df)
        if not records:
            return 0
        session.bulk_insert_mappings(TimeseriesPoint, records)
        session.commit()
        return len(records)
    finally:
        session.close()


