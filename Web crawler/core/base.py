from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

import pandas as pd

from .storage import read_timeseries_csv, write_timeseries_csv, merge_timeseries
from .utils import utc_now, ensure_timestamp_index, iter_time_chunks, parse_timestamp


@dataclass
class ProviderSettings:
    frequency: str = '1H'  # pandas offset alias, e.g., '1H', '1D'
    chunk_hours: int = 720  # default chunk size in hours (30 days)
    chunk_days: int = 30
    start_date: Optional[str] = None  # ISO date string or None
    max_backfill_days: Optional[int] = None
    rate_limit_sleep: float = 0.0  # seconds to sleep between chunk calls
    backfill_gaps: bool = True


class ProviderBase:
    def __init__(self, name: str, params: Dict[str, Any], output_path: str):
        self.name = name
        self.params = params or {}
        self.output_path = output_path

        self.settings = ProviderSettings(
            frequency=self.params.get('frequency', '1H'),
            chunk_hours=int(self.params.get('chunk_hours', 720)),
            chunk_days=int(self.params.get('chunk_days', 30)),
            start_date=self.params.get('start_date'),
            max_backfill_days=self.params.get('max_backfill_days'),
            rate_limit_sleep=float(self.params.get('rate_limit_sleep', 0.0)),
            backfill_gaps=bool(self.params.get('backfill_gaps', True)),
        )

    # ---- Methods to override in subclasses ----
    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        raise NotImplementedError

    # ---- Optional overrides ----
    def initial_start(self) -> Optional[datetime]:
        if self.settings.start_date:
            return parse_timestamp(self.settings.start_date)
        return None

    # ---- Core update logic ----
    def update(self) -> None:
        existing = read_timeseries_csv(self.output_path)
        now = utc_now()

        # Determine start
        start: Optional[datetime]
        if existing is not None and not existing.empty:
            last_ts = existing.index.max().to_pydatetime()
            start = last_ts
        else:
            start = self.initial_start()
            if start is None:
                # default: backfill last 90 days
                start = now - timedelta(days=int(self.params.get('default_lookback_days', 90)))

        # Apply max_backfill_days cap if provided
        if self.settings.max_backfill_days is not None:
            cap_start = now - timedelta(days=int(self.settings.max_backfill_days))
            if start < cap_start:
                start = cap_start

        # Build chunks
        freq = self.settings.frequency.upper()
        if 'H' in freq:
            step = timedelta(hours=self.settings.chunk_hours)
        elif 'D' in freq or 'W' in freq or 'M' in freq or 'Y' in freq:
            step = timedelta(days=self.settings.chunk_days)
        else:
            step = timedelta(days=self.settings.chunk_days)

        frames: List[pd.DataFrame] = []
        for chunk_start, chunk_end in iter_time_chunks(start, now, step):
            df = self.fetch_chunk(chunk_start, chunk_end)
            if df is None or df.empty:
                continue
            df = ensure_timestamp_index(df)
            frames.append(df)
            if self.settings.rate_limit_sleep > 0:
                import time
                time.sleep(self.settings.rate_limit_sleep)

        if frames:
            new_data = pd.concat(frames).sort_index()
            new_data = new_data[~new_data.index.duplicated(keep='last')]
        else:
            new_data = pd.DataFrame()

        merged = merge_timeseries(existing, new_data)

        # Attempt to backfill missing intervals within the historical range
        if self.settings.backfill_gaps and merged is not None and not merged.empty:
            try:
                from datetime import timedelta as _td
                freq = self.settings.frequency
                # Build expected index from earliest to latest
                expected_index = pd.date_range(start=merged.index.min(), end=merged.index.max(), freq=str(freq).lower())
                missing_index = expected_index.difference(merged.index)
                if len(missing_index) > 0:
                    # group contiguous missing spans
                    missing_times = sorted(missing_index.to_pydatetime())
                    spans = []
                    if missing_times:
                        current_start = missing_times[0]
                        prev = missing_times[0]
                        # Safe spacing heuristic
                        freq_upper = str(freq).upper()
                        if 'H' in freq_upper:
                            delta = _td(hours=1)
                        elif 'D' in freq_upper:
                            delta = _td(days=1)
                        elif 'W' in freq_upper:
                            delta = _td(days=7)
                        else:
                            delta = _td(days=1)
                        for t in missing_times[1:]:
                            if (t - prev) > delta:
                                spans.append((current_start, prev + delta))
                                current_start = t
                            prev = t
                        spans.append((current_start, prev + delta))

                    frames: List[pd.DataFrame] = []
                    for s, e in spans:
                        # Respect max_backfill_days cap
                        if self.settings.max_backfill_days is not None:
                            if (now - s).days > int(self.settings.max_backfill_days):
                                s = now - timedelta(days=int(self.settings.max_backfill_days))
                        if s >= e:
                            continue
                        try:
                            gap_df = self.fetch_chunk(s, e)
                        except Exception:
                            gap_df = None
                        if gap_df is not None and not gap_df.empty:
                            frames.append(ensure_timestamp_index(gap_df))
                    if frames:
                        gap_data = pd.concat(frames).sort_index()
                        gap_data = gap_data[~gap_data.index.duplicated(keep='last')]
                        merged = merge_timeseries(merged, gap_data)
            except Exception:
                # best-effort gap fill; ignore errors
                pass

        write_timeseries_csv(self.output_path, merged)


