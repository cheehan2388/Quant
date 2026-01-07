from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import sys

import pandas as pd

# Support running as a script or as a module
try:
    from ..core.storage import read_timeseries_csv, write_timeseries_csv  # type: ignore
    from ..core.utils import ensure_timestamp_index  # type: ignore
    from ..providers import get_provider_class  # type: ignore
except Exception:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from core.storage import read_timeseries_csv, write_timeseries_csv
    from core.utils import ensure_timestamp_index
    from providers import get_provider_class


def pick_reference_price(data_dir: Path) -> tuple[str, pd.DataFrame]:
    """Pick a reference USD spot series (no key) in order of preference."""
    candidates = [
        ("binance_btcusdt_1h.csv", "binance"),
        ("kraken_xbt_usd_1h.csv", "kraken"),
        ("bitfinex_btcusd_1h.csv", "bitfinex"),
        ("bitstamp_btcusd_1h.csv", "bitstamp"),
    ]
    for filename, name in candidates:
        df = read_timeseries_csv(data_dir / filename)
        if df is not None and not df.empty:
            return name, df
    raise FileNotFoundError("No reference series found (binance/kraken/bitfinex/bitstamp 1h). Run crawler first.")


def compute_premium_ohlc(coinbase: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Compute Coinbase premium OHLC: (CB - REF) / REF for each of O/H/L/C."""
    cb = ensure_timestamp_index(coinbase)
    rf = ensure_timestamp_index(ref)

    # Align on intersection of timestamps
    cols_needed = ['open', 'high', 'low', 'close']
    for df in (cb, rf):
        for c in cols_needed:
            if c not in df.columns:
                raise ValueError(f"Missing column '{c}' in input data")

    merged = cb[cols_needed].join(rf[cols_needed], how='inner', lsuffix='_cb', rsuffix='_ref')
    if merged.empty:
        return pd.DataFrame()

    premium = pd.DataFrame(index=merged.index)
    for c in cols_needed:
        premium[f'{c}'] = (merged[f'{c}_cb'] - merged[f'{c}_ref']) / merged[f'{c}_ref']

    premium = premium.sort_index()
    premium = premium[~premium.index.duplicated(keep='last')]
    premium = premium.rename_axis('timestamp').reset_index().set_index('timestamp')
    return premium


def fetch_ohlc(provider_type: str, params: dict, start: pd.Timestamp, end: pd.Timestamp, chunk_days: int = 7, rps_limit: float | None = None) -> pd.DataFrame:
    """Fetch OHLC from a provider in chunks without writing CSVs."""
    provider_class = get_provider_class(provider_type)
    if provider_class is None:
        raise ValueError(f"Unknown provider type: {provider_type}")
    fetcher = provider_class(name=f"fetch_{provider_type}", params=params, output_path="/dev/null")

    frames: list[pd.DataFrame] = []
    cursor = start
    step = timedelta(days=chunk_days)
    import time
    sleep_s = (1.0 / rps_limit) if rps_limit and rps_limit > 0 else 0.0
    while cursor < end:
        window_end = min(cursor + step, end)
        try:
            df = fetcher.fetch_chunk(cursor.to_pydatetime(), window_end.to_pydatetime())
        except Exception:
            df = pd.DataFrame()
        if df is not None and not df.empty:
            df = ensure_timestamp_index(df)
            frames.append(df[['open', 'high', 'low', 'close']].copy() if all(c in df.columns for c in ['open', 'high', 'low', 'close']) else df.copy())
        cursor = window_end
        if sleep_s > 0:
            time.sleep(sleep_s)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep='last')]
    return out


def main():
    ap = argparse.ArgumentParser(description='Compute Coinbase Premium Index OHLC from 1h spot data')
    ap.add_argument('--start', default='2020-01-01', help='ISO start date (UTC)')
    ap.add_argument('--end', default='2025-08-16', help='ISO end date (UTC)')
    ap.add_argument('--output', default='coinbase_premium_btcusd_1h.csv')
    ap.add_argument('--autofetch', action='store_true', help='Fetch inputs from public APIs (no CSVs)')
    ap.add_argument('--ref', default='binance', choices=['binance', 'kraken', 'bitfinex', 'bitstamp'], help='Reference venue')
    ap.add_argument('--cb-chunk-days', type=int, default=7, help='Coinbase fetch chunk size in days (candles limit ~12 days)')
    ap.add_argument('--ref-chunk-days', type=int, default=30, help='Reference fetch chunk size in days')
    ap.add_argument('--rps', type=float, default=1.0, help='Requests per second limit between chunks')
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    # Always fetch from providers (no CSV dependency)
    start = pd.to_datetime(args.start, utc=True)
    end = pd.to_datetime(args.end, utc=True)

    cb_df = fetch_ohlc(
        provider_type='coinbase_candles',
        params={'product_id': 'BTC-USD', 'granularity': 3600, 'frequency': '1H'},
        start=start,
        end=end,
        chunk_days=int(args.cb_chunk_days),
        rps_limit=float(args.rps),
    )

    ref_name = args.ref
    if args.ref == 'binance':
        ref_df = fetch_ohlc('binance_klines', {'symbol': 'BTCUSDT', 'interval': '1h', 'frequency': '1H'}, start, end, chunk_days=int(args.ref_chunk_days), rps_limit=float(args.rps))
    elif args.ref == 'kraken':
        ref_df = fetch_ohlc('kraken_ohlc', {'pair': 'XBTUSD', 'interval_minutes': 60, 'frequency': '1H'}, start, end, chunk_days=int(args.ref_chunk_days), rps_limit=float(args.rps))
    elif args.ref == 'bitfinex':
        ref_df = fetch_ohlc('bitfinex_candles', {'symbol': 'tBTCUSD', 'timeframe': '1h', 'frequency': '1H'}, start, end, chunk_days=int(args.ref_chunk_days), rps_limit=float(args.rps))
    elif args.ref == 'bitstamp':
        ref_df = fetch_ohlc('bitstamp_ohlc', {'pair': 'btcusd', 'step': 3600, 'frequency': '1H'}, start, end, chunk_days=int(args.ref_chunk_days), rps_limit=float(args.rps))
    else:
        # default to binance
        ref_df = fetch_ohlc('binance_klines', {'symbol': 'BTCUSDT', 'interval': '1h', 'frequency': '1H'}, start, end, chunk_days=int(args.ref_chunk_days), rps_limit=float(args.rps))

    out_df = compute_premium_ohlc(cb_df, ref_df)
    if out_df is None or out_df.empty:
        print("[WARN] No overlapping data to compute premium.")
        return 0

    out_path = data_dir / args.output
    write_timeseries_csv(out_path, out_df)
    print(f"[OK] Wrote premium OHLC to {out_path} using reference={ref_name}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


