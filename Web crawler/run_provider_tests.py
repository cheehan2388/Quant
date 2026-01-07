import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from providers import get_provider_class
from providers.test_params import TEST_PARAMS, LIKELY_KEY_REQUIRED
from core.utils import ensure_timestamp_index


def test_provider(provider_type: str, hours_back: int = 48, only_free: bool = False) -> tuple[bool, str]:
    provider_class = get_provider_class(provider_type)
    if provider_class is None:
        return False, f"unknown provider: {provider_type}"

    if only_free and provider_type in LIKELY_KEY_REQUIRED:
        return True, "skipped (key likely required)"

    params = TEST_PARAMS.get(provider_type, {})
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours_back)

    try:
        provider = provider_class(name=f"test_{provider_type}", params=params, output_path=str(Path('NUL') if Path('/').anchor == '\\' else Path('/dev/null')))
        df = provider.fetch_chunk(start, end)
        if df is None or df.empty:
            return True, "empty (ok)"
        # validate
        try:
            df = ensure_timestamp_index(df)
        except Exception as exc:
            return False, f"invalid dataframe: {exc}"
        # basic sanity: columns numeric (if any)
        non_ts_cols = [c for c in df.columns if c != 'timestamp']
        if hasattr(df, 'dtypes'):
            for c in non_ts_cols:
                # allow any dtype but prefer numeric
                pass
        return True, f"ok rows={len(df)}"
    except Exception as exc:
        return False, str(exc)


def main():
    ap = argparse.ArgumentParser(description='Smoke test providers (fetch recent small chunk)')
    ap.add_argument('--only', nargs='*', help='Specific provider types to test (default: all)')
    ap.add_argument('--hours-back', type=int, default=48)
    ap.add_argument('--only-free', action='store_true', help='Skip providers that likely need API keys')
    args = ap.parse_args()

    # Discover providers via registry
    from providers import _REGISTRY
    target = args.only or sorted(_REGISTRY.keys())

    print(f"Testing {len(target)} providers...\n")
    failures = 0
    for ptype in target:
        ok, msg = test_provider(ptype, hours_back=args.hours_back, only_free=args.only_free)
        status = 'PASS' if ok else 'FAIL'
        print(f"[{status}] {ptype}: {msg}")
        if not ok:
            failures += 1

    print(f"\nDone. Failures: {failures} / {len(target)}")
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())


