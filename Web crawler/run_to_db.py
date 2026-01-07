import argparse
from pathlib import Path
import yaml

from core.utils import utc_now
from providers import get_provider_class
from core.db_sink import ingest_csv_to_db


def load_config(config_path: Path):
    with config_path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_and_load(
    config_path: Path,
    database_url: str,
    *,
    skip_run: bool = False,
    only_names: list[str] | None = None,
    only_types: list[str] | None = None,
    csv_path: str | None = None,
    dataset_name: str | None = None,
    provider_type: str | None = None,
    frequency: str | None = None,
    all_csvs: bool = False,
):
    """Run crawlers (optional) then ingest CSVs into the database.

    Supports selecting specific providers or ingesting a single CSV directly.
    """
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / 'data'

    # Single CSV ingestion mode (bypass config providers)
    if csv_path:
        if not (dataset_name and provider_type):
            raise SystemExit("--csv requires --dataset-name and --provider-type (and optional --frequency)")
        try:
            count = ingest_csv_to_db(database_url, dataset_name, provider_type, frequency, csv_path, extra={})
            print(f"[DB] {dataset_name}: inserted {count} rows from CSV {csv_path}")
        except Exception as exc:
            print(f"[DB][ERROR] {dataset_name} failed: {exc}")
        return

    # Ingest ALL CSV files in data directory (bypass per-provider selection)
    if all_csvs:
        # Optional: use config to enrich metadata (name, type, frequency)
        provider_by_output = {}
        try:
            from run import load_config as _load
            cfg = _load(config_path)
            for p in cfg.get('providers', []) or []:
                out = p.get('output') or f"{p.get('type')}.csv"
                provider_by_output[out] = {
                    'name': p.get('name', p.get('type')),
                    'type': p.get('type'),
                    'frequency': (p.get('params') or {}).get('frequency'),
                    'extra': {'params': p.get('params', {})},
                }
        except Exception:
            pass

        # Helper to guess frequency from filename when not found in config
        import re
        def _guess_freq(stem: str) -> str | None:
            # Looks for tokens like 1h, 4h, 1d, 7d, 15m, 1w, 1y near the end
            m = re.search(r'(\d+)([smhdwMy])$', stem)
            if m:
                num, unit = m.groups()
                unit = unit.lower()
                if unit == 's':
                    return f"{num}S"
                if unit == 'm':
                    return f"{num}T"  # pandas minute alias T
                if unit == 'h':
                    return f"{num}H"
                if unit == 'd':
                    return f"{num}D"
                if unit == 'w':
                    return f"{num}W"
                if unit in ('M', 'y'):
                    return f"{num}{unit.upper()}"
            # Also allow pattern _1h or _1d appearing anywhere
            m = re.search(r'_(\d+)([smhdwMy])_', f"_{stem}_")
            if m:
                num, unit = m.groups()
                mapping = {'s': 'S', 'm': 'T', 'h': 'H', 'd': 'D', 'w': 'W', 'M': 'M', 'y': 'Y'}
                return f"{num}{mapping.get(unit, unit.upper())}"
            return None

        ingested = 0
        for csv_file in sorted(data_dir.glob('*.csv')):
            out_name = csv_file.name
            meta = provider_by_output.get(out_name)
            if meta:
                ds_name = meta['name']
                ptype = meta['type']
                freq = meta['frequency']
                extra = meta['extra']
            else:
                ds_name = csv_file.stem
                ptype = 'csv_import'
                freq = _guess_freq(csv_file.stem)
                extra = {'source': 'csv_scan'}
            try:
                count = ingest_csv_to_db(database_url, ds_name, ptype, freq, str(csv_file), extra)
                if count == 0:
                    print(f"[DB] {ds_name}: no clean rows to insert (skipped) [{out_name}]")
                else:
                    print(f"[DB] {ds_name}: inserted {count} rows [{out_name}]")
                    ingested += count
            except Exception as exc:
                print(f"[DB][ERROR] {ds_name} failed: {exc} [{out_name}]")
        print(f"[DB] Total inserted: {ingested}")
        return

    # Config-driven mode
    from run import run_once, load_config as _load
    cfg = _load(config_path)

    if not skip_run:
        # Run once to update CSVs
        run_once(cfg)

    # Filter providers by name/type if requested
    providers = list(cfg.get('providers', []) or [])
    if only_names:
        name_set = {n.strip() for n in only_names}
        providers = [p for p in providers if p.get('name', p.get('type')) in name_set]
    if only_types:
        type_set = {t.strip() for t in only_types}
        providers = [p for p in providers if p.get('type') in type_set]

    ingested = 0
    for provider_cfg in providers:
        if not provider_cfg.get('enabled', True):
            continue
        ptype = provider_cfg['type']
        name = provider_cfg.get('name', ptype)
        output_file = provider_cfg.get('output', f"{ptype}.csv")
        csv_file = str(data_dir / output_file)
        freq = provider_cfg.get('params', {}).get('frequency')
        extra = {'params': provider_cfg.get('params', {})}
        try:
            count = ingest_csv_to_db(database_url, name, ptype, freq, csv_file, extra)
            if count == 0:
                print(f"[DB] {name}: no clean rows to insert (skipped)")
            else:
                print(f"[DB] {name}: inserted {count} rows")
                ingested += count
        except Exception as exc:
            print(f"[DB][ERROR] {name} failed: {exc}")
    print(f"[DB] Total inserted: {ingested}")


def main():
    ap = argparse.ArgumentParser(description='Run crawlers and/or load CSVs into Postgres')
    ap.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    ap.add_argument('--db', required=True, help='Database URL, e.g., postgresql+psycopg2://user:pass@host:5432/dbname')
    ap.add_argument('--skip-run', action='store_true', help='Skip running crawlers; only load existing CSVs')
    ap.add_argument('--only-names', nargs='*', help='Only load providers with these names (matches config "name")')
    ap.add_argument('--only-types', nargs='*', help='Only load providers with these types (matches config "type")')
    # Single CSV mode
    ap.add_argument('--csv', help='Path to a CSV to load directly (bypasses config providers)')
    ap.add_argument('--dataset-name', help='Dataset name for single-CSV mode')
    ap.add_argument('--provider-type', help='Provider type label for single-CSV mode')
    ap.add_argument('--frequency', help='Frequency label for single-CSV mode (e.g., 1H, 1D)')
    # All CSVs in data directory
    ap.add_argument('--all-csvs', action='store_true', help='Ingest all CSV files under data/ (skips running crawlers)')
    args = ap.parse_args()

    run_and_load(
        Path(args.config),
        args.db,
        skip_run=args.skip_run or bool(args.csv) or args.all_csvs,
        only_names=args.only_names,
        only_types=args.only_types,
        csv_path=args.csv,
        dataset_name=args.dataset_name,
        provider_type=args.provider_type,
        frequency=args.frequency,
        all_csvs=args.all_csvs,
    )


if __name__ == '__main__':
    main()


