import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import yaml
import re


from core.utils import utc_now
from providers import get_provider_class


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    # Optionally merge external bulk.yaml to simplify main config maintenance
    bulk_path = config_path.with_name('bulk.yaml')
    try:
        if bulk_path.exists():
            with bulk_path.open('r', encoding='utf-8') as bf:
                bcfg = yaml.safe_load(bf) or {}
            # accept either top-level under 'bulk' or raw bulk mapping
            ext_bulk = bcfg.get('bulk', bcfg)
            if isinstance(ext_bulk, dict) and ext_bulk:
                cfg['bulk'] = ext_bulk
    except Exception:
        # non-fatal
        pass
    return cfg


def _slugify(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "item"


def expand_bulk(config: Dict[str, Any]) -> list[Dict[str, Any]]:
    bulk_cfg = config.get('bulk', {}) or {}
    providers: list[Dict[str, Any]] = []

    # Wikipedia pageviews bulk
    wiki = bulk_cfg.get('wikipedia_pageviews')
    if wiki:
        enabled = bool(wiki.get('enabled', True))
        base_params = wiki.get('base_params', {
            'project': 'en.wikipedia',
            'access': 'all-access',
            'agent': 'all-agents',
            'granularity': 'daily',
            'frequency': '1D',
        })
        output_tpl = wiki.get('output_tpl', 'wikipedia_{slug}_daily.csv')
        for article in wiki.get('articles', []):
            slug = _slugify(article)
            providers.append({
                'type': 'wikipedia_pageviews',
                'name': f'wikipedia_{slug}_daily',
                'enabled': enabled,
                'output': output_tpl.format(slug=slug),
                'params': {**base_params, 'article': article},
            })

    # Open-Meteo hourly bulk
    meteo = bulk_cfg.get('open_meteo_hourly')
    if meteo:
        enabled = bool(meteo.get('enabled', True))
        base_params = meteo.get('base_params', {'hourly': 'temperature_2m', 'frequency': '1H'})
        output_tpl = meteo.get('output_tpl', 'weather_{name}_1h.csv')
        for loc in meteo.get('locations', []):
            name = _slugify(loc.get('name'))
            lat = float(loc.get('latitude') or loc.get('lat'))
            lon = float(loc.get('longitude') or loc.get('lon'))
            params = {**base_params, 'latitude': lat, 'longitude': lon}
            providers.append({
                'type': 'open_meteo_hourly',
                'name': f'weather_{name}_1h',
                'enabled': enabled,
                'output': output_tpl.format(name=name),
                'params': params,
            })

    # ExchangeRate Host bulk (cross product of bases and symbols)
    fx = bulk_cfg.get('exchangerate_host_timeseries')
    if fx:
        enabled = bool(fx.get('enabled', True))
        base_params = fx.get('base_params', {'frequency': '1D'})
        output_tpl = fx.get('output_tpl', 'fx_{base}_{sym}_1d.csv')
        bases = fx.get('base_currencies', [])
        symbols = fx.get('symbols', [])
        for base in bases:
            for sym in symbols:
                if str(sym).upper() == str(base).upper():
                    continue
                name = f"fx_{str(base).lower()}_{str(sym).lower()}_1d"
                params = {**base_params, 'base': base, 'symbols': sym}
                providers.append({
                    'type': 'exchangerate_host_timeseries',
                    'name': name,
                    'enabled': enabled,
                    'output': output_tpl.format(base=str(base).lower(), sym=str(sym).lower()),
                    'params': params,
                })

    # ECB FX daily bulk
    ecb = bulk_cfg.get('ecb_fx_daily')
    if ecb:
        enabled = bool(ecb.get('enabled', True))
        base_params = ecb.get('base_params', {'frequency': '1D'})
        output_tpl = ecb.get('output_tpl', 'ecb_eur_{sym}_1d.csv')
        for sym in ecb.get('symbols', []):
            name = f"ecb_eur_{str(sym).lower()}_1d"
            params = {**base_params}
            providers.append({
                'type': 'ecb_fx_daily',
                'name': name,
                'enabled': enabled,
                'output': output_tpl.format(sym=str(sym).lower()),
                'params': {'symbols': sym, **params},
            })

    # World Bank bulk (cross product countries x indicators)
    wb = bulk_cfg.get('worldbank_indicator')
    if wb:
        enabled = bool(wb.get('enabled', True))
        base_params = wb.get('base_params', {'frequency': '1Y', 'chunk_days': 3650})
        output_tpl = wb.get('output_tpl', 'worldbank_{country}_{indicator}.csv')
        countries = wb.get('countries', [])
        indicators = wb.get('indicators', [])
        for country in countries:
            for indicator in indicators:
                cname = _slugify(country)
                iname = _slugify(indicator)
                name = f"worldbank_{cname}_{iname}"
                params = {**base_params, 'country': country, 'indicator': indicator}
                providers.append({
                    'type': 'worldbank_indicator',
                    'name': name,
                    'enabled': enabled,
                    'output': output_tpl.format(country=cname, indicator=iname),
                    'params': params,
                })

    return providers


def run_once(config: Dict[str, Any]) -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    providers = list(config.get('providers', []) or [])
    # Expand any bulk sections into concrete provider entries
    providers.extend(expand_bulk(config))
    default_params: Dict[str, Any] = config.get('defaults', {}) or {}
    for provider_cfg in providers:
        if not provider_cfg.get('enabled', True):
            continue

        provider_type = provider_cfg['type']
        provider_name = provider_cfg.get('name', provider_type)
        output_file = provider_cfg.get('output', f"{provider_type}.csv")

        provider_class = get_provider_class(provider_type)
        if provider_class is None:
            print(f"[WARN] Unknown provider type: {provider_type}")
            continue

        try:
            merged_params = {**default_params, **provider_cfg.get('params', {})}
            provider = provider_class(
                name=provider_name,
                params=merged_params,
                output_path=str(data_dir / output_file),
            )
            print(f"[INFO] Updating {provider_name} at {utc_now().isoformat()}...")
            provider.update()
            print(f"[INFO] Done {provider_name} -> {output_file}")
        except Exception as exc:
            print(f"[ERROR] Provider {provider_name} failed: {exc}")


def run_schedule(config: Dict[str, Any]) -> None:
    # Lightweight scheduler: loop forever with sleep interval per run block.
    # For finer control, use APScheduler in a later iteration.
    import time
    interval_seconds = int(config.get('schedule', {}).get('interval_seconds', 6 * 3600))
    if interval_seconds < 60:
        interval_seconds = 60
    print(f"[INFO] Starting loop schedule, interval={interval_seconds}s")
    while True:
        try:
            run_once(config)
        except Exception as exc:
            print(f"[ERROR] Scheduled run failed: {exc}")
        time.sleep(interval_seconds)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Multi-source Data Crawler')
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--once', action='store_true', help='Run all enabled providers once and exit')
    group.add_argument('--schedule', action='store_true', help='Run in a simple loop scheduler')
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        return 2

    config = load_config(config_path)

    if args.schedule:
        run_schedule(config)
    else:
        run_once(config)
    return 0


if __name__ == '__main__':
    sys.exit(main())


