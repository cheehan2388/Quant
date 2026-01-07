## Multi-source Data Crawler

This folder contains a modular, config-driven framework to automatically fetch time series data from 20+ public data sources, store them with a datetime index, and backfill missing dates.

### Features
- 20+ provider adapters (crypto, FX, macro, markets, weather, news, and more)
- Config-driven registry to enable/disable sources and set parameters
- Storage to CSV under `data/` with `timestamp` index and consistent columns
- Automatic incremental updates and gap backfilling for regular frequencies
- Simple CLI runner and optional scheduler

### Quickstart
1) Install dependencies:
```bash
python -m pip install -r "Quant/Web crawler/requirements.txt"
```

2) Run all enabled sources once (history from 2020 via `defaults.start_date` in config):
```bash
python "Quant/Web crawler/run.py" --once
```

3) Schedule periodic runs (every 6 hours by default):
```bash
python "Quant/Web crawler/run.py" --schedule
```

### Output
CSV files are written to `Quant/Web crawler/data/{provider_name}.csv` with a `timestamp` column as the datetime index. Providers aggregate or normalize data to daily or hourly frequency where applicable.

### Config
Edit `config.yaml` to enable/disable sources and adjust parameters (symbols, intervals, locations, etc.).

### Load CSVs into PostgreSQL (no CSV changes)
- Install a PostgreSQL server and note its connection string, e.g.:
  `postgresql+psycopg2://postgres:password@localhost:5432/marketdata`
- Ensure the database exists.
- Run crawlers (CSV remains the same), then load into DB:
```bash
python "Quant/Web crawler/run_to_db.py" --config "Quant/Web crawler/config.yaml" --db "postgresql+psycopg2://postgres:password@localhost:5432/marketdata"
```
- Tables created:
  - `datasets(id, name, provider_type, description, frequency, extra)`
  - `timeseries_points(id, dataset_id, timestamp, values JSON, is_final)`
- Ingestion behavior:
  - Reads CSVs from `data/`
  - Keeps `timestamp` as primary index
  - Stores all numeric columns in `values` JSON (flexible schema, no CSV change required)
  - Unique constraint on `(dataset_id, timestamp)` prevents duplicates

### Provider smoke tests
Run small, safe tests against each provider without touching CSV outputs:
```bash
python "Quant/Web crawler/run_provider_tests.py" --only-free
```
- Test a single provider:
```bash
python "Quant/Web crawler/run_provider_tests.py" --only binance_klines
```
- Control lookback window:
```bash
python "Quant/Web crawler/run_provider_tests.py" --hours-back 12
```
- Add a friendly UA for Wikipedia (env or param):
```powershell
$env:WIKI_USER_AGENT = "cndatest-data-collector/1.0 (mailto:you@example.com)"
```

### Limits report
See `Quant/Web crawler/report_limits.md` for which providers require API keys or have rate limits. Default config enables only free/no-key providers and uses `start_date: 2020-01-01` to backfill history where available.


