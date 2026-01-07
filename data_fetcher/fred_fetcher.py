import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests
from dotenv import load_dotenv


FRED_API_BASE = "https://api.stlouisfed.org/fred"


@dataclass
class FredFetchConfig:
    api_key: str
    observation_start: str = "2018-01-01"
    observation_end: str = "2025-08-11"
    output_dir: str = os.path.join("Data", "FRED")
    # Requests per second ceiling (approx). A small sleep is applied per request.
    requests_per_second: float = 4.0
    # Number of concurrent series to fetch
    max_workers: int = 6
    # Optional series list file (newline/comma separated), or comma-list via CLI
    series_ids: List[str] = field(default_factory=list)
    # Fetch all series under these category IDs (recursively)
    category_ids: List[int] = field(default_factory=list)
    # Whether to skip writing files that already exist
    skip_existing: bool = True
    # Manifest file for resume support
    manifest_path: Optional[str] = None


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self.interval = 1.0 / max(0.01, requests_per_second)
        self.lock = threading.Lock()
        self.next_time = time.perf_counter()

    def wait(self) -> None:
        with self.lock:
            now = time.perf_counter()
            if self.next_time > now:
                time.sleep(self.next_time - now)
            self.next_time = max(self.next_time + self.interval, time.perf_counter())


def load_api_key_from_env() -> Optional[str]:
    load_dotenv()
    return os.getenv("FRED_API_KEY") or os.getenv("fred_api_key")


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _fred_get(endpoint: str, params: Dict[str, str], api_key: str, rate_limiter: RateLimiter) -> Dict:
    rate_limiter.wait()
    p = {"api_key": api_key, "file_type": "json"}
    p.update(params)
    url = f"{FRED_API_BASE}/{endpoint}"
    resp = requests.get(url, params=p, timeout=30)
    resp.raise_for_status()
    return resp.json()


def list_series_in_category_recursive(category_id: int, api_key: str, rate_limiter: RateLimiter) -> Set[str]:
    discovered: Set[str] = set()
    # BFS categories
    queue: List[int] = [category_id]
    while queue:
        cid = queue.pop(0)
        # Add children categories
        try:
            data_children = _fred_get(
                "category/children", {"category_id": str(cid)}, api_key, rate_limiter
            )
            for child in data_children.get("categories", []) or []:
                queue.append(int(child["id"]))
        except Exception:
            # Continue on errors to be robust over very large traversals
            pass

        # Add series in this category (paginated)
        offset = 0
        limit = 1000
        while True:
            try:
                data_series = _fred_get(
                    "category/series",
                    {
                        "category_id": str(cid),
                        "offset": str(offset),
                        "limit": str(limit),
                        "order_by": "series_id",
                    },
                    api_key,
                    rate_limiter,
                )
                items = data_series.get("seriess", []) or []
                if not items:
                    break
                for s in items:
                    sid = s.get("id") or s.get("series_id")
                    if sid:
                        discovered.add(str(sid))
                if len(items) < limit:
                    break
                offset += limit
            except Exception:
                # On pagination error, break page loop
                break
    return discovered


def fetch_series_observations(
    series_id: str,
    api_key: str,
    start_date: str,
    end_date: str,
    rate_limiter: RateLimiter,
) -> List[Dict[str, str]]:
    # FRED returns up to 100,000 obs per call. Our range is small enough, but paginate for safety.
    observations: List[Dict[str, str]] = []
    params = {
        "series_id": series_id,
        "observation_start": start_date,
        "observation_end": end_date,
        "sort_order": "asc",
    }
    data = _fred_get("series/observations", params, api_key, rate_limiter)
    observations.extend(data.get("observations", []) or [])
    return observations


def write_series_csv(series_id: str, rows: List[Dict[str, str]], output_dir: str) -> str:
    ensure_output_dir(output_dir)
    path = os.path.join(output_dir, f"{series_id}.csv")
    # Determine columns from first row
    columns: List[str]
    if rows:
        # Keep stable column order
        candidate_cols = list(rows[0].keys())
        # Ensure date and value leading if exist
        cols_order = []
        for c in ("date", "value"):
            if c in candidate_cols:
                cols_order.append(c)
        for c in candidate_cols:
            if c not in cols_order:
                cols_order.append(c)
        columns = cols_order
    else:
        columns = ["date", "value"]

    # Write CSV
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for r in rows:
            values = [str(r.get(c, "")) for c in columns]
            # Escape commas by wrapping whole field in quotes if needed
            safe_values = []
            for v in values:
                if "," in v or "\n" in v or '"' in v:
                    v = '"' + v.replace('"', '""') + '"'
                safe_values.append(v)
            f.write(",".join(safe_values) + "\n")
    return path


def load_manifest(path: str) -> Dict[str, Dict]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_manifest(path: str, manifest: Dict[str, Dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, path)


def parse_series_ids_from_file_or_list(series_arg: Optional[str]) -> List[str]:
    if not series_arg:
        return []
    if os.path.isfile(series_arg):
        items: List[str] = []
        with open(series_arg, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "," in line:
                    items.extend([x.strip() for x in line.split(",") if x.strip()])
                else:
                    items.append(line)
        return sorted(set(items))
    # Otherwise, treat as comma-separated list
    return sorted({x.strip() for x in series_arg.split(",") if x.strip()})


def build_series_universe(config: FredFetchConfig, api_key: str, rate_limiter: RateLimiter) -> List[str]:
    all_series: Set[str] = set(config.series_ids)
    for cat_id in config.category_ids:
        discovered = list_series_in_category_recursive(cat_id, api_key, rate_limiter)
        all_series.update(discovered)
    return sorted(all_series)


def fetch_and_write_one(
    series_id: str,
    api_key: str,
    start_date: str,
    end_date: str,
    output_dir: str,
    rate_limiter: RateLimiter,
) -> Tuple[str, Optional[str], Optional[str]]:
    try:
        rows = fetch_series_observations(series_id, api_key, start_date, end_date, rate_limiter)
        out_path = write_series_csv(series_id, rows, output_dir)
        return (series_id, out_path, None)
    except Exception as e:
        return (series_id, None, str(e))


def run_cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "FRED data fetcher: fetch multiple series observations for a given date range, "
            "with optional category-based discovery, resume, and rate limiting."
        )
    )
    parser.add_argument(
        "--api-key",
        help="FRED API key. If omitted, tries FRED_API_KEY from environment/.env",
    )
    parser.add_argument(
        "--start",
        default="2018-01-01",
        help="Observation start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default="2025-08-11",
        help="Observation end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("Data", "FRED"),
        help="Directory to write series CSV files",
    )
    parser.add_argument(
        "--series",
        help=(
            "Series IDs to fetch. Provide a comma-separated list or a path to a file "
            "with one series_id per line (commas allowed)."
        ),
    )
    parser.add_argument(
        "--category-ids",
        help="Comma-separated FRED category IDs to discover series from (recursive)",
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=4.0,
        help="Requests per second cap (approx)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Max concurrent series fetches",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip series whose CSV already exists",
    )
    parser.add_argument(
        "--manifest",
        help="Optional path to a JSON manifest for resume tracking",
    )

    args = parser.parse_args(argv)

    api_key = args.api_key or load_api_key_from_env()
    if not api_key:
        print("ERROR: FRED API key not provided. Use --api-key or set FRED_API_KEY in environment.", file=sys.stderr)
        return 2

    series_ids = parse_series_ids_from_file_or_list(args.series)
    category_ids: List[int] = []
    if args.category_ids:
        for part in args.category_ids.split(","):
            part = part.strip()
            if part:
                try:
                    category_ids.append(int(part))
                except ValueError:
                    print(f"WARNING: Invalid category id '{part}', skipping.")

    config = FredFetchConfig(
        api_key=api_key,
        observation_start=args.start,
        observation_end=args.end,
        output_dir=args.output_dir,
        requests_per_second=args.rps,
        max_workers=args.workers,
        series_ids=series_ids,
        category_ids=category_ids,
        skip_existing=not args.no_skip_existing,
        manifest_path=args.manifest or os.path.join(args.output_dir, "_manifest.json"),
    )

    ensure_output_dir(config.output_dir)
    rate_limiter = RateLimiter(config.requests_per_second)

    # Build series universe
    universe = build_series_universe(config, api_key, rate_limiter)
    if not universe:
        print("No series specified or discovered. Provide --series and/or --category-ids.")
        return 1

    # Load manifest
    manifest = load_manifest(config.manifest_path) if config.manifest_path else {}
    manifest.setdefault("meta", {})
    manifest.setdefault("series", {})
    manifest["meta"].update(
        {
            "observation_start": config.observation_start,
            "observation_end": config.observation_end,
            "last_run": datetime.utcnow().isoformat() + "Z",
        }
    )

    # Optionally skip those already done
    todo: List[str] = []
    for sid in universe:
        out_path = os.path.join(config.output_dir, f"{sid}.csv")
        if config.skip_existing and os.path.exists(out_path):
            manifest["series"].setdefault(sid, {})
            manifest["series"][sid].update({"status": "exists", "path": out_path})
            continue
        status = manifest.get("series", {}).get(sid, {}).get("status")
        if status == "done" and config.skip_existing:
            continue
        todo.append(sid)

    print(f"Total series: {len(universe)} | To fetch: {len(todo)} | Output: {config.output_dir}")

    # Fetch in parallel
    successes = 0
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = [
            executor.submit(
                fetch_and_write_one,
                sid,
                api_key,
                config.observation_start,
                config.observation_end,
                config.output_dir,
                rate_limiter,
            )
            for sid in todo
        ]
        for fut in concurrent.futures.as_completed(futures):
            sid, path, err = fut.result()
            manifest["series"].setdefault(sid, {})
            if err is None and path:
                successes += 1
                manifest["series"][sid].update({"status": "done", "path": path})
                print(f"OK  {sid} -> {path}")
            else:
                failures += 1
                manifest["series"][sid].update({"status": "error", "error": err})
                print(f"ERR {sid} | {err}")
            if config.manifest_path:
                save_manifest(config.manifest_path, manifest)

    print(f"Completed. Success: {successes} | Failed: {failures} | Skipped: {len(universe) - len(todo)}")
    if config.manifest_path:
        save_manifest(config.manifest_path, manifest)
    return 0 if failures == 0 else 3


if __name__ == "__main__":
    raise SystemExit(run_cli())


