from __future__ import annotations

from datetime import datetime

import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index, to_utc_timestamp


class HNStoriesCount(ProviderBase):
    BASE_URL = 'https://hn.algolia.com/api/v1/search_by_date'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        query = self.params.get('query', 'bitcoin')
        params = {
            'query': query,
            'tags': 'story',
            'numericFilters': f"created_at_i>={int(start.timestamp())},created_at_i<={int(end.timestamp())}",
            'hitsPerPage': 1000,
            'page': 0,
        }
        total = 0
        while True:
            resp = requests.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            d = resp.json()
            hits = d.get('hits', [])
            total += len(hits)
            if params['page'] >= d.get('nbPages', 0) - 1:
                break
            params['page'] += 1
        df = pd.DataFrame({'timestamp': [to_utc_timestamp(start)], 'stories_count': [total]})
        return ensure_timestamp_index(df)


