from __future__ import annotations

from datetime import datetime
from urllib.parse import quote
import os
import time

import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index


class WikipediaPageviews(ProviderBase):
    BASE_URL = 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/{access}/{agent}/{article}/{granularity}/{start}/{end}'
    DEFAULT_UA = 'cndatest-data-collector/1.0 (https://www.example.com/contact)'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        project = self.params.get('project', 'en.wikipedia')
        access = self.params.get('access', 'all-access')
        agent = self.params.get('agent', 'all-agents')
        article = self.params.get('article', 'Bitcoin')
        article_escaped = quote(str(article), safe='')
        granularity = self.params.get('granularity', 'daily')
        start_str = start.strftime('%Y%m%d') + '00'
        end_str = end.strftime('%Y%m%d') + '00'
        url = self.BASE_URL.format(project=project, access=access, agent=agent, article=article_escaped, granularity=granularity, start=start_str, end=end_str)

        headers = {
            'User-Agent': self.params.get('user_agent') or os.getenv('WIKI_USER_AGENT') or self.DEFAULT_UA
        }

        # Retry politely on 429, stop on 403 per policy
        for attempt in range(3):
            resp = requests.get(url, timeout=30, headers=headers)
            if resp.status_code == 404:
                return pd.DataFrame()
            if resp.status_code == 403:
                return pd.DataFrame()
            if resp.status_code == 429 and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            try:
                resp.raise_for_status()
            except Exception:
                return pd.DataFrame()
            break

        items = resp.json().get('items', [])
        if not items:
            return pd.DataFrame()
        df = pd.DataFrame(items)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d%H', utc=True)
        df = df.rename(columns={'views': 'pageviews'})
        df = df[['timestamp', 'pageviews']]
        return ensure_timestamp_index(df)


