from __future__ import annotations

from datetime import datetime

import requests
import pandas as pd

from core.base import ProviderBase
from core.utils import ensure_timestamp_index


class EtherscanTokenSupply(ProviderBase):
    BASE_URL = 'https://api.etherscan.io/api'

    def fetch_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        # Snapshot current supply; use block-based if needed for history (requires pro/archives)
        contract = self.params.get('contract', '0x514910771AF9Ca656af840dff83E8264EcF986CA')  # LINK
        api_key = self.params.get('api_key')
        params = {
            'module': 'stats',
            'action': 'tokensupply',
            'contractaddress': contract,
            'apikey': api_key or '',
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        if resp.status_code != 200:
            return pd.DataFrame()
        d = resp.json()
        val = pd.to_numeric(d.get('result', 'nan'), errors='coerce')
        ts = pd.Timestamp.utcnow().tz_localize('UTC')
        df = pd.DataFrame([{'timestamp': ts, 'supply': val}])
        return ensure_timestamp_index(df)


