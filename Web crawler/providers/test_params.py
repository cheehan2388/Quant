from __future__ import annotations

# Minimal example params per provider type for quick smoke tests.
# These are chosen to avoid API keys and minimize rate-limit risk.

TEST_PARAMS = {
    # Crypto OHLC
    'binance_klines': {
        'symbol': 'BTCUSDT',
        'interval': '1h',
        'frequency': '1H',
    },
    'bitfinex_candles': {
        'symbol': 'tBTCUSD',
        'timeframe': '1h',
        'frequency': '1H',
    },
    'kraken_ohlc': {
        'pair': 'XBTUSD',
        'interval_minutes': 60,
        'frequency': '1H',
    },
    'bitstamp_ohlc': {
        'pair': 'btcusd',
        'step': 3600,
        'frequency': '1H',
    },
    'okx_candles': {
        'instId': 'BTC-USDT',
        'bar': '1H',
        'frequency': '1H',
    },
    'kucoin_candles': {
        'symbol': 'BTC-USDT',
        'type': '1hour',
        'frequency': '1H',
    },
    'coinbase_candles': {
        'product_id': 'BTC-USD',
        'granularity': 3600,
        'frequency': '1H',
    },

    # Public non-crypto
    'coindesk_bpi': {
        'currency': 'USD',
        'frequency': '1D',
    },
    'exchangerate_host_timeseries': {
        'base': 'USD',
        'symbols': 'EUR',
        'frequency': '1D',
    },
    'ecb_fx_daily': {
        'symbols': 'USD',
        'frequency': '1D',
    },
    'open_meteo_hourly': {
        'latitude': 25.0375,
        'longitude': 121.5637,
        'hourly': 'temperature_2m',
        'frequency': '1H',
    },
    'usgs_earthquakes': {
        'frequency': '1D',
    },
    'wikipedia_pageviews': {
        'project': 'en.wikipedia',
        'access': 'all-access',
        'agent': 'all-agents',
        'article': 'Bitcoin',
        'granularity': 'daily',
        'frequency': '1D',
        # Provide a UA via env or override here if needed:
        # 'user_agent': 'cndatest-data-collector/1.0 (mailto:you@example.com)'
    },

    # CoinGecko (may 429 on free tier; test will tolerate empty)
    'coingecko_market_chart': {
        'coin_id': 'bitcoin',
        'vs_currency': 'usd',
        'frequency': '1D',
    },
}

# Providers that generally require API keys; skipped when --only-free
LIKELY_KEY_REQUIRED = {
    'coinmetrics_community',
    'glassnode_metric',
    'alphavantage_fx_daily',
    'fred_series',
    'santiment_metric',
    'coinapi_trades',
}


