from typing import Optional, Type

from .binance import BinanceKlines
from .coingecko import CoingeckoMarketChart
from .yahoo import YahooPrice
from .exchangerate_host import ExchangeRateHostTimeseries
from .worldbank import WorldBankIndicator
from .owid_covid import OWIDCovidCountry
from .usgs_quakes import USGSEarthquakes
from .open_meteo import OpenMeteoHourly
from .bitfinex import BitfinexCandles
from .kraken import KrakenOHLC
from .bitstamp import BitstampOHLC
from .wikipedia_pageviews import WikipediaPageviews
from .coinpaprika import CoinpaprikaHistorical
from .fear_greed import FearGreedIndex
from .coinbase import CoinbaseCandles
from .kucoin import KucoinCandles
from .okx import OkxCandles
from .huobi import HuobiKlines
from .coindesk import CoindeskBPI
from .hn_stories import HNStoriesCount
from .ecb_fx import ECBFxDaily
from .bybit import BybitKlines
from .ftx_archive import FTXArchiveKlines
from .coinmetrics import CoinMetricsCommunity
from .glassnode import GlassnodeMetric
from .cryptocompare import CryptoCompareHourly
from .alphavantage import AlphaVantageFXDaily
from .fred import FREDSeries
from .coinbase_spotbook import CoinbaseBestBidAsk
from .theta import ThetaNetworkPrice
from .cme_commitments import COTFutures
from .treasury_yield import USTreasuryDaily
from .binance_funding import BinanceFundingRate
from .binance_open_interest import BinanceOpenInterest
from .santiment import SantimentMetric
from .coinapi import CoinAPITrades
from .messari import MessariAssetTimeseries
from .defillama import DefiLlamaTVL
from .cboe_vix import CBOEVIXDaily


_REGISTRY = {
    'binance_klines': BinanceKlines,
    'coingecko_market_chart': CoingeckoMarketChart,
    'yahoo_price': YahooPrice,
    'exchangerate_host_timeseries': ExchangeRateHostTimeseries,
    'worldbank_indicator': WorldBankIndicator,
    'owid_covid_country': OWIDCovidCountry,
    'usgs_earthquakes': USGSEarthquakes,
    'open_meteo_hourly': OpenMeteoHourly,
    'bitfinex_candles': BitfinexCandles,
    'kraken_ohlc': KrakenOHLC,
    'bitstamp_ohlc': BitstampOHLC,
    'wikipedia_pageviews': WikipediaPageviews,
    'coinpaprika_historical': CoinpaprikaHistorical,
    'fear_greed_index': FearGreedIndex,
    'coinbase_candles': CoinbaseCandles,
    'kucoin_candles': KucoinCandles,
    'okx_candles': OkxCandles,
    'huobi_klines': HuobiKlines,
    'coindesk_bpi': CoindeskBPI,
    'hn_stories_count': HNStoriesCount,
    'ecb_fx_daily': ECBFxDaily,
    'bybit_klines': BybitKlines,
    'ftx_archive_klines': FTXArchiveKlines,
    'coinmetrics_community': CoinMetricsCommunity,
    'glassnode_metric': GlassnodeMetric,
    'cryptocompare_hourly': CryptoCompareHourly,
    'alphavantage_fx_daily': AlphaVantageFXDaily,
    'fred_series': FREDSeries,
    'coinbase_best_bid_ask': CoinbaseBestBidAsk,
    'theta_network_price': ThetaNetworkPrice,
    'cot_futures': COTFutures,
    'us_treasury_daily': USTreasuryDaily,
    'binance_funding_rate': BinanceFundingRate,
    'binance_open_interest': BinanceOpenInterest,
    'santiment_metric': SantimentMetric,
    'coinapi_trades': CoinAPITrades,
    'messari_asset_timeseries': MessariAssetTimeseries,
    'defillama_tvl': DefiLlamaTVL,
    'cboe_vix_daily': CBOEVIXDaily,
}


def get_provider_class(key: str) -> Optional[Type]:
    return _REGISTRY.get(key)


