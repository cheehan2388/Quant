"""
Data Provider module for fetching market data from various exchanges.
Supports multiple data sources and preprocessing methods.
"""

import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataProvider(ABC):
    """Abstract base class for data providers"""
    
    @abstractmethod
    async def fetch_open_interest(self, symbol: str) -> Dict[str, Any]:
        """Fetch open interest data for a symbol"""
        pass
    
    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the data provider"""
        pass

class BinanceDataProvider(DataProvider):
    """Binance data provider for fetching market data"""
    
    def __init__(self, api_key: str = '', secret_key: str = '', sandbox: bool = False):
        self.exchange = ccxt_async.binance({
            'apiKey': api_key,
            'secret': secret_key,
            'sandbox': sandbox,
            'enableRateLimit': True,
        })
        
    def get_name(self) -> str:
        return "Binance"
    
    async def close(self):
        """Close the exchange connection"""
        if hasattr(self.exchange, 'close'):
            await self.exchange.close()
    
    async def fetch_open_interest(self, symbol: str) -> Dict[str, Any]:
        """Fetch open interest data from Binance"""
        try:
            # Convert symbol format if needed
            symbol_formatted = symbol.replace('/', '')
            
            # Fetch open interest data (async call)
            open_interest = await self.exchange.fetch_open_interest(symbol)
            
            return {
                'symbol': symbol,
                'open_interest': open_interest['openInterestAmount'],
                'timestamp': open_interest['timestamp'],
                'datetime': open_interest['datetime'],
                'provider': self.get_name()
            }
        except Exception as e:
            logger.error(f"Error fetching open interest from Binance: {e}")
            return None
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data from Binance"""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV from Binance: {e}")
            return pd.DataFrame()

class BybitDataProvider(DataProvider):
    """Bybit data provider for fetching market data"""
    
    def __init__(self, api_key: str = '', secret_key: str = '', sandbox: bool = True):
        self.exchange = ccxt_async.bybit({
            'apiKey': api_key,
            'secret': secret_key,
            'sandbox': sandbox,
            'enableRateLimit': True,
        })
        
    def get_name(self) -> str:
        return "Bybit"
    
    async def close(self):
        """Close the exchange connection"""
        if hasattr(self.exchange, 'close'):
            await self.exchange.close()
    
    async def fetch_open_interest(self, symbol: str) -> Dict[str, Any]:
        """Fetch open interest data from Bybit"""
        try:
            open_interest = await self.exchange.fetch_open_interest(symbol)
            
            return {
                'symbol': symbol,
                'open_interest': open_interest['openInterestAmount'],
                'timestamp': open_interest['timestamp'],
                'datetime': open_interest['datetime'],
                'provider': self.get_name()
            }
        except Exception as e:
            logger.error(f"Error fetching open interest from Bybit: {e}")
            return None
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data from Bybit"""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV from Bybit: {e}")
            return pd.DataFrame()

class DataManager:
    """Manages multiple data providers and aggregates data"""
    
    def __init__(self):
        self.providers: Dict[str, DataProvider] = {}
        self.data_history: Dict[str, List[Dict]] = {}
        
    def add_provider(self, name: str, provider: DataProvider):
        """Add a data provider"""
        self.providers[name] = provider
        logger.info(f"Added data provider: {name}")
    
    async def fetch_all_open_interest(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch open interest from all providers"""
        tasks = []
        for name, provider in self.providers.items():
            tasks.append(provider.fetch_open_interest(symbol))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error from provider {list(self.providers.keys())[i]}: {result}")
            elif result is not None:
                valid_results.append(result)
        
        return valid_results
    
    async def fetch_primary_open_interest(self, symbol: str, provider_name: str = None) -> Dict[str, Any]:
        """Fetch open interest from primary provider"""
        if provider_name and provider_name in self.providers:
            provider = self.providers[provider_name]
        else:
            # Use first available provider
            provider = list(self.providers.values())[0] if self.providers else None
            
        if not provider:
            raise ValueError("No data providers available")
        
        return await provider.fetch_open_interest(symbol)
    
    def store_data(self, symbol: str, data: Dict[str, Any]):
        """Store data in history"""
        if symbol not in self.data_history:
            self.data_history[symbol] = []
        
        self.data_history[symbol].append(data)
        
        # Keep only recent data (last 1000 entries)
        if len(self.data_history[symbol]) > 1000:
            self.data_history[symbol] = self.data_history[symbol][-1000:]
    
    def get_historical_data(self, symbol: str, limit: int = None) -> List[Dict[str, Any]]:
        """Get historical data for a symbol"""
        if symbol not in self.data_history:
            return []
        
        data = self.data_history[symbol]
        if limit:
            return data[-limit:]
        return data
    
    def get_data_as_dataframe(self, symbol: str, limit: int = None) -> pd.DataFrame:
        """Convert historical data to pandas DataFrame"""
        data = self.get_historical_data(symbol, limit)
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
        
        return df
