import ccxt
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
import requests


class BinanceIndicatorsFetcher:
    """
    Fetches Open Interest and Long/Short Ratio data from Binance
    """
    
    def __init__(self, api_key: str = None, secret: str = None):
        """
        Initialize Binance data fetcher
        
        Args:
            api_key: Binance API key (optional for public data)
            secret: Binance secret (optional for public data)
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize Binance exchange (for public data, API keys are optional)
        self.exchange = ccxt.binance({
            'apiKey': api_key or '',
            'secret': secret or '',
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # Base URL for Binance futures API
        self.base_url = 'https://fapi.binance.com'
        
    def get_open_interest(self, symbol: str) -> Optional[float]:
        """
        Fetch current open interest for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            
        Returns:
            Open interest value or None if failed
        """
        try:
            url = f"{self.base_url}/fapi/v1/openInterest"
            params = {'symbol': symbol}
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            open_interest = float(data['openInterest'])
            
            self.logger.info(f"Open Interest for {symbol}: {open_interest}")
            return open_interest
            
        except Exception as e:
            self.logger.error(f"Error fetching open interest for {symbol}: {e}")
            return None
    
    def get_long_short_ratio(self, symbol: str, period: str = '5m') -> Optional[float]:
        """
        Fetch Long/Short Ratio for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            period: Time period ('5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d')
            
        Returns:
            Long/Short ratio or None if failed
        """
        try:
            url = f"{self.base_url}/fapi/v1/globalLongShortAccountRatio"
            params = {
                'symbol': symbol,
                'period': period,
                'limit': 1  # Get only the latest data
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if not data:
                self.logger.warning(f"No long/short ratio data for {symbol}")
                return None
                
            latest_data = data[0]  # Get the most recent entry
            long_short_ratio = float(latest_data['longShortRatio'])
            
            self.logger.info(f"Long/Short Ratio for {symbol}: {long_short_ratio}")
            return long_short_ratio
            
        except Exception as e:
            self.logger.error(f"Error fetching long/short ratio for {symbol}: {e}")
            return None
    
    def get_combined_indicator(self, symbol: str, period: str = '5m') -> Optional[float]:
        """
        Get the combined indicator (Open Interest * Long/Short Ratio)
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            period: Time period for long/short ratio
            
        Returns:
            Combined indicator value or None if failed
        """
        try:
            # Fetch both indicators
            open_interest = self.get_open_interest(symbol)
            long_short_ratio = self.get_long_short_ratio(symbol, period)
            
            if open_interest is None or long_short_ratio is None:
                self.logger.warning(f"Could not fetch complete data for {symbol}")
                return None
            
            # Calculate combined indicator
            combined_indicator = open_interest * long_short_ratio
            
            self.logger.info(f"Combined Indicator for {symbol}: {combined_indicator}")
            return combined_indicator
            
        except Exception as e:
            self.logger.error(f"Error calculating combined indicator for {symbol}: {e}")
            return None
    
    def get_historical_combined_data(self, symbol: str, limit: int = 100, period: str = '5m') -> pd.DataFrame:
        """
        Get historical combined indicator data for backtesting or analysis
        
        Args:
            symbol: Trading symbol
            limit: Number of historical points to fetch
            period: Time period
            
        Returns:
            DataFrame with timestamp, open_interest, long_short_ratio, and combined_indicator
        """
        try:
            # Get historical long/short ratio data
            url = f"{self.base_url}/fapi/v1/globalLongShortAccountRatio"
            params = {
                'symbol': symbol,
                'period': period,
                'limit': limit
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            ls_data = response.json()
            
            # Create DataFrame
            df = pd.DataFrame(ls_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['longShortRatio'] = df['longShortRatio'].astype(float)
            df['longAccount'] = df['longAccount'].astype(float)
            df['shortAccount'] = df['shortAccount'].astype(float)
            
            # Note: Historical open interest data requires different endpoint
            # For now, we'll use current open interest as approximation
            current_oi = self.get_open_interest(symbol)
            if current_oi:
                df['open_interest'] = current_oi  # This is simplified - in reality you'd want historical OI
                df['combined_indicator'] = df['longShortRatio'] * df['open_interest']
            else:
                df['open_interest'] = np.nan
                df['combined_indicator'] = np.nan
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.info(f"Fetched {len(df)} historical data points for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()


def test_binance_fetcher():
    """Test function for the Binance indicators fetcher"""
    logging.basicConfig(level=logging.INFO)
    
    fetcher = BinanceIndicatorsFetcher()
    symbol = 'BTCUSDT'
    
    # Test individual indicators
    oi = fetcher.get_open_interest(symbol)
    ls_ratio = fetcher.get_long_short_ratio(symbol)
    combined = fetcher.get_combined_indicator(symbol)
    
    print(f"Open Interest: {oi}")
    print(f"Long/Short Ratio: {ls_ratio}")
    print(f"Combined Indicator: {combined}")
    
    # Test historical data
    df = fetcher.get_historical_combined_data(symbol, limit=20)
    print("\nHistorical Data:")
    print(df.tail())


if __name__ == "__main__":
    test_binance_fetcher()