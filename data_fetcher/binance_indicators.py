import ccxt
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List
import requests
import sys
import os

# Add config path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.settings import get_config


class BinanceIndicatorsFetcher:
    """
    Fetches Open Interest and Long/Short Ratio data from Binance with proper timing
    """
    
    def __init__(self, api_key: str = None, secret: str = None, config_manager=None):
        """
        Initialize Binance data fetcher
        
        Args:
            api_key: Binance API key (optional for public data)
            secret: Binance secret (optional for public data)
            config_manager: Configuration manager instance
        """
        self.logger = logging.getLogger(__name__)
        
        # Get configuration
        self.config = config_manager or get_config()
        
        # Initialize Binance exchange (for public data, API keys are optional)
        self.exchange = ccxt.binance({
            'apiKey': api_key or '',
            'secret': secret or '',
            'sandbox': False,
            'enableRateLimit': self.config.exchange.binance_rate_limit,
        })
        
        # Base URL for Binance futures API
        self.base_url = 'https://fapi.binance.com'
        
        # Data storage for timestamps and caching
        self.last_fetched_timestamp = None
        self.cached_data = {}
        self.data_history = []  # Store historical data for initial loading
        
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
    
    def get_current_hour_timestamp(self) -> datetime:
        """
        Get the current hour timestamp (e.g., 2024-01-01 14:00:00)
        
        Returns:
            Current hour timestamp
        """
        now = datetime.utcnow()
        return now.replace(minute=0, second=0, microsecond=0)
    
    def is_new_hour_data_available(self, symbol: str, target_timestamp: datetime) -> bool:
        """
        Check if data for the target hour is available
        
        Args:
            symbol: Trading symbol
            target_timestamp: Target hour timestamp
            
        Returns:
            True if data is available, False otherwise
        """
        try:
            # Fetch latest long/short ratio data
            url = f"{self.base_url}/fapi/v1/globalLongShortAccountRatio"
            params = {
                'symbol': symbol,
                'period': self.config.data.binance_long_short_period,
                'limit': 1
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if not data:
                return False
            
            # Check if the latest data timestamp matches our target
            latest_timestamp = datetime.utcfromtimestamp(data[0]['timestamp'] / 1000)
            latest_hour = latest_timestamp.replace(minute=0, second=0, microsecond=0)
            
            self.logger.debug(f"Target: {target_timestamp}, Latest: {latest_hour}")
            
            return latest_hour >= target_timestamp
            
        except Exception as e:
            self.logger.error(f"Error checking data availability: {e}")
            return False
    
    def wait_for_new_hour_data(self, symbol: str, max_wait_minutes: int = None) -> bool:
        """
        Wait for new hour data to become available
        
        Args:
            symbol: Trading symbol
            max_wait_minutes: Maximum time to wait (uses config if None)
            
        Returns:
            True if data became available, False if timeout
        """
        if max_wait_minutes is None:
            max_wait_minutes = self.config.data.max_retry_attempts * self.config.data.retry_interval_minutes
        
        target_timestamp = self.get_current_hour_timestamp()
        
        # If we're not fetching on hour marks, just check once
        if not self.config.data.fetch_on_hour_mark:
            return self.is_new_hour_data_available(symbol, target_timestamp)
        
        self.logger.info(f"Waiting for {target_timestamp} data to become available...")
        
        start_time = datetime.utcnow()
        attempts = 0
        
        while attempts < self.config.data.max_retry_attempts:
            if self.is_new_hour_data_available(symbol, target_timestamp):
                self.logger.info(f"Data for {target_timestamp} is now available")
                return True
            
            attempts += 1
            elapsed_minutes = (datetime.utcnow() - start_time).total_seconds() / 60
            
            if elapsed_minutes >= max_wait_minutes:
                self.logger.warning(f"Timeout waiting for {target_timestamp} data")
                break
            
            self.logger.info(f"Data not yet available, waiting {self.config.data.retry_interval_minutes} minutes... (attempt {attempts}/{self.config.data.max_retry_attempts})")
            time.sleep(self.config.data.retry_interval_minutes * 60)
        
        return False
    
    def get_combined_indicator_with_timestamp(self, symbol: str, period: str = None) -> Optional[Tuple[float, datetime]]:
        """
        Get the combined indicator with its timestamp, ensuring data freshness
        
        Args:
            symbol: Trading symbol
            period: Time period (uses config default if None)
            
        Returns:
            Tuple of (combined_indicator, timestamp) or None if failed
        """
        if period is None:
            period = self.config.data.binance_long_short_period
        
        try:
            # If configured to fetch only on hour marks, wait for fresh data
            if self.config.data.fetch_on_hour_mark:
                current_hour = self.get_current_hour_timestamp()
                
                # Check if we need to wait for new data
                if (self.last_fetched_timestamp is None or 
                    self.last_fetched_timestamp < current_hour):
                    
                    if not self.wait_for_new_hour_data(symbol):
                        self.logger.error("Failed to get fresh hourly data")
                        return None
            
            # Fetch the indicators
            open_interest = self.get_open_interest(symbol)
            if open_interest is None:
                return None
            
            # Get long/short ratio with timestamp
            url = f"{self.base_url}/fapi/v1/globalLongShortAccountRatio"
            params = {
                'symbol': symbol,
                'period': period,
                'limit': 1
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if not data:
                return None
            
            latest_data = data[0]
            long_short_ratio = float(latest_data['longShortRatio'])
            data_timestamp = datetime.utcfromtimestamp(latest_data['timestamp'] / 1000)
            
            # Check data freshness
            now = datetime.utcnow()
            age_minutes = (now - data_timestamp).total_seconds() / 60
            
            if age_minutes > self.config.data.data_freshness_threshold_minutes:
                self.logger.warning(f"Data is {age_minutes:.1f} minutes old, may be stale")
            
            # Calculate combined indicator
            combined_indicator = open_interest * long_short_ratio
            
            # Update tracking
            self.last_fetched_timestamp = data_timestamp
            
            self.logger.info(f"Combined indicator: {combined_indicator:.6f} at {data_timestamp}")
            
            return combined_indicator, data_timestamp
            
        except Exception as e:
            self.logger.error(f"Error fetching combined indicator with timestamp: {e}")
            return None
    
    def load_initial_historical_data(self, symbol: str, hours: int = None) -> List[Tuple[float, datetime]]:
        """
        Load initial historical data to populate the rolling window
        
        Args:
            symbol: Trading symbol
            hours: Hours of historical data to fetch (uses config if None)
            
        Returns:
            List of (combined_indicator, timestamp) tuples
        """
        if hours is None:
            hours = self.config.data.initial_data_hours
        
        self.logger.info(f"Loading {hours} hours of initial historical data for {symbol}")
        
        try:
            # Calculate how many data points we need
            # For 5m periods, we need 12 points per hour
            period_minutes = self._period_to_minutes(self.config.data.binance_long_short_period)
            points_per_hour = 60 // period_minutes
            total_points = hours * points_per_hour
            
            # Fetch historical long/short ratio data
            url = f"{self.base_url}/fapi/v1/globalLongShortAccountRatio"
            params = {
                'symbol': symbol,
                'period': self.config.data.binance_long_short_period,
                'limit': min(total_points, 500)  # API limit
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            ls_data = response.json()
            
            if not ls_data:
                self.logger.error("No historical long/short ratio data available")
                return []
            
            # Get current open interest (simplified - in reality you'd want historical OI)
            current_oi = self.get_open_interest(symbol)
            if current_oi is None:
                self.logger.error("Could not fetch current open interest")
                return []
            
            # Process historical data
            historical_data = []
            for item in reversed(ls_data):  # Reverse to get chronological order
                timestamp = datetime.utcfromtimestamp(item['timestamp'] / 1000)
                long_short_ratio = float(item['longShortRatio'])
                combined_indicator = current_oi * long_short_ratio
                
                historical_data.append((combined_indicator, timestamp))
            
            self.data_history = historical_data
            self.logger.info(f"Loaded {len(historical_data)} historical data points")
            
            return historical_data
            
        except Exception as e:
            self.logger.error(f"Error loading initial historical data: {e}")
            return []
    
    def _period_to_minutes(self, period: str) -> int:
        """
        Convert period string to minutes
        
        Args:
            period: Period string (e.g., '5m', '1h')
            
        Returns:
            Number of minutes
        """
        if period.endswith('m'):
            return int(period[:-1])
        elif period.endswith('h'):
            return int(period[:-1]) * 60
        elif period.endswith('d'):
            return int(period[:-1]) * 24 * 60
        else:
            return 5  # Default to 5 minutes
    
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