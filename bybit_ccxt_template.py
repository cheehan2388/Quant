#!/usr/bin/env python3
"""
CCXT Template for Bybit Exchange
A comprehensive template for interacting with Bybit using the CCXT library.
"""

import ccxt
import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BybitCCXTTemplate:
    """
    A comprehensive CCXT template for Bybit exchange operations.
    
    This template provides methods for:
    - Authentication and connection setup
    - Market data retrieval
    - Trading operations (buy/sell)
    - Position management
    - Account information
    - Error handling and rate limiting
    """
    
    def __init__(self, api_key: str = None, secret: str = None, sandbox: bool = False):
        """
        Initialize the Bybit CCXT template.
        
        Args:
            api_key (str): Your Bybit API key
            secret (str): Your Bybit API secret
            sandbox (bool): Whether to use sandbox/testnet (default: False)
        """
        self.api_key = api_key or os.getenv('BYBIT_API_KEY')
        self.secret = secret or os.getenv('BYBIT_SECRET')
        self.sandbox = sandbox
        
        if not self.api_key or not self.secret:
            raise ValueError("API key and secret are required. Set them as parameters or environment variables.")
        
        # Initialize the exchange
        self.exchange = ccxt.bybit({
            'apiKey': self.api_key,
            'secret': self.secret,
            'sandbox': self.sandbox,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # 'spot', 'linear', 'inverse'
            }
        })
        
        logger.info(f"Initialized Bybit CCXT template (sandbox: {self.sandbox})")
    
    def test_connection(self) -> bool:
        """
        Test the connection to Bybit API.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            # Fetch account balance to test connection
            balance = self.exchange.fetch_balance()
            logger.info("Connection test successful")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information including balances.
        
        Returns:
            Dict containing account information
        """
        try:
            balance = self.exchange.fetch_balance()
            account_info = {
                'total_balance': balance['total'],
                'free_balance': balance['free'],
                'used_balance': balance['used'],
                'timestamp': balance['timestamp'],
                'datetime': balance['datetime']
            }
            logger.info("Account information retrieved successfully")
            return account_info
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise
    
    def get_markets(self) -> Dict[str, Any]:
        """
        Get available markets on Bybit.
        
        Returns:
            Dict containing market information
        """
        try:
            markets = self.exchange.load_markets()
            logger.info(f"Loaded {len(markets)} markets")
            return markets
        except Exception as e:
            logger.error(f"Failed to get markets: {e}")
            raise
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker information for a symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            
        Returns:
            Dict containing ticker information
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            logger.info(f"Retrieved ticker for {symbol}")
            return ticker
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            raise
    
    def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """
        Get order book for a symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            limit (int): Number of orders to retrieve (default: 20)
            
        Returns:
            Dict containing order book data
        """
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit)
            logger.info(f"Retrieved order book for {symbol}")
            return order_book
        except Exception as e:
            logger.error(f"Failed to get order book for {symbol}: {e}")
            raise
    
    def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """
        Get OHLCV (Open, High, Low, Close, Volume) data.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            timeframe (str): Timeframe ('1m', '5m', '15m', '1h', '4h', '1d', etc.)
            limit (int): Number of candles to retrieve
            
        Returns:
            pandas.DataFrame: OHLCV data with datetime index
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            logger.info(f"Retrieved {len(df)} OHLCV candles for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Failed to get OHLCV for {symbol}: {e}")
            raise
    
    def place_market_order(self, symbol: str, side: str, amount: float, 
                          params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Place a market order.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            side (str): 'buy' or 'sell'
            amount (float): Amount to trade
            params (Dict): Additional parameters
            
        Returns:
            Dict containing order information
        """
        try:
            order = self.exchange.create_market_order(symbol, side, amount, params or {})
            logger.info(f"Placed {side} market order for {amount} {symbol}")
            return order
        except Exception as e:
            logger.error(f"Failed to place market order: {e}")
            raise
    
    def place_limit_order(self, symbol: str, side: str, amount: float, price: float,
                         params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Place a limit order.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            side (str): 'buy' or 'sell'
            amount (float): Amount to trade
            price (float): Limit price
            params (Dict): Additional parameters
            
        Returns:
            Dict containing order information
        """
        try:
            order = self.exchange.create_limit_order(symbol, side, amount, price, params or {})
            logger.info(f"Placed {side} limit order for {amount} {symbol} at {price}")
            return order
        except Exception as e:
            logger.error(f"Failed to place limit order: {e}")
            raise
    
    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Cancel an existing order.
        
        Args:
            order_id (str): Order ID to cancel
            symbol (str): Trading symbol
            
        Returns:
            Dict containing cancellation information
        """
        try:
            result = self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Cancelled order {order_id} for {symbol}")
            return result
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise
    
    def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Get the status of an order.
        
        Args:
            order_id (str): Order ID
            symbol (str): Trading symbol
            
        Returns:
            Dict containing order status
        """
        try:
            order = self.exchange.fetch_order(order_id, symbol)
            logger.info(f"Retrieved status for order {order_id}")
            return order
        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            raise
    
    def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Get all open orders.
        
        Args:
            symbol (str): Optional symbol filter
            
        Returns:
            List of open orders
        """
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            logger.info(f"Retrieved {len(orders)} open orders")
            return orders
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            raise
    
    def get_trade_history(self, symbol: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get trade history.
        
        Args:
            symbol (str): Optional symbol filter
            limit (int): Number of trades to retrieve
            
        Returns:
            List of trades
        """
        try:
            trades = self.exchange.fetch_my_trades(symbol, limit=limit)
            logger.info(f"Retrieved {len(trades)} trades")
            return trades
        except Exception as e:
            logger.error(f"Failed to get trade history: {e}")
            raise
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions (for futures trading).
        
        Returns:
            List of positions
        """
        try:
            positions = self.exchange.fetch_positions()
            logger.info(f"Retrieved {len(positions)} positions")
            return positions
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise
    
    def calculate_position_size(self, symbol: str, risk_percentage: float, 
                              stop_loss_percentage: float) -> float:
        """
        Calculate position size based on risk management.
        
        Args:
            symbol (str): Trading symbol
            risk_percentage (float): Risk percentage of account balance
            stop_loss_percentage (float): Stop loss percentage
            
        Returns:
            float: Calculated position size
        """
        try:
            # Get account balance
            balance = self.get_account_info()
            usdt_balance = balance['free_balance'].get('USDT', 0)
            
            # Get current price
            ticker = self.get_ticker(symbol)
            current_price = ticker['last']
            
            # Calculate risk amount
            risk_amount = usdt_balance * (risk_percentage / 100)
            
            # Calculate position size
            position_size = risk_amount / (stop_loss_percentage / 100)
            
            logger.info(f"Calculated position size: {position_size} for {symbol}")
            return position_size
        except Exception as e:
            logger.error(f"Failed to calculate position size: {e}")
            raise
    
    def wait_for_order_completion(self, order_id: str, symbol: str, 
                                 timeout: int = 300) -> Dict[str, Any]:
        """
        Wait for an order to be completed.
        
        Args:
            order_id (str): Order ID
            symbol (str): Trading symbol
            timeout (int): Timeout in seconds
            
        Returns:
            Dict containing completed order information
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                order = self.get_order_status(order_id, symbol)
                
                if order['status'] in ['closed', 'canceled']:
                    logger.info(f"Order {order_id} completed with status: {order['status']}")
                    return order
                
                time.sleep(2)  # Wait 2 seconds before checking again
                
            except Exception as e:
                logger.error(f"Error checking order status: {e}")
                time.sleep(5)
        
        raise TimeoutError(f"Order {order_id} did not complete within {timeout} seconds")
    
    def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Get funding rate for perpetual futures.
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            Dict containing funding rate information
        """
        try:
            funding_rate = self.exchange.fetch_funding_rate(symbol)
            logger.info(f"Retrieved funding rate for {symbol}")
            return funding_rate
        except Exception as e:
            logger.error(f"Failed to get funding rate for {symbol}: {e}")
            raise
    
    def get_historical_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical trades for a symbol.
        
        Args:
            symbol (str): Trading symbol
            limit (int): Number of trades to retrieve
            
        Returns:
            List of historical trades
        """
        try:
            trades = self.exchange.fetch_trades(symbol, limit=limit)
            logger.info(f"Retrieved {len(trades)} historical trades for {symbol}")
            return trades
        except Exception as e:
            logger.error(f"Failed to get historical trades for {symbol}: {e}")
            raise


def main():
    """
    Example usage of the Bybit CCXT template.
    """
    # Example configuration
    api_key = os.getenv('BYBIT_API_KEY')
    secret = os.getenv('BYBIT_SECRET')
    
    if not api_key or not secret:
        print("Please set BYBIT_API_KEY and BYBIT_SECRET environment variables")
        return
    
    try:
        # Initialize the template
        bybit = BybitCCXTTemplate(api_key=api_key, secret=secret, sandbox=True)
        
        # Test connection
        if not bybit.test_connection():
            print("Failed to connect to Bybit")
            return
        
        print("Successfully connected to Bybit!")
        
        # Get account information
        account_info = bybit.get_account_info()
        print(f"Account balance: {account_info['total_balance']}")
        
        # Get markets
        markets = bybit.get_markets()
        print(f"Available markets: {len(markets)}")
        
        # Get ticker for BTC/USDT
        ticker = bybit.get_ticker('BTC/USDT')
        print(f"BTC/USDT price: ${ticker['last']}")
        
        # Get OHLCV data
        ohlcv = bybit.get_ohlcv('BTC/USDT', '1h', 24)
        print(f"Retrieved {len(ohlcv)} hours of BTC/USDT data")
        
        # Example of placing an order (commented out for safety)
        # order = bybit.place_limit_order('BTC/USDT', 'buy', 0.001, 50000)
        # print(f"Placed order: {order['id']}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()