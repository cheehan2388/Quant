#!/usr/bin/env python3
"""
Configuration file for Bybit CCXT template
Contains all the necessary settings and parameters for Bybit trading.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
BYBIT_CONFIG = {
    'api_key': os.getenv('BYBIT_API_KEY'),
    'secret': os.getenv('BYBIT_SECRET'),
    'sandbox': os.getenv('BYBIT_SANDBOX', 'true').lower() == 'true',  # Default to sandbox for safety
    'enable_rate_limit': True,
    'default_type': 'spot',  # 'spot', 'linear', 'inverse'
}

# Trading Configuration
TRADING_CONFIG = {
    'default_symbol': 'BTC/USDT',
    'default_timeframe': '1h',
    'default_limit': 100,
    'max_retries': 3,
    'retry_delay': 1,  # seconds
    'order_timeout': 300,  # seconds
}

# Risk Management Configuration
RISK_CONFIG = {
    'max_risk_per_trade': 2.0,  # percentage of account balance
    'max_position_size': 10.0,  # percentage of account balance
    'default_stop_loss': 5.0,  # percentage
    'default_take_profit': 10.0,  # percentage
    'max_open_orders': 5,
}

# Market Data Configuration
MARKET_DATA_CONFIG = {
    'ohlcv_limit': 1000,
    'order_book_limit': 20,
    'trades_limit': 100,
    'ticker_update_interval': 1,  # seconds
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'bybit_trading.log',
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
}

# Supported Timeframes
TIMEFRAMES = [
    '1m', '3m', '5m', '15m', '30m',
    '1h', '2h', '4h', '6h', '8h', '12h',
    '1d', '3d', '1w', '1M'
]

# Popular Trading Pairs
POPULAR_PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
    'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'MATIC/USDT',
    'LINK/USDT', 'LTC/USDT', 'BCH/USDT', 'UNI/USDT', 'ATOM/USDT'
]

# Order Types
ORDER_TYPES = {
    'market': 'market',
    'limit': 'limit',
    'stop': 'stop',
    'stop_limit': 'stop_limit',
    'take_profit': 'take_profit',
    'take_profit_limit': 'take_profit_limit'
}

# Position Sides
POSITION_SIDES = {
    'long': 'long',
    'short': 'short'
}

# Order Sides
ORDER_SIDES = {
    'buy': 'buy',
    'sell': 'sell'
}

# Error Messages
ERROR_MESSAGES = {
    'api_key_missing': 'Bybit API key is required. Set BYBIT_API_KEY environment variable.',
    'secret_missing': 'Bybit API secret is required. Set BYBIT_SECRET environment variable.',
    'connection_failed': 'Failed to connect to Bybit API. Check your credentials and network connection.',
    'insufficient_balance': 'Insufficient balance for this trade.',
    'invalid_symbol': 'Invalid trading symbol provided.',
    'order_failed': 'Failed to place order. Check your parameters and account balance.',
    'rate_limit_exceeded': 'Rate limit exceeded. Please wait before making more requests.',
}

# Success Messages
SUCCESS_MESSAGES = {
    'connection_successful': 'Successfully connected to Bybit API.',
    'order_placed': 'Order placed successfully.',
    'order_cancelled': 'Order cancelled successfully.',
    'data_retrieved': 'Data retrieved successfully.',
}

def validate_config():
    """
    Validate the configuration settings.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    errors = []
    
    # Check required environment variables
    if not BYBIT_CONFIG['api_key']:
        errors.append(ERROR_MESSAGES['api_key_missing'])
    
    if not BYBIT_CONFIG['secret']:
        errors.append(ERROR_MESSAGES['secret_missing'])
    
    # Validate trading configuration
    if RISK_CONFIG['max_risk_per_trade'] <= 0 or RISK_CONFIG['max_risk_per_trade'] > 100:
        errors.append('max_risk_per_trade must be between 0 and 100')
    
    if RISK_CONFIG['max_position_size'] <= 0 or RISK_CONFIG['max_position_size'] > 100:
        errors.append('max_position_size must be between 0 and 100')
    
    # Validate market data configuration
    if MARKET_DATA_CONFIG['ohlcv_limit'] <= 0:
        errors.append('ohlcv_limit must be greater than 0')
    
    if len(errors) > 0:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True

def get_config_summary():
    """
    Get a summary of the current configuration.
    
    Returns:
        dict: Configuration summary
    """
    return {
        'api_configured': bool(BYBIT_CONFIG['api_key'] and BYBIT_CONFIG['secret']),
        'sandbox_mode': BYBIT_CONFIG['sandbox'],
        'default_symbol': TRADING_CONFIG['default_symbol'],
        'default_timeframe': TRADING_CONFIG['default_timeframe'],
        'max_risk_per_trade': RISK_CONFIG['max_risk_per_trade'],
        'max_position_size': RISK_CONFIG['max_position_size'],
    }

if __name__ == "__main__":
    # Test configuration validation
    if validate_config():
        print("Configuration is valid!")
        summary = get_config_summary()
        print(f"Configuration summary: {summary}")
    else:
        print("Configuration validation failed!")