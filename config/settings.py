#!/usr/bin/env python3
"""
Centralized configuration management for the Z-Score Trading Bot
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv


@dataclass
class TradingConfig:
    """Trading strategy configuration"""
    # Strategy parameters
    z_score_threshold: float = 2.1
    rolling_window: int = 15
    
    # Position management
    position_size: float = 0.001
    max_position_time_hours: float = 1.0
    
    # Risk management
    max_daily_trades: int = 10
    max_consecutive_losses: int = 3
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None


@dataclass
class DataConfig:
    """Data fetching configuration"""
    # Timing parameters
    fetch_on_hour_mark: bool = True  # Only fetch at :00 of each hour
    retry_interval_minutes: int = 1  # Retry every minute if data not available
    max_retry_attempts: int = 10     # Max retries before giving up
    
    # Data sources
    binance_long_short_period: str = '5m'  # Period for long/short ratio
    
    # Initial data requirements
    initial_data_hours: int = 24  # Hours of historical data to fetch initially
    
    # Stale data handling
    max_stale_data_tolerance: int = 3
    data_freshness_threshold_minutes: int = 10  # Consider data stale if older than this


@dataclass
class ExchangeConfig:
    """Exchange configuration"""
    # Bybit settings
    bybit_testnet: bool = True  # Set to False for live trading
    bybit_rate_limit: bool = True
    
    # Binance settings
    binance_rate_limit: bool = True
    
    # Order execution
    order_timeout_seconds: int = 30
    order_retry_attempts: int = 3


@dataclass
class SystemConfig:
    """System-level configuration"""
    # Logging
    log_level: str = 'INFO'
    log_to_file: bool = True
    log_to_console: bool = True
    max_log_files: int = 10
    
    # Monitoring
    status_log_interval_minutes: int = 10
    performance_log_interval_minutes: int = 60
    
    # Data persistence
    save_data_to_file: bool = True
    data_file_path: str = 'data/trading_data.csv'
    
    # Safety
    enable_trading: bool = True  # Master switch for trading
    dry_run_mode: bool = False   # Log trades but don't execute


class ConfigManager:
    """
    Centralized configuration manager
    """
    
    def __init__(self, config_file: str = '.env'):
        """
        Initialize configuration manager
        
        Args:
            config_file: Path to environment file
        """
        self.logger = logging.getLogger(__name__)
        self.config_file = config_file
        
        # Load environment variables
        load_dotenv(config_file)
        
        # Initialize configuration objects
        self.trading = TradingConfig()
        self.data = DataConfig()
        self.exchange = ExchangeConfig()
        self.system = SystemConfig()
        
        # API credentials
        self.api_keys = self._load_api_keys()
        
        # Trading symbol
        self.symbol = os.getenv('SYMBOL', 'BTCUSDT')
        
        # Load and validate configuration
        self._load_from_environment()
        self._validate_config()
        
        self.logger.info("Configuration loaded successfully")
    
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment"""
        return {
            'bybit_api_key': os.getenv('BYBIT_API_KEY', ''),
            'bybit_secret': os.getenv('BYBIT_SECRET', ''),
            'binance_api_key': os.getenv('BINANCE_API_KEY', ''),
            'binance_secret': os.getenv('BINANCE_SECRET', ''),
        }
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        # Trading configuration
        self.trading.z_score_threshold = float(os.getenv('Z_SCORE_THRESHOLD', self.trading.z_score_threshold))
        self.trading.rolling_window = int(os.getenv('ROLLING_WINDOW', self.trading.rolling_window))
        self.trading.position_size = float(os.getenv('POSITION_SIZE', self.trading.position_size))
        self.trading.max_position_time_hours = float(os.getenv('MAX_POSITION_TIME_HOURS', self.trading.max_position_time_hours))
        self.trading.max_daily_trades = int(os.getenv('MAX_DAILY_TRADES', self.trading.max_daily_trades))
        self.trading.max_consecutive_losses = int(os.getenv('MAX_CONSECUTIVE_LOSSES', self.trading.max_consecutive_losses))
        
        # Optional risk management
        stop_loss = os.getenv('STOP_LOSS_PCT')
        if stop_loss:
            self.trading.stop_loss_pct = float(stop_loss)
        
        take_profit = os.getenv('TAKE_PROFIT_PCT')
        if take_profit:
            self.trading.take_profit_pct = float(take_profit)
        
        # Data configuration
        self.data.fetch_on_hour_mark = os.getenv('FETCH_ON_HOUR_MARK', 'true').lower() == 'true'
        self.data.retry_interval_minutes = int(os.getenv('RETRY_INTERVAL_MINUTES', self.data.retry_interval_minutes))
        self.data.max_retry_attempts = int(os.getenv('MAX_RETRY_ATTEMPTS', self.data.max_retry_attempts))
        self.data.binance_long_short_period = os.getenv('BINANCE_LS_PERIOD', self.data.binance_long_short_period)
        self.data.initial_data_hours = int(os.getenv('INITIAL_DATA_HOURS', self.data.initial_data_hours))
        self.data.max_stale_data_tolerance = int(os.getenv('MAX_STALE_DATA_TOLERANCE', self.data.max_stale_data_tolerance))
        self.data.data_freshness_threshold_minutes = int(os.getenv('DATA_FRESHNESS_THRESHOLD_MINUTES', self.data.data_freshness_threshold_minutes))
        
        # Exchange configuration
        self.exchange.bybit_testnet = os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'
        self.exchange.order_timeout_seconds = int(os.getenv('ORDER_TIMEOUT_SECONDS', self.exchange.order_timeout_seconds))
        self.exchange.order_retry_attempts = int(os.getenv('ORDER_RETRY_ATTEMPTS', self.exchange.order_retry_attempts))
        
        # System configuration
        self.system.log_level = os.getenv('LOG_LEVEL', self.system.log_level).upper()
        self.system.log_to_file = os.getenv('LOG_TO_FILE', 'true').lower() == 'true'
        self.system.log_to_console = os.getenv('LOG_TO_CONSOLE', 'true').lower() == 'true'
        self.system.status_log_interval_minutes = int(os.getenv('STATUS_LOG_INTERVAL_MINUTES', self.system.status_log_interval_minutes))
        self.system.performance_log_interval_minutes = int(os.getenv('PERFORMANCE_LOG_INTERVAL_MINUTES', self.system.performance_log_interval_minutes))
        self.system.save_data_to_file = os.getenv('SAVE_DATA_TO_FILE', 'true').lower() == 'true'
        self.system.data_file_path = os.getenv('DATA_FILE_PATH', self.system.data_file_path)
        self.system.enable_trading = os.getenv('ENABLE_TRADING', 'true').lower() == 'true'
        self.system.dry_run_mode = os.getenv('DRY_RUN_MODE', 'false').lower() == 'true'
    
    def _validate_config(self):
        """Validate configuration values"""
        errors = []
        
        # Validate trading config
        if self.trading.z_score_threshold <= 0:
            errors.append("Z_SCORE_THRESHOLD must be greater than 0")
        
        if self.trading.rolling_window < 2:
            errors.append("ROLLING_WINDOW must be at least 2")
        
        if self.trading.position_size <= 0:
            errors.append("POSITION_SIZE must be greater than 0")
        
        if self.trading.max_position_time_hours <= 0:
            errors.append("MAX_POSITION_TIME_HOURS must be greater than 0")
        
        # Validate data config
        if self.data.retry_interval_minutes < 1:
            errors.append("RETRY_INTERVAL_MINUTES must be at least 1")
        
        if self.data.max_retry_attempts < 1:
            errors.append("MAX_RETRY_ATTEMPTS must be at least 1")
        
        if self.data.initial_data_hours < 1:
            errors.append("INITIAL_DATA_HOURS must be at least 1")
        
        # Validate API keys
        if not self.api_keys['bybit_api_key'] or not self.api_keys['bybit_secret']:
            errors.append("Bybit API keys are required")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as a dictionary"""
        return {
            'symbol': self.symbol,
            'trading': self.trading.__dict__,
            'data': self.data.__dict__,
            'exchange': self.exchange.__dict__,
            'system': self.system.__dict__,
            'api_keys_configured': {
                'bybit': bool(self.api_keys['bybit_api_key'] and self.api_keys['bybit_secret']),
                'binance': bool(self.api_keys['binance_api_key'] and self.api_keys['binance_secret'])
            }
        }
    
    def log_config_summary(self):
        """Log configuration summary"""
        self.logger.info("=== CONFIGURATION SUMMARY ===")
        self.logger.info(f"Symbol: {self.symbol}")
        self.logger.info(f"Z-Score Threshold: ±{self.trading.z_score_threshold}")
        self.logger.info(f"Rolling Window: {self.trading.rolling_window}")
        self.logger.info(f"Position Size: {self.trading.position_size}")
        self.logger.info(f"Max Position Time: {self.trading.max_position_time_hours} hours")
        self.logger.info(f"Fetch on Hour Mark: {self.data.fetch_on_hour_mark}")
        self.logger.info(f"Initial Data Hours: {self.data.initial_data_hours}")
        self.logger.info(f"Bybit Testnet: {self.exchange.bybit_testnet}")
        self.logger.info(f"Trading Enabled: {self.system.enable_trading}")
        self.logger.info(f"Dry Run Mode: {self.system.dry_run_mode}")
        self.logger.info("=============================")
    
    def update_config(self, section: str, key: str, value: Any):
        """
        Update configuration value at runtime
        
        Args:
            section: Configuration section (trading, data, exchange, system)
            key: Configuration key
            value: New value
        """
        config_obj = getattr(self, section)
        if hasattr(config_obj, key):
            old_value = getattr(config_obj, key)
            setattr(config_obj, key, value)
            self.logger.info(f"Updated {section}.{key}: {old_value} -> {value}")
        else:
            raise ValueError(f"Unknown configuration key: {section}.{key}")
    
    def save_to_file(self, filename: str = None):
        """
        Save current configuration to file
        
        Args:
            filename: Output filename (optional)
        """
        import time
        if filename is None:
            filename = f"config_backup_{int(time.time())}.env"
        
        config_dict = self.get_all_config()
        
        # Convert to environment variable format
        env_lines = []
        env_lines.append("# Trading Bot Configuration")
        env_lines.append(f"SYMBOL={self.symbol}")
        
        # Trading config
        env_lines.append("\n# Trading Configuration")
        env_lines.append(f"Z_SCORE_THRESHOLD={self.trading.z_score_threshold}")
        env_lines.append(f"ROLLING_WINDOW={self.trading.rolling_window}")
        env_lines.append(f"POSITION_SIZE={self.trading.position_size}")
        env_lines.append(f"MAX_POSITION_TIME_HOURS={self.trading.max_position_time_hours}")
        env_lines.append(f"MAX_DAILY_TRADES={self.trading.max_daily_trades}")
        env_lines.append(f"MAX_CONSECUTIVE_LOSSES={self.trading.max_consecutive_losses}")
        
        if self.trading.stop_loss_pct is not None:
            env_lines.append(f"STOP_LOSS_PCT={self.trading.stop_loss_pct}")
        if self.trading.take_profit_pct is not None:
            env_lines.append(f"TAKE_PROFIT_PCT={self.trading.take_profit_pct}")
        
        # Data config
        env_lines.append("\n# Data Configuration")
        env_lines.append(f"FETCH_ON_HOUR_MARK={str(self.data.fetch_on_hour_mark).lower()}")
        env_lines.append(f"RETRY_INTERVAL_MINUTES={self.data.retry_interval_minutes}")
        env_lines.append(f"MAX_RETRY_ATTEMPTS={self.data.max_retry_attempts}")
        env_lines.append(f"BINANCE_LS_PERIOD={self.data.binance_long_short_period}")
        env_lines.append(f"INITIAL_DATA_HOURS={self.data.initial_data_hours}")
        env_lines.append(f"MAX_STALE_DATA_TOLERANCE={self.data.max_stale_data_tolerance}")
        env_lines.append(f"DATA_FRESHNESS_THRESHOLD_MINUTES={self.data.data_freshness_threshold_minutes}")
        
        # System config
        env_lines.append("\n# System Configuration")
        env_lines.append(f"LOG_LEVEL={self.system.log_level}")
        env_lines.append(f"ENABLE_TRADING={str(self.system.enable_trading).lower()}")
        env_lines.append(f"DRY_RUN_MODE={str(self.system.dry_run_mode).lower()}")
        env_lines.append(f"BYBIT_TESTNET={str(self.exchange.bybit_testnet).lower()}")
        
        with open(filename, 'w') as f:
            f.write('\n'.join(env_lines))
        
        self.logger.info(f"Configuration saved to {filename}")


# Global configuration instance
config = None

def get_config() -> ConfigManager:
    """Get global configuration instance"""
    global config
    if config is None:
        config = ConfigManager()
    return config

def reload_config():
    """Reload configuration from file"""
    global config
    config = ConfigManager()
    return config


if __name__ == "__main__":
    # Test configuration loading
    import time
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        cfg = ConfigManager()
        cfg.log_config_summary()
        
        print("\nAll Configuration:")
        import json
        print(json.dumps(cfg.get_all_config(), indent=2, default=str))
        
    except Exception as e:
        print(f"Configuration error: {e}")