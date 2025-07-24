#!/usr/bin/env python3
"""
Z-Score Trading Bot for Bybit Futures

This bot trades Bybit futures contracts based on Z-score analysis of combined
Open Interest and Long/Short Ratio indicators from Binance.

Author: Trading Bot
"""

import os
import sys
import time
import logging
import signal
from datetime import datetime, timedelta
from typing import Optional
import ccxt
from dotenv import load_dotenv

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_fetcher'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'trading'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))

from data_fetcher.binance_indicators import BinanceIndicatorsFetcher
from trading.strategy import ZScoreStrategy
from trading.position_manager import PositionManager
from trading.order_manager import OrderManager
from config.settings import ConfigManager


class TradingBot:
    """
    Main trading bot class that orchestrates all components
    """
    
    def __init__(self):
        """Initialize the trading bot"""
        # Initialize configuration first
        self.config = ConfigManager()
        
        # Setup logging based on configuration
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Log configuration summary
        self.config.log_config_summary()
        
        # Initialize components
        self.running = False
        self.binance_fetcher = None
        self.bybit_exchange = None
        self.strategy = None
        self.position_manager = None
        self.order_manager = None
        
        # Performance tracking
        self.start_time = None
        self.total_signals = 0
        self.total_trades = 0
        self.last_trade_time = None
        self.last_hour_check = None
        
        # Strategy state
        self.strategy_initialized = False
        
        self.logger.info("TradingBot initialized")
    
    def setup_logging(self):
        """Setup comprehensive logging based on configuration"""
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Setup logging configuration
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.system.log_level))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # File handler (if enabled)
        if self.config.system.log_to_file:
            file_handler = logging.FileHandler(
                f'logs/trading_bot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            )
            file_handler.setLevel(getattr(logging, self.config.system.log_level))
            file_handler.setFormatter(logging.Formatter(log_format))
            root_logger.addHandler(file_handler)
            print(f"File logging enabled: {file_handler.baseFilename}")
        
        # Console handler (if enabled)
        if self.config.system.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, self.config.system.log_level))
            console_handler.setFormatter(logging.Formatter(log_format))
            root_logger.addHandler(console_handler)
            print("Console logging enabled")
        
        print(f"Logging initialized at {self.config.system.log_level} level")
    

    
    def initialize_components(self):
        """Initialize all trading components"""
        try:
            # Initialize Binance data fetcher
            self.binance_fetcher = BinanceIndicatorsFetcher(
                api_key=self.config.api_keys['binance_api_key'],
                secret=self.config.api_keys['binance_secret'],
                config_manager=self.config
            )
            
            # Initialize Bybit exchange
            self.bybit_exchange = ccxt.bybit({
                'apiKey': self.config.api_keys['bybit_api_key'],
                'secret': self.config.api_keys['bybit_secret'],
                'sandbox': self.config.exchange.bybit_testnet,
                'enableRateLimit': self.config.exchange.bybit_rate_limit,
            })
            
            # Test exchange connection
            balance = self.bybit_exchange.fetch_balance()
            self.logger.info(f"Bybit connection successful. USDT Balance: {balance.get('USDT', {}).get('total', 'N/A')}")
            
            # Initialize strategy (will be populated with historical data later)
            self.strategy = ZScoreStrategy(config_manager=self.config)
            
            # Initialize position manager
            # Convert symbol format for Bybit (BTCUSDT -> BTC/USDT:USDT)
            bybit_symbol = f"{self.config.symbol[:-4]}/USDT:USDT" if self.config.symbol.endswith('USDT') else self.config.symbol
            self.position_manager = PositionManager(self.bybit_exchange, bybit_symbol)
            
            # Initialize order manager
            self.order_manager = OrderManager(self.bybit_exchange, bybit_symbol, self.config.trading.position_size)
            
            self.logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            return False
    
    def initialize_strategy_with_historical_data(self) -> bool:
        """Initialize strategy with historical data"""
        try:
            self.logger.info("Loading initial historical data...")
            
            # Load historical data
            historical_data = self.binance_fetcher.load_initial_historical_data(self.config.symbol)
            
            if not historical_data:
                self.logger.error("Failed to load initial historical data")
                return False
            
            # Initialize strategy with historical data
            success = self.strategy.initialize_with_historical_data(historical_data)
            
            if not success:
                self.logger.error("Failed to initialize strategy with historical data")
                return False
            
            self.strategy_initialized = True
            self.logger.info("Strategy initialized with historical data successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing strategy with historical data: {e}")
            return False
    
    def should_fetch_new_data(self) -> bool:
        """Check if we should fetch new data based on timing configuration"""
        current_time = datetime.now()
        
        # If not fetching on hour marks, use regular interval
        if not self.config.data.fetch_on_hour_mark:
            return True  # Let the main loop handle timing
        
        # Check if we're at a new hour
        current_hour = current_time.replace(minute=0, second=0, microsecond=0)
        
        if self.last_hour_check is None or self.last_hour_check < current_hour:
            self.last_hour_check = current_hour
            return True
        
        return False
    
    def fetch_and_process_data(self) -> bool:
        """Fetch data and process through strategy"""
        try:
            # Check if we should fetch new data
            if not self.should_fetch_new_data():
                return True  # Not time to fetch yet, but not an error
            
            # Fetch combined indicator with timestamp
            result = self.binance_fetcher.get_combined_indicator_with_timestamp(self.config.symbol)
            
            if result is None:
                self.logger.warning("Could not fetch combined indicator with timestamp")
                return False
            
            combined_indicator, timestamp = result
            
            # Add to strategy
            success = self.strategy.add_indicator_value(combined_indicator, timestamp)
            
            if not success:
                self.logger.warning("Failed to add indicator value to strategy")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error fetching and processing data: {e}")
            return False
    
    def execute_trading_logic(self):
        """Execute the main trading logic"""
        try:
            # Update position information
            self.position_manager.update_position_from_exchange()
            
            # Check if strategy is ready
            if not self.strategy.is_ready():
                self.logger.info(f"Strategy not ready yet. Data points: {len(self.strategy.indicator_history)}/{self.rolling_window}")
                return
            
            # Generate signal
            signal = self.strategy.generate_signal()
            
            if signal is None:
                self.logger.warning("Could not generate signal")
                return
            
            self.total_signals += 1
            
            # Execute trades based on signal
            if signal == 'LONG' and self.position_manager.can_open_long():
                self.logger.info("LONG signal received - Opening long position")
                order = self.order_manager.open_long_position()
                if order:
                    self.total_trades += 1
                    self.last_trade_time = datetime.now()
                    self.logger.info(f"Long position opened successfully. Order ID: {order['id']}")
                else:
                    self.logger.error("Failed to open long position")
            
            elif signal == 'SHORT' and self.position_manager.can_open_short():
                self.logger.info("SHORT signal received - Opening short position")
                order = self.order_manager.open_short_position()
                if order:
                    self.total_trades += 1
                    self.last_trade_time = datetime.now()
                    self.logger.info(f"Short position opened successfully. Order ID: {order['id']}")
                else:
                    self.logger.error("Failed to open short position")
            
            elif signal == 'HOLD':
                self.logger.debug("HOLD signal - No action taken")
            
            # Check for position management (stop loss, take profit, max time)
            self.manage_existing_position()
            
        except Exception as e:
            self.logger.error(f"Error in trading logic: {e}")
    
    def manage_existing_position(self):
        """Manage existing positions (time-based exit, etc.)"""
        try:
            if not self.position_manager.is_in_position():
                return
            
                    # Check maximum position time
        max_position_time = timedelta(hours=self.config.trading.max_position_time_hours)
        if (self.position_manager.entry_time and 
            datetime.now() - self.position_manager.entry_time > max_position_time):
                
                self.logger.info(f"Position held for too long, closing position")
                
                if self.position_manager.is_long():
                    order = self.order_manager.close_long_position()
                elif self.position_manager.is_short():
                    order = self.order_manager.close_short_position()
                
                if order:
                    self.logger.info(f"Position closed due to time limit. Order ID: {order['id']}")
                else:
                    self.logger.error("Failed to close position due to time limit")
            
        except Exception as e:
            self.logger.error(f"Error managing existing position: {e}")
    
    def log_status(self):
        """Log current bot status"""
        try:
            uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
            
            self.logger.info("=== BOT STATUS ===")
            self.logger.info(f"Uptime: {uptime}")
            self.logger.info(f"Total Signals: {self.total_signals}")
            self.logger.info(f"Total Trades: {self.total_trades}")
            self.logger.info(f"Last Trade: {self.last_trade_time or 'None'}")
            
            # Position status
            self.position_manager.log_position_status()
            
            # Strategy stats
            strategy_stats = self.strategy.get_strategy_stats()
            self.logger.info(f"Strategy Stats: {strategy_stats}")
            
            # Latest Z-score
            latest_z = self.strategy.get_latest_z_score()
            if latest_z is not None:
                self.logger.info(f"Latest Z-Score: {latest_z:.4f}")
            
            self.logger.info("==================")
            
        except Exception as e:
            self.logger.error(f"Error logging status: {e}")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}. Shutting down gracefully...")
        self.running = False
    
    def run(self):
        """Main bot execution loop"""
        try:
            self.logger.info("Starting trading bot...")
            
            # Initialize components
            if not self.initialize_components():
                self.logger.error("Failed to initialize components. Exiting.")
                return False
            
            # Initialize strategy with historical data
            if not self.initialize_strategy_with_historical_data():
                self.logger.error("Failed to initialize strategy with historical data. Exiting.")
                return False
            
            # Setup signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            
            self.running = True
            self.start_time = datetime.now()
            
            last_data_fetch = datetime.min
            last_status_log = datetime.min
            
            self.logger.info("Trading bot started successfully and ready for trading")
            
            while self.running:
                current_time = datetime.now()
                
                try:
                    # Handle data fetching based on configuration
                    should_fetch = False
                    
                    if self.config.data.fetch_on_hour_mark:
                        # Only fetch at hour marks
                        if self.should_fetch_new_data():
                            should_fetch = True
                    else:
                        # Fetch at regular intervals (every 60 seconds by default)
                        if (current_time - last_data_fetch).total_seconds() >= 60:
                            should_fetch = True
                    
                    if should_fetch:
                        self.logger.debug("Fetching new data...")
                        if self.fetch_and_process_data():
                            last_data_fetch = current_time
                            
                            # Execute trading logic after data update
                            if self.config.system.enable_trading:
                                self.execute_trading_logic()
                            else:
                                self.logger.info("Trading disabled in configuration")
                    
                    # Log status at configured intervals
                    status_interval = self.config.system.status_log_interval_minutes * 60
                    if (current_time - last_status_log).total_seconds() >= status_interval:
                        self.log_status()
                        last_status_log = current_time
                    
                    # Sleep for a short time to prevent excessive CPU usage
                    sleep_time = 60 if self.config.data.fetch_on_hour_mark else 5
                    time.sleep(sleep_time)
                    
                except KeyboardInterrupt:
                    self.logger.info("Keyboard interrupt received. Shutting down...")
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    time.sleep(10)  # Wait before retrying
            
            self.logger.info("Trading bot stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Critical error in bot execution: {e}")
            return False
    
    def shutdown(self):
        """Shutdown the bot gracefully"""
        self.logger.info("Shutting down trading bot...")
        self.running = False
        
        # Final status log
        self.log_status()
        
        # Log final summary
        if self.order_manager:
            self.order_manager.log_order_summary()
        
        self.logger.info("Trading bot shutdown complete")


def main():
    """Main entry point"""
    print("Z-Score Trading Bot")
    print("===================")
    
    try:
        bot = TradingBot()
        success = bot.run()
        
        if not success:
            print("Bot execution failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting...")
    except Exception as e:
        print(f"Critical error: {e}")
        sys.exit(1)
    
    print("Bot execution completed")


if __name__ == "__main__":
    main()