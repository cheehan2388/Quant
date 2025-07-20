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

from data_fetcher.binance_indicators import BinanceIndicatorsFetcher
from trading.strategy import ZScoreStrategy
from trading.position_manager import PositionManager
from trading.order_manager import OrderManager


class TradingBot:
    """
    Main trading bot class that orchestrates all components
    """
    
    def __init__(self):
        """Initialize the trading bot"""
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.load_config()
        
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
        
        self.logger.info("TradingBot initialized")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Setup logging configuration
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # File handler
        file_handler = logging.FileHandler(
            f'logs/trading_bot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # Root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        print(f"Logging initialized. Log file: {file_handler.baseFilename}")
    
    def load_config(self):
        """Load configuration from environment variables"""
        load_dotenv()
        
        # API Keys
        self.bybit_api_key = os.getenv('BYBIT_API_KEY')
        self.bybit_secret = os.getenv('BYBIT_SECRET')
        self.binance_api_key = os.getenv('BINANCE_API_KEY', '')
        self.binance_secret = os.getenv('BINANCE_SECRET', '')
        
        # Trading parameters
        self.symbol = os.getenv('SYMBOL', 'BTCUSDT')
        self.position_size = float(os.getenv('POSITION_SIZE', '0.001'))
        self.z_threshold = float(os.getenv('Z_SCORE_THRESHOLD', '2.1'))
        self.rolling_window = int(os.getenv('ROLLING_WINDOW', '15'))
        
        # Operational parameters
        self.data_fetch_interval = 60  # seconds
        self.position_check_interval = 30  # seconds
        self.max_position_time = 3600  # 1 hour max position time
        
        # Validate required configuration
        if not self.bybit_api_key or not self.bybit_secret:
            raise ValueError("Bybit API keys are required")
        
        self.logger.info(f"Configuration loaded:")
        self.logger.info(f"  Symbol: {self.symbol}")
        self.logger.info(f"  Position Size: {self.position_size}")
        self.logger.info(f"  Z-Score Threshold: ±{self.z_threshold}")
        self.logger.info(f"  Rolling Window: {self.rolling_window}")
    
    def initialize_components(self):
        """Initialize all trading components"""
        try:
            # Initialize Binance data fetcher
            self.binance_fetcher = BinanceIndicatorsFetcher(
                api_key=self.binance_api_key,
                secret=self.binance_secret
            )
            
            # Initialize Bybit exchange
            self.bybit_exchange = ccxt.bybit({
                'apiKey': self.bybit_api_key,
                'secret': self.bybit_secret,
                'sandbox': False,  # Set to True for testnet
                'enableRateLimit': True,
            })
            
            # Test exchange connection
            balance = self.bybit_exchange.fetch_balance()
            self.logger.info(f"Bybit connection successful. USDT Balance: {balance.get('USDT', {}).get('total', 'N/A')}")
            
            # Initialize strategy
            self.strategy = ZScoreStrategy(
                rolling_window=self.rolling_window,
                z_threshold=self.z_threshold
            )
            
            # Initialize position manager
            # Convert symbol format for Bybit (BTCUSDT -> BTC/USDT:USDT)
            bybit_symbol = f"{self.symbol[:-4]}/USDT:USDT" if self.symbol.endswith('USDT') else self.symbol
            self.position_manager = PositionManager(self.bybit_exchange, bybit_symbol)
            
            # Initialize order manager
            self.order_manager = OrderManager(self.bybit_exchange, bybit_symbol, self.position_size)
            
            self.logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            return False
    
    def fetch_and_process_data(self) -> bool:
        """Fetch data and process through strategy"""
        try:
            # Fetch combined indicator
            combined_indicator = self.binance_fetcher.get_combined_indicator(self.symbol)
            
            if combined_indicator is None:
                self.logger.warning("Could not fetch combined indicator")
                return False
            
            # Add to strategy
            success = self.strategy.add_indicator_value(combined_indicator)
            
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
            if (self.position_manager.entry_time and 
                datetime.now() - self.position_manager.entry_time > timedelta(seconds=self.max_position_time)):
                
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
            
            # Setup signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            
            self.running = True
            self.start_time = datetime.now()
            
            last_data_fetch = datetime.min
            last_status_log = datetime.min
            
            self.logger.info("Trading bot started successfully")
            
            while self.running:
                current_time = datetime.now()
                
                try:
                    # Fetch data at specified intervals
                    if (current_time - last_data_fetch).total_seconds() >= self.data_fetch_interval:
                        self.logger.debug("Fetching new data...")
                        if self.fetch_and_process_data():
                            last_data_fetch = current_time
                            
                            # Execute trading logic after data update
                            self.execute_trading_logic()
                    
                    # Log status every 10 minutes
                    if (current_time - last_status_log).total_seconds() >= 600:
                        self.log_status()
                        last_status_log = current_time
                    
                    # Sleep for a short time to prevent excessive CPU usage
                    time.sleep(5)
                    
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