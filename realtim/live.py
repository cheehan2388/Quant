import os
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import ccxt
from dotenv import load_dotenv

load_dotenv()
# -------------------- Configuration --------------------
# SECURITY FIX: Use proper environment variable names
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

# Validate API credentials
if not all([BINANCE_API_KEY, BINANCE_API_SECRET, BYBIT_API_KEY, BYBIT_API_SECRET]):
    raise ValueError("Missing API credentials. Please set environment variables: BINANCE_API_KEY, BINANCE_API_SECRET, BYBIT_API_KEY, BYBIT_API_SECRET")

# Symbol definitions
PAIR = "BTCUSDT"               # Binance/Bybit symbol
INTERVAL = "1h"                # Data interval
ROLL_WINDOW = 20               # Increased for better statistical significance
TRADE_AMOUNT = 0.001           # Amount in BTC per trade
LEVERAGE = 1.5                 # Desired leverage
ZSCORE_THRESHOLD = 2.0         # Z-score threshold for signals
MIN_STD_THRESHOLD = 0.0001     # Minimum standard deviation to avoid division by zero

# -------------------- Exchange Setup --------------------
def initialize_exchanges():
    """Initialize and validate exchange connections"""
    try:
        binance = ccxt.binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_API_SECRET,
            'enableRateLimit': True,
            'sandbox': False,  # Set to True for testing
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
            }
        })
        
        bybit = ccxt.bybit({
            'apiKey': BYBIT_API_KEY,
            'secret': BYBIT_API_SECRET,
            'enableRateLimit': True,
            'sandbox': False,  # Set to True for testing
            'options': {
                'defaultType': 'swap',
                'adjustForTimeDifference': True
            }
        })
        
        # Test connections and load markets
        binance.load_markets()
        bybit.load_markets()
        
        # Validate symbols exist
        if PAIR not in binance.markets:
            raise ValueError(f"Symbol {PAIR} not found in Binance markets")
        if f"BTC/USDT:USDT" not in bybit.markets:
            raise ValueError(f"Symbol BTC/USDT:USDT not found in Bybit markets")
            
        logger.info("Exchanges initialized successfully")
        return binance, bybit
        
    except Exception as e:
        logger.error("Failed to initialize exchanges: %s", e)
        raise

# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------- Data Fetch Functions --------------------
def fetch_open_interest(binance_exchange, limit=ROLL_WINDOW):
    """
    Fetch recent open interest data from Binance futures.
    Returns a DataFrame with ['timestamp','openInterest']
    """
    try:
        # FIXED: Use correct API method name
        data = binance_exchange.fapiPublicGetOpenInterestHist({
            'symbol': PAIR,
            'period': INTERVAL,
            'limit': limit
        })
        
        if not data:
            raise ValueError("No open interest data received")
            
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['openInterest'] = df['sumOpenInterest'].astype(float)
        
        logger.info("Fetched %d open interest records", len(df))
        return df[['timestamp', 'openInterest']]
        
    except Exception as e:
        logger.error("Error fetching open interest: %s", e)
        raise


def fetch_long_short_ratio(binance_exchange, limit=ROLL_WINDOW):
    """
    Fetch recent global long/short account ratio from Binance futures.
    Returns a DataFrame with ['timestamp','longShortRatio']
    """
    try:
        # FIXED: Use correct API method name
        data = binance_exchange.fapiPublicGetTopLongShortPositionRatio({
            'symbol': PAIR,
            'period': INTERVAL,
            'limit': limit
        })
        
        if not data:
            raise ValueError("No long/short ratio data received")
            
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['longShortRatio'] = df['longShortRatio'].astype(float)
        
        logger.info("Fetched %d long/short ratio records", len(df))
        return df[['timestamp', 'longShortRatio']]
        
    except Exception as e:
        logger.error("Error fetching long/short ratio: %s", e)
        raise


def get_aligned_data(binance_exchange, limit=ROLL_WINDOW, max_retries=5):
    """
    Ensure open interest and ratio data share the same latest timestamps.
    Returns merged DataFrame of length <= limit.
    """
    retry_count = 0
    while retry_count < max_retries:
        try:
            df_oi = fetch_open_interest(binance_exchange, limit)
            df_ratio = fetch_long_short_ratio(binance_exchange, limit)
            
            # Check if we have enough data
            if len(df_oi) == 0 or len(df_ratio) == 0:
                raise ValueError("No data received from APIs")
            
            ts_oi = df_oi['timestamp'].iloc[-1]
            ts_ratio = df_ratio['timestamp'].iloc[-1]
            
            # Allow small time differences (within 1 hour)
            time_diff = abs((ts_oi - ts_ratio).total_seconds())
            if time_diff <= 60:  # 1 hour tolerance
                df = pd.merge(df_oi, df_ratio, on='timestamp', how='inner')
                logger.info("Data aligned successfully with %d records", len(df))
                return df
                
            logger.warning(
                "Timestamp mismatch (OI: %s, Ratio: %s, diff: %.0fs). Retry %d/%d in 30s...",
                ts_oi, ts_ratio, time_diff, retry_count + 1, max_retries
            )
            time.sleep(30)  # Reduced wait time
            retry_count += 1
            
        except Exception as e:
            logger.error("Error in get_aligned_data (retry %d/%d): %s", retry_count + 1, max_retries, e)
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(30)
    
    raise RuntimeError(f"Failed to get aligned data after {max_retries} retries")

# -------------------- Trading Functions --------------------
class PositionManager:
    def __init__(self):
        self.current_position = None  # 'long', 'short', or None
        self.position_size = 0.0
        self.entry_price = 0.0
        
    def get_position(self):
        return self.current_position
        
    def set_position(self, position_type, size=TRADE_AMOUNT, price=0.0):
        self.current_position = position_type
        self.position_size = size
        self.entry_price = price

position_manager = PositionManager()


def set_leverage(bybit_exchange):
    """Set leverage on Bybit for the symbol."""
    try:
        response = bybit_exchange.private_post_v5_position_set_leverage({
            'category': 'linear',
            'symbol': PAIR,
            'buyLeverage': str(LEVERAGE),
            'sellLeverage': str(LEVERAGE)
        })
        logger.info("Leverage set to %sx: %s", LEVERAGE, response)
        return True
    except Exception as e:
        logger.error("Failed to set leverage: %s", e)
        return False


def get_current_price(bybit_exchange, symbol="BTC/USDT:USDT"):
    """Get current market price"""
    try:
        ticker = bybit_exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        logger.error("Failed to get current price: %s", e)
        return None


def open_position(bybit_exchange, direction):
    """Open a market position on Bybit: 'long' or 'short'."""
    market_symbol = "BTC/USDT:USDT"
    
    try:
        current_price = get_current_price(bybit_exchange, market_symbol)
        if not current_price:
            raise ValueError("Could not get current price")
            
        if direction == 'long':
            order = bybit_exchange.create_market_buy_order(market_symbol, TRADE_AMOUNT)
        else:
            order = bybit_exchange.create_market_sell_order(market_symbol, TRADE_AMOUNT)
            
        logger.info("Opened %s position: %f BTC at ~$%.2f - Order: %s", 
                   direction, TRADE_AMOUNT, current_price, order['id'])
        
        position_manager.set_position(direction, TRADE_AMOUNT, current_price)
        return True
        
    except Exception as e:
        logger.error("Failed to open %s position: %s", direction, e)
        return False


def close_position(bybit_exchange):
    """Close the current position on Bybit."""
    current_pos = position_manager.get_position()
    if not current_pos:
        logger.info("No position to close")
        return True
        
    market_symbol = "BTC/USDT:USDT"
    
    try:
        current_price = get_current_price(bybit_exchange, market_symbol)
        if not current_price:
            raise ValueError("Could not get current price")
            
        if current_pos == 'long':
            order = bybit_exchange.create_market_sell_order(market_symbol, TRADE_AMOUNT)
        else:
            order = bybit_exchange.create_market_buy_order(market_symbol, TRADE_AMOUNT)
            
        logger.info("Closed %s position: %f BTC at ~$%.2f - Order: %s", 
                   current_pos, TRADE_AMOUNT, current_price, order['id'])
        
        position_manager.set_position(None, 0, 0)
        return True
        
    except Exception as e:
        logger.error("Failed to close %s position: %s", current_pos, e)
        return False

# -------------------- Strategy Functions --------------------
def calculate_signals(df):
    """Calculate trading signals based on the strategy"""
    if len(df) < ROLL_WINDOW:
        return None, 0, 0, 0
        
    # Compute metric and Z-score
    df = df.copy()
    df['metric'] = df['openInterest'] * df['longShortRatio']
    
    # Use rolling window for statistics
    last_window = df.tail(ROLL_WINDOW)
    mu = last_window['metric'].mean()
    sigma = last_window['metric'].std()
    latest = last_window['metric'].iloc[-1]
    
    # Avoid division by zero
    if sigma < MIN_STD_THRESHOLD:
        logger.warning("Standard deviation too low (%.6f), no signal generated", sigma)
        return None, latest, mu, sigma
        
    zscore = (latest - mu) / sigma
    
    # Generate signals
    if zscore > ZSCORE_THRESHOLD:
        signal = 'short'  # Contrarian: high ratio suggests bearish
    elif zscore < -ZSCORE_THRESHOLD:
        signal = 'long'   # Contrarian: low ratio suggests bullish
    else:
        signal = None
        
    return signal, latest, mu, sigma

# -------------------- Main Strategy Loop --------------------
def main():
    """Main trading loop"""
    logger.info("Starting CCXT Trading Bot")
    
    try:
        # Initialize exchanges
        binance, bybit = initialize_exchanges()
        
        # Set leverage
        if not set_leverage(bybit):
            logger.error("Failed to set leverage, exiting")
            return
            
        # Initialize rolling window using historical data
        logger.info("Initializing rolling window with %d data points...", ROLL_WINDOW)
        
        while True:
            try:
                df_history = get_aligned_data(binance, limit=ROLL_WINDOW)
                if len(df_history) >= ROLL_WINDOW:
                    logger.info("Rolling window initialized with %d points", len(df_history))
                    break
                else:
                    logger.info("Only %d/%d data points available. Waiting 10 minutes...", 
                              len(df_history), ROLL_WINDOW)
                    time.sleep(600)  # Wait 10 minutes
            except Exception as e:
                logger.error("Error during initialization: %s", e)
                time.sleep(300)  # Wait 5 minutes before retry

        # Main trading loop
        while True:
            try:
                # Wait until a few minutes past the next hour for fresh data
                now = datetime.utcnow()
                next_hour = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
                wait_seconds = (next_hour - now).total_seconds() + 180  # 3 minutes buffer
                
                if wait_seconds > 0:
                    logger.info("Waiting %.0f seconds until next data window...", wait_seconds)
                    time.sleep(wait_seconds)

                # Fetch and analyze latest data
                df = get_aligned_data(binance, limit=ROLL_WINDOW + 5)  # Get a few extra points
                signal, latest_metric, mean_metric, std_metric = calculate_signals(df)
                
                if len(df) > 0:
                    timestamp = df['timestamp'].iloc[-1]
                    zscore = (latest_metric - mean_metric) / std_metric if std_metric > MIN_STD_THRESHOLD else 0
                    
                    logger.info(
                        "Analysis at %s: Metric=%.4f, Mean=%.4f, Std=%.4f, Z-score=%.4f, Signal=%s",
                        timestamp, latest_metric, mean_metric, std_metric, zscore, signal or "HOLD"
                    )

                # Execute trading logic
                current_pos = position_manager.get_position()
                
                if signal == 'short' and current_pos != 'short':
                    logger.info("Signal: SHORT - Current position: %s", current_pos or "None")
                    if current_pos:
                        close_position(bybit)
                        time.sleep(2)  # Brief pause between close and open
                    open_position(bybit, 'short')
                    
                elif signal == 'long' and current_pos != 'long':
                    logger.info("Signal: LONG - Current position: %s", current_pos or "None")
                    if current_pos:
                        close_position(bybit)
                        time.sleep(2)  # Brief pause between close and open
                    open_position(bybit, 'long')
                    
                else:
                    logger.info("No action needed. Signal: %s, Position: %s", 
                              signal or "HOLD", current_pos or "None")

            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.error("Error in main loop: %s", e)
                time.sleep(60)  # Wait 1 minute before continuing

    except Exception as e:
        logger.error("Critical error in main: %s", e)
        raise
    finally:
        # Cleanup
        logger.info("Bot shutting down")

if __name__ == "__main__":
    main()