import logging
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
import ccxt


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"
    NONE = "none"


class PositionManager:
    """
    Manages trading positions and keeps track of current position state
    """
    
    def __init__(self, exchange: ccxt.Exchange, symbol: str):
        """
        Initialize position manager
        
        Args:
            exchange: CCXT exchange instance
            symbol: Trading symbol (e.g., 'BTC/USDT:USDT')
        """
        self.logger = logging.getLogger(__name__)
        self.exchange = exchange
        self.symbol = symbol
        
        # Position tracking
        self.current_position_side = PositionSide.NONE
        self.current_position_size = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.last_update_time = None
        
        # Initialize position state
        self.update_position_from_exchange()
    
    def update_position_from_exchange(self) -> bool:
        """
        Update position information from the exchange
        
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Fetch position from exchange
            positions = self.exchange.fetch_positions([self.symbol])
            
            if not positions:
                self.logger.warning(f"No position data found for {self.symbol}")
                return False
            
            position = positions[0]  # Get the first (and should be only) position
            
            # Update position information
            contracts = float(position['contracts']) if position['contracts'] else 0.0
            side = position['side']
            
            if contracts == 0 or side is None:
                self.current_position_side = PositionSide.NONE
                self.current_position_size = 0.0
                self.entry_price = 0.0
                self.entry_time = None
            else:
                self.current_position_side = PositionSide.LONG if side == 'long' else PositionSide.SHORT
                self.current_position_size = abs(contracts)
                self.entry_price = float(position['entryPrice']) if position['entryPrice'] else 0.0
                
                # Try to get entry time if available
                if position.get('timestamp'):
                    self.entry_time = datetime.fromtimestamp(position['timestamp'] / 1000)
            
            self.last_update_time = datetime.now()
            
            self.logger.info(f"Position updated: {self.current_position_side.value}, "
                           f"Size: {self.current_position_size}, "
                           f"Entry Price: {self.entry_price}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating position from exchange: {e}")
            return False
    
    def is_in_position(self) -> bool:
        """
        Check if currently in a position
        
        Returns:
            True if in a position, False otherwise
        """
        return self.current_position_side != PositionSide.NONE and self.current_position_size > 0
    
    def is_long(self) -> bool:
        """
        Check if currently in a long position
        
        Returns:
            True if in long position, False otherwise
        """
        return self.current_position_side == PositionSide.LONG and self.current_position_size > 0
    
    def is_short(self) -> bool:
        """
        Check if currently in a short position
        
        Returns:
            True if in short position, False otherwise
        """
        return self.current_position_side == PositionSide.SHORT and self.current_position_size > 0
    
    def can_open_long(self) -> bool:
        """
        Check if can open a long position
        
        Returns:
            True if can open long, False otherwise
        """
        return not self.is_in_position()
    
    def can_open_short(self) -> bool:
        """
        Check if can open a short position
        
        Returns:
            True if can open short, False otherwise
        """
        return not self.is_in_position()
    
    def can_close_position(self) -> bool:
        """
        Check if can close current position
        
        Returns:
            True if can close position, False otherwise
        """
        return self.is_in_position()
    
    def get_position_info(self) -> Dict[str, Any]:
        """
        Get current position information
        
        Returns:
            Dictionary with position information
        """
        return {
            'side': self.current_position_side.value,
            'size': self.current_position_size,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'last_update': self.last_update_time.isoformat() if self.last_update_time else None,
            'in_position': self.is_in_position()
        }
    
    def log_position_status(self):
        """
        Log current position status
        """
        if self.is_in_position():
            self.logger.info(f"Current Position: {self.current_position_side.value.upper()} "
                           f"{self.current_position_size} contracts at {self.entry_price}")
        else:
            self.logger.info("No current position")
    
    def reset_position(self):
        """
        Reset position tracking (use with caution)
        """
        self.logger.warning("Resetting position tracking")
        self.current_position_side = PositionSide.NONE
        self.current_position_size = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.last_update_time = datetime.now()


def test_position_manager():
    """Test function for position manager"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize exchange (you'll need to set up your API keys)
    exchange = ccxt.bybit({
        'apiKey': os.getenv('BYBIT_API_KEY', ''),
        'secret': os.getenv('BYBIT_SECRET', ''),
        'sandbox': True,  # Use testnet for testing
        'enableRateLimit': True,
    })
    
    # Test position manager
    pm = PositionManager(exchange, 'BTC/USDT:USDT')
    pm.log_position_status()
    
    print(f"In position: {pm.is_in_position()}")
    print(f"Can open long: {pm.can_open_long()}")
    print(f"Can open short: {pm.can_open_short()}")
    print(f"Position info: {pm.get_position_info()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_position_manager()