import ccxt
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import time


class OrderManager:
    """
    Manages order execution for Bybit futures trading
    """
    
    def __init__(self, exchange: ccxt.Exchange, symbol: str, position_size: float):
        """
        Initialize order manager
        
        Args:
            exchange: CCXT exchange instance
            symbol: Trading symbol (e.g., 'BTC/USDT:USDT')
            position_size: Position size in contracts
        """
        self.logger = logging.getLogger(__name__)
        self.exchange = exchange
        self.symbol = symbol
        self.position_size = position_size
        
        # Order tracking
        self.last_order = None
        self.order_history = []
        
        self.logger.info(f"Initialized OrderManager for {symbol} with position size {position_size}")
    
    def create_market_order(self, side: str, amount: float, reduce_only: bool = False) -> Optional[Dict[str, Any]]:
        """
        Create a market order
        
        Args:
            side: 'buy' or 'sell'
            amount: Order amount in contracts
            reduce_only: Whether this is a reduce-only order
            
        Returns:
            Order information or None if failed
        """
        try:
            self.logger.info(f"Creating {side} market order: {amount} contracts, reduce_only={reduce_only}")
            
            # Prepare order parameters
            params = {}
            if reduce_only:
                params['reduceOnly'] = True
            
            # Create the order
            order = self.exchange.create_market_order(
                symbol=self.symbol,
                side=side,
                amount=amount,
                params=params
            )
            
            self.last_order = order
            self.order_history.append(order)
            
            self.logger.info(f"Order created successfully: {order['id']}")
            self.logger.info(f"Order details: Side={order['side']}, Amount={order['amount']}, "
                           f"Status={order['status']}")
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error creating {side} market order: {e}")
            return None
    
    def open_long_position(self) -> Optional[Dict[str, Any]]:
        """
        Open a long position
        
        Returns:
            Order information or None if failed
        """
        self.logger.info(f"Opening LONG position with {self.position_size} contracts")
        return self.create_market_order('buy', self.position_size)
    
    def open_short_position(self) -> Optional[Dict[str, Any]]:
        """
        Open a short position
        
        Returns:
            Order information or None if failed
        """
        self.logger.info(f"Opening SHORT position with {self.position_size} contracts")
        return self.create_market_order('sell', self.position_size)
    
    def close_long_position(self, amount: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Close a long position
        
        Args:
            amount: Amount to close (defaults to position_size)
            
        Returns:
            Order information or None if failed
        """
        close_amount = amount or self.position_size
        self.logger.info(f"Closing LONG position with {close_amount} contracts")
        return self.create_market_order('sell', close_amount, reduce_only=True)
    
    def close_short_position(self, amount: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Close a short position
        
        Args:
            amount: Amount to close (defaults to position_size)
            
        Returns:
            Order information or None if failed
        """
        close_amount = amount or self.position_size
        self.logger.info(f"Closing SHORT position with {close_amount} contracts")
        return self.create_market_order('buy', close_amount, reduce_only=True)
    
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get order status
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order status or None if failed
        """
        try:
            order = self.exchange.fetch_order(order_id, self.symbol)
            self.logger.info(f"Order {order_id} status: {order['status']}")
            return order
            
        except Exception as e:
            self.logger.error(f"Error fetching order status for {order_id}: {e}")
            return None
    
    def wait_for_order_completion(self, order_id: str, timeout: int = 30) -> bool:
        """
        Wait for order to complete
        
        Args:
            order_id: Order ID to wait for
            timeout: Timeout in seconds
            
        Returns:
            True if order completed, False if timeout or error
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                order = self.get_order_status(order_id)
                
                if order is None:
                    self.logger.error(f"Could not fetch order status for {order_id}")
                    return False
                
                if order['status'] in ['closed', 'filled']:
                    self.logger.info(f"Order {order_id} completed successfully")
                    return True
                elif order['status'] in ['canceled', 'rejected']:
                    self.logger.error(f"Order {order_id} was {order['status']}")
                    return False
                
                # Wait a bit before checking again
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error waiting for order completion: {e}")
                return False
        
        self.logger.error(f"Timeout waiting for order {order_id} to complete")
        return False
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        try:
            self.logger.info(f"Cancelling order {order_id}")
            self.exchange.cancel_order(order_id, self.symbol)
            self.logger.info(f"Order {order_id} cancelled successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_recent_trades(self, limit: int = 10) -> list:
        """
        Get recent trades
        
        Args:
            limit: Number of recent trades to fetch
            
        Returns:
            List of recent trades
        """
        try:
            trades = self.exchange.fetch_my_trades(self.symbol, limit=limit)
            self.logger.info(f"Fetched {len(trades)} recent trades")
            return trades
            
        except Exception as e:
            self.logger.error(f"Error fetching recent trades: {e}")
            return []
    
    def get_last_order_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the last order
        
        Returns:
            Last order information or None
        """
        return self.last_order
    
    def get_order_history(self, n: int = 10) -> list:
        """
        Get order history
        
        Args:
            n: Number of recent orders to return
            
        Returns:
            List of recent orders
        """
        return self.order_history[-n:] if self.order_history else []
    
    def log_order_summary(self):
        """
        Log a summary of recent orders
        """
        if not self.order_history:
            self.logger.info("No orders in history")
            return
        
        self.logger.info(f"Order History Summary ({len(self.order_history)} orders):")
        for order in self.order_history[-5:]:  # Show last 5 orders
            self.logger.info(f"  {order['datetime']}: {order['side'].upper()} "
                           f"{order['amount']} @ {order.get('price', 'MARKET')} "
                           f"Status: {order['status']}")


def test_order_manager():
    """Test function for order manager (requires valid API keys)"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize exchange with testnet
    exchange = ccxt.bybit({
        'apiKey': os.getenv('BYBIT_API_KEY', ''),
        'secret': os.getenv('BYBIT_SECRET', ''),
        'sandbox': True,  # Use testnet for testing
        'enableRateLimit': True,
    })
    
    # Test order manager
    om = OrderManager(exchange, 'BTC/USDT:USDT', 0.001)
    
    # Note: These would create actual orders on testnet
    # Uncomment to test (make sure you have testnet API keys)
    
    # Test opening a position
    # order = om.open_long_position()
    # if order:
    #     print(f"Order created: {order['id']}")
    #     
    #     # Wait for completion
    #     if om.wait_for_order_completion(order['id']):
    #         print("Order completed successfully")
    #     else:
    #         print("Order did not complete")
    
    # Get recent trades
    trades = om.get_recent_trades()
    print(f"Recent trades: {len(trades)}")
    
    om.log_order_summary()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_order_manager()