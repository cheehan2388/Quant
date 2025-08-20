"""
Trader module for executing trades on various exchanges.
Handles order management, position tracking, and risk management.
"""

import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import asyncio
from strategy_v2 import TradingSignal, SignalType

logger = logging.getLogger(__name__)

class Position:
    """Represents a trading position"""
    
    def __init__(self, symbol: str, side: str, size: float, entry_price: float, 
                 timestamp: datetime = None):
        self.symbol = symbol
        self.side = side  # 'long' or 'short'
        self.size = abs(size)
        self.entry_price = entry_price
        self.timestamp = timestamp or datetime.now()
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
    def update_pnl(self, current_price: float):
        """Update unrealized PnL based on current price"""
        if self.side == 'long':
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        else:  # short
            self.unrealized_pnl = (self.entry_price - current_price) * self.size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary"""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'size': self.size,
            'entry_price': self.entry_price,
            'timestamp': self.timestamp.isoformat(),
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl
        }
    
    def __str__(self):
        return f"{self.side.upper()} {self.size} {self.symbol} @ {self.entry_price} (PnL: {self.unrealized_pnl:.2f})"

class RiskManager:
    """Risk management for trading operations"""
    
    def __init__(self, max_position_size: float = 0.1, stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.10, max_daily_loss: float = 0.20):
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_daily_loss = max_daily_loss
        self.daily_pnl = 0.0
        self.daily_reset_time = datetime.now().date()
        
    def check_position_size(self, requested_size: float) -> float:
        """Check and adjust position size based on risk limits"""
        return min(abs(requested_size), self.max_position_size)
    
    def should_stop_trading(self) -> bool:
        """Check if trading should be stopped due to daily loss limit"""
        current_date = datetime.now().date()
        
        # Reset daily PnL if it's a new day
        if current_date != self.daily_reset_time:
            self.daily_pnl = 0.0
            self.daily_reset_time = current_date
        
        return self.daily_pnl <= -self.max_daily_loss
    
    def update_daily_pnl(self, pnl: float):
        """Update daily PnL tracking"""
        self.daily_pnl += pnl
    
    def should_close_position(self, position: Position, current_price: float) -> bool:
        """Check if position should be closed based on stop loss or take profit"""
        if not position:
            return False
        
        position.update_pnl(current_price)
        pnl_pct = position.unrealized_pnl / (position.entry_price * position.size)
        
        # Stop loss check
        if pnl_pct <= -self.stop_loss_pct:
            logger.warning(f"Stop loss triggered for {position}: {pnl_pct:.2%}")
            return True
        
        # Take profit check
        if pnl_pct >= self.take_profit_pct:
            logger.info(f"Take profit triggered for {position}: {pnl_pct:.2%}")
            return True
        
        return False

class Trader:
    """Main trader class for executing trades"""
    
    def __init__(self, exchange_config: Dict[str, Any], risk_manager: RiskManager = None):
        self.exchange = self._initialize_exchange(exchange_config)
        self.risk_manager = risk_manager or RiskManager()
        self.positions: Dict[str, Position] = {}
        self.order_history: List[Dict] = []
        self.balance = 0.0
        
    def _initialize_exchange(self, config: Dict[str, Any]) -> ccxt_async.Exchange:
        """Initialize exchange connection"""
        exchange_class = getattr(ccxt_async, config['exchange_name'])
        
        exchange = exchange_class({
            'apiKey': config['api_key'],
            'secret': config['secret_key'],
            'sandbox': config.get('sandbox', True),
            'enableRateLimit': True,
        })
        
        logger.info(f"Initialized {config['exchange_name']} exchange (sandbox: {config.get('sandbox', True)})")
        return exchange
    
    async def close(self):
        """Close the exchange connection"""
        if hasattr(self.exchange, 'close'):
            await self.exchange.close()
    
    async def get_balance(self) -> float:
        """Get account balance"""
        try:
            balance = await self.exchange.fetch_balance()
            self.balance = balance['total'].get('USDT', 0.0)  # Assuming USDT balance
            return self.balance
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0.0
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current market price"""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return 0.0
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol"""
        try:
            positions = await self.exchange.fetch_positions([symbol])
            
            for pos in positions:
                if pos['symbol'] == symbol and pos['size'] > 0:
                    side = 'long' if pos['side'] == 'long' else 'short'
                    return Position(
                        symbol=symbol,
                        side=side,
                        size=pos['size'],
                        entry_price=pos['entryPrice'],
                        timestamp=datetime.now()
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching position for {symbol}: {e}")
            return None
    
    async def place_market_order(self, symbol: str, side: str, amount: float) -> Dict[str, Any]:
        """Place a market order"""
        try:
            # Adjust amount based on risk management
            adjusted_amount = self.risk_manager.check_position_size(amount)
            
            if adjusted_amount != amount:
                logger.info(f"Position size adjusted from {amount} to {adjusted_amount}")
            
            # Place order
            order = await self.exchange.create_market_order(symbol, side, adjusted_amount)
            
            # Log order
            order_info = {
                'id': order['id'],
                'symbol': symbol,
                'side': side,
                'amount': adjusted_amount,
                'price': order.get('price', 0),
                'timestamp': datetime.now().isoformat(),
                'status': order['status']
            }
            
            self.order_history.append(order_info)
            logger.info(f"Placed {side} order: {order_info}")
            
            return order
            
        except Exception as e:
            logger.error(f"Error placing {side} order for {symbol}: {e}")
            return None
    
    async def close_position(self, symbol: str) -> bool:
        """Close current position for symbol"""
        try:
            position = await self.get_position(symbol)
            if not position:
                logger.info(f"No position to close for {symbol}")
                return True
            
            # Determine order side to close position
            close_side = 'sell' if position.side == 'long' else 'buy'
            
            # Place closing order
            order = await self.place_market_order(symbol, close_side, position.size)
            
            if order:
                # Calculate realized PnL
                current_price = await self.get_current_price(symbol)
                position.update_pnl(current_price)
                realized_pnl = position.unrealized_pnl
                
                # Update risk manager
                self.risk_manager.update_daily_pnl(realized_pnl)
                
                # Remove from positions
                if symbol in self.positions:
                    del self.positions[symbol]
                
                logger.info(f"Closed position: {position} with PnL: {realized_pnl:.2f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return False
    
    async def execute_signal(self, signal: TradingSignal, symbol: str) -> bool:
        """Execute a trading signal"""
        try:
            # Check if trading should be stopped
            if self.risk_manager.should_stop_trading():
                logger.warning("Trading stopped due to daily loss limit")
                return False
            
            current_position = await self.get_position(symbol)
            current_price = await self.get_current_price(symbol)
            
            # Update position PnL if exists
            if current_position:
                current_position.update_pnl(current_price)
                
                # Check for risk management exit
                if self.risk_manager.should_close_position(current_position, current_price):
                    logger.info("Risk management triggered position close")
                    return await self.close_position(symbol)
            
            # Execute signal based on type
            if signal.signal_type == SignalType.LONG:
                return await self._execute_long_signal(signal, symbol, current_position)
                
            elif signal.signal_type == SignalType.SHORT:
                return await self._execute_short_signal(signal, symbol, current_position)
                
            elif signal.signal_type == SignalType.CLOSE_LONG:
                if current_position and current_position.side == 'long':
                    return await self.close_position(symbol)
                    
            elif signal.signal_type == SignalType.CLOSE_SHORT:
                if current_position and current_position.side == 'short':
                    return await self.close_position(symbol)
                    
            elif signal.signal_type == SignalType.HOLD:
                logger.info(f"Holding position for {symbol}")
                return True
                
            else:  # NO_SIGNAL
                logger.debug(f"No signal for {symbol}")
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing signal for {symbol}: {e}")
            return False
    
    async def _execute_long_signal(self, signal: TradingSignal, symbol: str, 
                                  current_position: Optional[Position]) -> bool:
        """Execute a long signal"""
        try:
            # Close short position if exists
            if current_position and current_position.side == 'short':
                await self.close_position(symbol)
            
            # Calculate position size based on signal strength and balance
            balance = await self.get_balance()
            base_size = balance * 0.1  # Use 10% of balance as base
            position_size = base_size * signal.strength
            
            # Place long order
            order = await self.place_market_order(symbol, 'buy', position_size)
            
            if order:
                # Update positions tracking
                current_price = await self.get_current_price(symbol)
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side='long',
                    size=position_size,
                    entry_price=current_price
                )
                
                logger.info(f"Opened long position: {self.positions[symbol]}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing long signal: {e}")
            return False
    
    async def _execute_short_signal(self, signal: TradingSignal, symbol: str, 
                                   current_position: Optional[Position]) -> bool:
        """Execute a short signal"""
        try:
            # Close long position if exists
            if current_position and current_position.side == 'long':
                await self.close_position(symbol)
            
            # Calculate position size based on signal strength and balance
            balance = await self.get_balance()
            base_size = balance * 0.1  # Use 10% of balance as base
            position_size = base_size * signal.strength
            
            # Place short order
            order = await self.place_market_order(symbol, 'sell', position_size)
            
            if order:
                # Update positions tracking
                current_price = await self.get_current_price(symbol)
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side='short',
                    size=position_size,
                    entry_price=current_price
                )
                
                logger.info(f"Opened short position: {self.positions[symbol]}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing short signal: {e}")
            return False
    
    async def update_positions(self):
        """Update all position PnL and check risk management"""
        for symbol, position in self.positions.items():
            try:
                current_price = await self.get_current_price(symbol)
                position.update_pnl(current_price)
                
                # Check if position should be closed due to risk management
                if self.risk_manager.should_close_position(position, current_price):
                    await self.close_position(symbol)
                    
            except Exception as e:
                logger.error(f"Error updating position for {symbol}: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        return {
            'balance': self.balance,
            'positions_count': len(self.positions),
            'total_unrealized_pnl': total_unrealized_pnl,
            'daily_pnl': self.risk_manager.daily_pnl,
            'positions': [pos.to_dict() for pos in self.positions.values()],
            'recent_orders': self.order_history[-10:] if self.order_history else []
        }