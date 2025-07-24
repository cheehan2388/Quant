"""
Enhanced Trading System with Feature Engineering and Strategy Selection
Integrates all components for a complete, flexible trading solution.
"""

import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import signal
import json

from config import Config, validate_config
from data_provider import DataManager, BinanceDataProvider, BybitDataProvider
from feature_engineering import create_default_feature_engineering_manager
from strategy_v2 import create_strategy_selector, SignalType
from trader import Trader, RiskManager

# Set up logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class EnhancedTradingSystem:
    """Enhanced trading system with feature engineering and strategy selection"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.feature_manager = create_default_feature_engineering_manager()
        self.strategy_selector = create_strategy_selector(threshold=Config.ZSCORE_LONG_THRESHOLD)
        self.trader = None
        self.running = False
        self.last_data_fetch = datetime.min
        self.performance_metrics = {
            'signals_generated': 0,
            'trades_executed': 0,
            'successful_trades': 0,
            'total_pnl': 0.0,
            'start_time': None
        }
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components"""
        logger.info("Initializing enhanced trading system...")
        self._initialize_data_providers()
        self._initialize_trader()
        logger.info("Enhanced trading system initialized")
    
    def _initialize_data_providers(self):
        """Initialize data providers"""
        logger.info("Initializing data providers...")
        binance_provider = BinanceDataProvider(
            api_key=Config.BINANCE_API_KEY,
            secret_key=Config.BINANCE_SECRET_KEY,
            sandbox=Config.BINANCE_SANDBOX
        )
        self.data_manager.add_provider("binance", binance_provider)
        
        bybit_provider = BybitDataProvider(
            api_key=Config.BYBIT_API_KEY,
            secret_key=Config.BYBIT_SECRET_KEY,
            sandbox=Config.BYBIT_SANDBOX
        )
        self.data_manager.add_provider("bybit", bybit_provider)
        
        logger.info("Data providers initialized")
    
    def _initialize_trader(self):
        """Initialize trader with risk management"""
        logger.info("Initializing trader...")
        risk_manager = RiskManager(
            max_position_size=Config.MAX_POSITION_SIZE,
            stop_loss_pct=Config.STOP_LOSS_PCT,
            take_profit_pct=Config.TAKE_PROFIT_PCT
        )
        trader_config = {
            'exchange_name': 'bybit',
            'api_key': Config.BYBIT_API_KEY,
            'secret_key': Config.BYBIT_SECRET_KEY,
            'sandbox': Config.BYBIT_SANDBOX
        }
        self.trader = Trader(trader_config, risk_manager)
        logger.info("Trader initialized")
    
    def set_strategy(self, strategy_name: str):
        try:
            self.strategy_selector.set_active_strategy(strategy_name)
            logger.info(f"Strategy changed to: {strategy_name}")
        except ValueError as e:
            logger.error(f"Failed to set strategy: {e}")
            raise
    
    def get_available_strategies(self) -> Dict[str, Dict[str, Any]]:
        return self.strategy_selector.get_strategy_info()
    
    def get_available_features(self) -> Dict[str, List[str]]:
        return self.feature_manager.get_available_features()
    
    async def run_single_cycle(self, symbol: str = None) -> Dict[str, Any]:
        symbol = symbol or Config.TRADING_SYMBOL
        cycle_results = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'success': False,
            'signal': None,
            'features': None,
            'error': None
        }
        
        try:
            logger.info(f"Fetching data for {symbol}...")
            data_result = await self.data_manager.fetch_primary_open_interest(symbol, "binance")
            
            if not data_result:
                logger.warning("No data received")
                cycle_results['error'] = "No data received"
                return cycle_results
            
            self.data_manager.store_data(symbol, data_result)
            historical_data = self.data_manager.get_data_as_dataframe(symbol, limit=Config.LOOKBACK_PERIOD + 10)
            
            if historical_data.empty or len(historical_data) < 5:
                logger.warning("Insufficient historical data")
                cycle_results['error'] = "Insufficient historical data"
                return cycle_results
            
            logger.debug("Calculating features...")
            features = self.feature_manager.calculate_all_features(historical_data)
            cycle_results['features'] = features
            
            current_position = await self.trader.get_position(symbol)
            position_dict = current_position.to_dict() if current_position else None
            
            logger.debug("Generating trading signal...")
            signal = self.strategy_selector.generate_signal(features, position_dict)
            cycle_results['signal'] = {
                'type': signal.signal_type.value,
                'strength': signal.strength,
                'reason': signal.reason,
                'metadata': signal.metadata
            }
            
            self.performance_metrics['signals_generated'] += 1
            
            if signal.signal_type != SignalType.NO_SIGNAL:
                logger.info(f"Executing signal: {signal}")
                execution_success = await self.trader.execute_signal(signal, symbol)
                
                if execution_success:
                    self.performance_metrics['trades_executed'] += 1
                    logger.info("Signal executed successfully")
                else:
                    logger.warning("Signal execution failed")
                
                cycle_results['execution_success'] = execution_success
            
            await self.trader.update_positions()
            cycle_results['success'] = True
            cycle_results['portfolio'] = self.trader.get_portfolio_summary()
            logger.info(f"Cycle completed: {signal.signal_type.value} signal with strength {signal.strength:.2f}")
        
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            cycle_results['error'] = str(e)
        
        return cycle_results
    
    async def run_continuous(self, symbol: str = None):
        symbol = symbol or Config.TRADING_SYMBOL
        self.running = True
        self.performance_metrics['start_time'] = datetime.now()
        
        logger.info(f"Starting continuous trading for {symbol}")
        logger.info(f"Active strategy: {self.strategy_selector.active_strategy}")
        logger.info(f"Data fetch interval: {Config.DATA_FETCH_INTERVAL} seconds")
        
        await self._test_connections()
        cycle_count = 0
        
        try:
            while self.running:
                cycle_count += 1
                logger.info(f"--- Trading Cycle #{cycle_count} ---")
                cycle_results = await self.run_single_cycle(symbol)
                
                if cycle_results['success']:
                    signal_info = cycle_results.get('signal', {})
                    logger.info(f"Cycle {cycle_count}: {signal_info.get('type', 'NO_SIGNAL')} "
                                f"(strength: {signal_info.get('strength', 0):.2f})")
                else:
                    logger.warning(f"Cycle {cycle_count} failed: {cycle_results.get('error', 'Unknown error')}")
                
                logger.debug(f"Waiting {Config.DATA_FETCH_INTERVAL} seconds until next cycle...")
                await asyncio.sleep(Config.DATA_FETCH_INTERVAL)
        
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, stopping...")
        except Exception as e:
            logger.error(f"Unexpected error in continuous trading: {e}")
        finally:
            await self.stop()
    
    async def _test_connections(self):
        logger.info("Testing connections...")
        
        try:
            test_symbol = Config.TRADING_SYMBOL
            test_data = await self.data_manager.fetch_primary_open_interest(test_symbol, "binance")
            if test_data:
                logger.info("✓ Data connection successful")
            else:
                logger.warning("⚠ Data connection test returned no data")
            
            balance = await self.trader.get_balance()
            logger.info(f"✓ Trading connection successful (Balance: {balance:.2f} USDT)")
        
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            raise
    
    async def stop(self):
        logger.info("Stopping enhanced trading system...")
        self.running = False
        self._log_performance_summary()
        
        try:
            if self.trader:
                await self.trader.close()
            
            for provider in self.data_manager.providers.values():
                if hasattr(provider, 'close'):
                    await provider.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        logger.info("Enhanced trading system stopped")
    
    def _log_performance_summary(self):
        metrics = self.performance_metrics
        
        if metrics['start_time']:
            runtime = datetime.now() - metrics['start_time']
            logger.info("=" * 50)
            logger.info("PERFORMANCE SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Runtime: {runtime}")
            logger.info(f"Signals Generated: {metrics['signals_generated']}")
            logger.info(f"Trades Executed: {metrics['trades_executed']}")
            logger.info(f"Success Rate: {metrics['successful_trades']}/{metrics['trades_executed']} "
                        f"({metrics['successful_trades']/max(1, metrics['trades_executed'])*100:.1f}%)")
            logger.info(f"Total PnL: {metrics['total_pnl']:.2f} USDT")
            
            if self.trader:
                portfolio = self.trader.get_portfolio_summary()
                logger.info(f"Final Balance: {portfolio['balance']:.2f} USDT")
                logger.info(f"Open Positions: {portfolio['positions_count']}")
                logger.info(f"Unrealized PnL: {portfolio['total_unrealized_pnl']:.2f} USDT")
            logger.info("=" * 50)
    
    def get_system_status(self) -> Dict[str, Any]:
        return {
            'running': self.running,
            'active_strategy': self.strategy_selector.active_strategy,
            'available_strategies': list(self.strategy_selector.get_available_strategies()),
            'performance_metrics': self.performance_metrics.copy(),
            'feature_engineers': list(self.feature_manager.engineers.keys()),
            'data_providers': list(self.data_manager.providers.keys()),
            'trader_connected': self.trader is not None,
            'last_data_fetch': self.last_data_fetch.isoformat() if self.last_data_fetch != datetime.min else None
        }

async def switch_strategy_demo(system: EnhancedTradingSystem):
    logger.info("🔄 Strategy Switching Demo")
    logger.info("=" * 40)
    
    strategies = system.get_available_strategies()
    
    for strategy_name, strategy_info in strategies.items():
        logger.info(f"\n📋 Testing Strategy: {strategy_name}")
        logger.info(f"   Required Features: {strategy_info['required_features']}")
        
        system.set_strategy(strategy_name)
        result = await system.run_single_cycle()
        
        if result['success']:
            signal = result['signal']
            logger.info(f"   ✅ Signal: {signal['type']} (strength: {signal['strength']:.2f})")
            logger.info(f"   📝 Reason: {signal['reason']}")
        else:
            logger.warning(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
        
        await asyncio.sleep(1)

def setup_signal_handlers(system: EnhancedTradingSystem):
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(system.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main():
    try:
        validate_config()
        system = EnhancedTradingSystem()
        setup_signal_handlers(system)
        
        if len(sys.argv) > 1 and sys.argv[1] == "--demo":
            logger.info("🎯 Running in DEMO mode")
            await switch_strategy_demo(system)
            return
        
        if len(sys.argv) > 1 and sys.argv[1] == "--strategy":
            if len(sys.argv) > 2:
                strategy_name = sys.argv[2]
                logger.info(f"🎯 Setting strategy to: {strategy_name}")
                system.set_strategy(strategy_name)
            else:
                logger.error("Please specify a strategy name")
                logger.info(f"Available strategies: {list(system.get_available_strategies().keys())}")
                return
        
        status = system.get_system_status()
        logger.info("🚀 Enhanced Trading System Status:")
        logger.info(f"   Active Strategy: {status['active_strategy']}")
        logger.info(f"   Available Strategies: {len(status['available_strategies'])}")
        logger.info(f"   Feature Engineers: {len(status['feature_engineers'])}")
        logger.info(f"   Data Providers: {len(status['data_providers'])}")
        
        await system.run_continuous()
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"System error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
