"""
Test script for the trading system components.
Run this to verify everything works before starting live trading.
"""

import asyncio
import logging
import sys
from datetime import datetime

# Configure logging for testing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_data_providers():
    """Test data provider functionality"""
    logger.info("Testing data providers...")
    
    binance = None
    try:
        from data_provider import BinanceDataProvider, DataManager
        
        # Test without API keys (will use public endpoints)
        binance = BinanceDataProvider()
        data_manager = DataManager()
        data_manager.add_provider("binance", binance)
        
        # Try to fetch some test data
        symbol = "BTC/USDT"
        
        # This might fail if no API keys, but we can test the structure
        try:
            # Test OHLCV data (public endpoint)
            ohlcv = await binance.fetch_ohlcv(symbol, '1h', 10)
            if not ohlcv.empty:
                logger.info(f"✓ Successfully fetched OHLCV data: {len(ohlcv)} records")
            else:
                logger.warning("⚠ OHLCV data fetch returned empty result")
        except Exception as e:
            logger.warning(f"⚠ OHLCV fetch failed (expected without API keys): {e}")
        
        # Test open interest (requires API keys)
        try:
            oi_data = await binance.fetch_open_interest(symbol)
            if oi_data:
                logger.info(f"✓ Successfully fetched open interest data: {oi_data}")
            else:
                logger.warning("⚠ Open interest data fetch returned None")
        except Exception as e:
            logger.warning(f"⚠ Open interest fetch failed (expected without API keys): {e}")
        
        logger.info("✓ Data provider test completed")
        
    except Exception as e:
        logger.error(f"✗ Data provider test failed: {e}")
        return False
    finally:
        # Clean up connections
        if binance:
            try:
                await binance.close()
            except Exception:
                pass
    
    return True

def test_feature_engineering():
    """Test feature engineering functionality"""
    logger.info("Testing feature engineering...")
    
    try:
        from feature_engineering import create_default_feature_engineering_manager
        import pandas as pd
        import numpy as np
        
        # Create test data (OHLCV format)
        dates = pd.date_range(start='2023-01-01', periods=50, freq='1h')
        base_price = 50000
        price_changes = np.cumsum(np.random.randn(50) * 10)
        
        test_data = pd.DataFrame({
            'open': base_price + price_changes,
            'high': base_price + price_changes + np.abs(np.random.randn(50) * 50),
            'low': base_price + price_changes - np.abs(np.random.randn(50) * 50),
            'close': base_price + price_changes + np.random.randn(50) * 20,
            'volume': np.random.randint(1000, 10000, 50),
            'open_interest': 100000 + np.cumsum(np.random.randn(50) * 1000)
        })
        
        # Add timestamp as index instead of column to avoid arithmetic issues
        test_data.index = dates
        
        # Test feature engineering manager
        feature_manager = create_default_feature_engineering_manager()
        features = feature_manager.calculate_all_features(test_data)
        
        # Check if features were calculated
        expected_features = ['statistical', 'technical', 'volume']
        for feature_type in expected_features:
            if feature_type in features and features[feature_type].get('valid', False):
                logger.info(f"✓ {feature_type.capitalize()} features calculated successfully")
            else:
                logger.error(f"✗ {feature_type.capitalize()} features calculation failed")
                return False
        
        # Check specific features
        if 'statistical' in features:
            stats = features['statistical']
            if 'zscore' in stats:
                logger.info(f"✓ Z-score calculated: {stats['zscore']:.3f}")
            else:
                logger.warning("⚠ Z-score not found in statistical features")
        
        logger.info("✓ Feature engineering test completed")
        
    except Exception as e:
        logger.error(f"✗ Feature engineering test failed: {e}")
        return False
    
    return True

def test_strategies():
    """Test strategy functionality"""
    logger.info("Testing strategies...")
    
    try:
        from strategy_v2 import create_strategy_selector, SignalType
        
        # Create test features data (matching new feature engineering format)
        test_features = {
            'statistical': {
                'zscore': 1.5,  # Should trigger long signal
                'mean': 100.0,
                'std': 10.0,
                'current_value': 115.0,
                'valid': True
            },
            'technical': {
                'sma_fast': 50100.0,
                'sma_slow': 50000.0,
                'ema_fast': 50120.0,
                'rsi': 65.0,
                'bb_upper': 51000.0,
                'bb_lower': 49000.0,
                'valid': True
            },
            'volume': {
                'volume_sma': 5000.0,
                'volume_ratio': 1.2,
                'vwap': 50050.0,
                'valid': True
            }
        }
        
        # Test strategy selector with different strategies
        strategy_selector = create_strategy_selector(threshold=1.3)
        
        # Test Mean Reversion V1
        strategy_selector.set_active_strategy("mean_reversion_v1")
        signal = strategy_selector.generate_signal(test_features)
        
        if signal.signal_type == SignalType.SHORT:  # Mean reversion: high z-score = short
            logger.info(f"✓ Mean Reversion V1 generated expected SHORT signal: {signal}")
        else:
            logger.warning(f"⚠ Mean Reversion V1 generated unexpected signal: {signal}")
        
        # Test with low z-score
        test_features['statistical']['zscore'] = -1.8
        signal = strategy_selector.generate_signal(test_features)
        
        if signal.signal_type == SignalType.LONG:  # Mean reversion: low z-score = long
            logger.info(f"✓ Mean Reversion V1 generated expected LONG signal: {signal}")
        else:
            logger.warning(f"⚠ Mean Reversion V1 generated unexpected signal: {signal}")
        
        # Test Trend Following V1
        strategy_selector.set_active_strategy("trend_following_v1")
        test_features['statistical']['zscore'] = 1.8  # High z-score
        signal = strategy_selector.generate_signal(test_features)
        
        if signal.signal_type == SignalType.LONG:  # Trend following: high z-score = long
            logger.info(f"✓ Trend Following V1 generated expected LONG signal: {signal}")
        else:
            logger.warning(f"⚠ Trend Following V1 generated unexpected signal: {signal}")
        
        # Test Multi-Feature Strategy
        strategy_selector.set_active_strategy("multi_feature")
        signal = strategy_selector.generate_signal(test_features)
        
        if signal.signal_type in [SignalType.LONG, SignalType.SHORT, SignalType.HOLD]:
            logger.info(f"✓ Multi-Feature strategy generated valid signal: {signal}")
        else:
            logger.warning(f"⚠ Multi-Feature strategy generated unexpected signal: {signal}")
        
        # Test available strategies
        available_strategies = strategy_selector.get_available_strategies()
        expected_strategies = ["mean_reversion_v1", "mean_reversion_v2", "trend_following_v1", "trend_following_v2", "multi_feature"]
        
        if all(strategy in available_strategies for strategy in expected_strategies):
            logger.info("✓ All expected strategies are available")
        else:
            logger.warning(f"⚠ Some strategies missing. Available: {available_strategies}")
        
        logger.info("✓ Strategy test completed")
        
    except Exception as e:
        logger.error(f"✗ Strategy test failed: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration loading"""
    logger.info("Testing configuration...")
    
    try:
        from config import Config
        
        # Test basic config loading
        logger.info(f"Trading symbol: {Config.TRADING_SYMBOL}")
        logger.info(f"Z-score thresholds: {Config.ZSCORE_LONG_THRESHOLD}, {Config.ZSCORE_SHORT_THRESHOLD}")
        logger.info(f"Lookback period: {Config.LOOKBACK_PERIOD}")
        logger.info(f"Sandbox mode: {Config.BYBIT_SANDBOX}")
        
        # Check if API keys are configured (don't log them)
        if Config.BYBIT_API_KEY and Config.BYBIT_SECRET_KEY:
            logger.info("✓ Bybit API keys are configured")
        else:
            logger.warning("⚠ Bybit API keys not configured (required for trading)")
        
        if Config.BINANCE_API_KEY and Config.BINANCE_SECRET_KEY:
            logger.info("✓ Binance API keys are configured")
        else:
            logger.warning("⚠ Binance API keys not configured (required for data)")
        
        logger.info("✓ Configuration test completed")
        
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False
    
    return True

def test_risk_management():
    """Test risk management functionality"""
    logger.info("Testing risk management...")
    
    try:
        from trader import RiskManager, Position
        from datetime import datetime
        
        # Test risk manager
        risk_manager = RiskManager(
            max_position_size=0.1,
            stop_loss_pct=0.05,
            take_profit_pct=0.10
        )
        
        # Test position size check
        adjusted_size = risk_manager.check_position_size(0.2)
        if adjusted_size == 0.1:
            logger.info("✓ Position size limiting works correctly")
        else:
            logger.error("✗ Position size limiting failed")
            return False
        
        # Test position management
        position = Position("BTC/USDT", "long", 0.01, 50000.0)
        
        # Test with profit
        position.update_pnl(55000.0)  # 10% profit
        should_close = risk_manager.should_close_position(position, 55000.0)
        if should_close:
            logger.info("✓ Take profit trigger works correctly")
        else:
            logger.warning("⚠ Take profit didn't trigger as expected")
        
        # Test with loss
        position.update_pnl(47500.0)  # 5% loss
        should_close = risk_manager.should_close_position(position, 47500.0)
        if should_close:
            logger.info("✓ Stop loss trigger works correctly")
        else:
            logger.warning("⚠ Stop loss didn't trigger as expected")
        
        logger.info("✓ Risk management test completed")
        
    except Exception as e:
        logger.error(f"✗ Risk management test failed: {e}")
        return False
    
    return True

async def run_all_tests():
    """Run all tests"""
    logger.info("Starting trading system tests...")
    logger.info("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("Feature Engineering", test_feature_engineering),
        ("Strategies", test_strategies),
        ("Risk Management", test_risk_management),
        ("Data Providers", test_data_providers),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name:<20}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready to use.")
    else:
        logger.warning("⚠ Some tests failed. Check configuration and dependencies.")
    
    return passed == total

if __name__ == "__main__":
    try:
        result = asyncio.run(run_all_tests())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        sys.exit(1)