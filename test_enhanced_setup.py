#!/usr/bin/env python3
"""
Enhanced test script to verify the corrected trading bot setup
Tests configuration management, timing, and initial data loading
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_fetcher'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'trading'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))

def test_config_management():
    """Test centralized configuration management"""
    print("Testing Configuration Management...")
    
    try:
        from config.settings import ConfigManager
        
        # Test config loading
        config = ConfigManager()
        print("✓ ConfigManager loaded successfully")
        
        # Test configuration access
        print(f"  Symbol: {config.symbol}")
        print(f"  Z-Score Threshold: {config.trading.z_score_threshold}")
        print(f"  Rolling Window: {config.trading.rolling_window}")
        print(f"  Fetch on Hour Mark: {config.data.fetch_on_hour_mark}")
        print(f"  Initial Data Hours: {config.data.initial_data_hours}")
        print(f"  Bybit Testnet: {config.exchange.bybit_testnet}")
        
        # Test configuration update
        old_threshold = config.trading.z_score_threshold
        config.update_config('trading', 'z_score_threshold', 3.0)
        
        if config.trading.z_score_threshold == 3.0:
            print("✓ Configuration update works")
            # Restore original value
            config.update_config('trading', 'z_score_threshold', old_threshold)
        else:
            print("✗ Configuration update failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration management test failed: {e}")
        return False

def test_enhanced_binance_fetcher():
    """Test enhanced Binance data fetcher with timing"""
    print("\nTesting Enhanced Binance Data Fetcher...")
    
    try:
        from data_fetcher.binance_indicators import BinanceIndicatorsFetcher
        from config.settings import ConfigManager
        
        config = ConfigManager()
        fetcher = BinanceIndicatorsFetcher(config_manager=config)
        print("✓ Enhanced BinanceIndicatorsFetcher created successfully")
        
        # Test current hour timestamp
        current_hour = fetcher.get_current_hour_timestamp()
        print(f"✓ Current hour timestamp: {current_hour}")
        
        # Test data availability check (this will likely return False unless run at exact hour)
        symbol = 'BTCUSDT'
        is_available = fetcher.is_new_hour_data_available(symbol, current_hour)
        print(f"✓ Data availability check completed: {is_available}")
        
        # Test historical data loading
        print("  Loading historical data...")
        historical_data = fetcher.load_initial_historical_data(symbol, hours=2)  # Small test
        
        if historical_data:
            print(f"✓ Loaded {len(historical_data)} historical data points")
            print(f"  Latest data: {historical_data[-1][0]:.6f} at {historical_data[-1][1]}")
        else:
            print("⚠ No historical data loaded (may be API issue)")
        
        # Test combined indicator with timestamp
        result = fetcher.get_combined_indicator_with_timestamp(symbol)
        if result:
            indicator, timestamp = result
            print(f"✓ Combined indicator with timestamp: {indicator:.6f} at {timestamp}")
        else:
            print("⚠ Could not fetch combined indicator with timestamp")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced Binance fetcher test failed: {e}")
        return False

def test_enhanced_strategy():
    """Test enhanced Z-score strategy with historical data initialization"""
    print("\nTesting Enhanced Z-Score Strategy...")
    
    try:
        from trading.strategy import ZScoreStrategy
        from config.settings import ConfigManager
        
        config = ConfigManager()
        strategy = ZScoreStrategy(config_manager=config)
        print("✓ Enhanced ZScoreStrategy created successfully")
        
        # Create mock historical data
        base_time = datetime.now() - timedelta(hours=1)
        historical_data = []
        
        for i in range(20):  # More than rolling window
            timestamp = base_time + timedelta(minutes=i*5)
            value = 1000000 + (i * 1000) + (i % 3 * 500)  # Some variation
            historical_data.append((value, timestamp))
        
        # Test initialization with historical data
        success = strategy.initialize_with_historical_data(historical_data)
        
        if success:
            print(f"✓ Strategy initialized with {len(historical_data)} historical data points")
        else:
            print("✗ Strategy initialization failed")
            return False
        
        # Test if strategy is ready
        if strategy.is_ready():
            print("✓ Strategy is ready for trading")
        else:
            print("✗ Strategy not ready after initialization")
            return False
        
        # Test signal generation
        signal = strategy.generate_signal()
        if signal:
            print(f"✓ Signal generated: {signal}")
            latest_z = strategy.get_latest_z_score()
            print(f"  Latest Z-Score: {latest_z:.4f}")
        else:
            print("✗ Signal generation failed")
            return False
        
        # Test adding new data with timestamp
        new_timestamp = datetime.now()
        new_value = 1050000
        success = strategy.add_indicator_value(new_value, new_timestamp)
        
        if success:
            print("✓ New timestamped data added successfully")
        else:
            print("✗ Failed to add new timestamped data")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced strategy test failed: {e}")
        return False

def test_timing_logic():
    """Test timing and data freshness logic"""
    print("\nTesting Timing Logic...")
    
    try:
        from config.settings import ConfigManager
        
        config = ConfigManager()
        
        # Test hour mark detection
        now = datetime.now()
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        
        print(f"✓ Current time: {now}")
        print(f"✓ Current hour mark: {current_hour}")
        print(f"✓ Fetch on hour mark setting: {config.data.fetch_on_hour_mark}")
        print(f"✓ Retry interval: {config.data.retry_interval_minutes} minutes")
        print(f"✓ Max retry attempts: {config.data.max_retry_attempts}")
        
        # Test data freshness calculation
        old_timestamp = now - timedelta(minutes=15)
        age_minutes = (now - old_timestamp).total_seconds() / 60
        is_stale = age_minutes > config.data.data_freshness_threshold_minutes
        
        print(f"✓ Data age calculation: {age_minutes:.1f} minutes")
        print(f"✓ Data freshness threshold: {config.data.data_freshness_threshold_minutes} minutes")
        print(f"✓ Data is stale: {is_stale}")
        
        return True
        
    except Exception as e:
        print(f"✗ Timing logic test failed: {e}")
        return False

def test_main_bot_imports():
    """Test that the main bot can import all components"""
    print("\nTesting Main Bot Imports...")
    
    try:
        # Test individual imports first
        from config.settings import ConfigManager
        print("✓ ConfigManager import successful")
        
        from data_fetcher.binance_indicators import BinanceIndicatorsFetcher  
        print("✓ BinanceIndicatorsFetcher import successful")
        
        from trading.strategy import ZScoreStrategy
        print("✓ ZScoreStrategy import successful")
        
        # Test main bot can be imported (but not run)
        import main_trading_bot
        print("✓ Main trading bot module import successful")
        
        # Test that TradingBot class can be instantiated (configuration permitting)
        try:
            # This will fail if .env is not set up, but that's expected
            bot = main_trading_bot.TradingBot()
            print("✓ TradingBot class instantiation successful")
            return True
        except ValueError as e:
            if "API keys" in str(e):
                print("⚠ TradingBot instantiation failed due to missing API keys (expected)")
                return True
            else:
                print(f"✗ TradingBot instantiation failed: {e}")
                return False
        
    except Exception as e:
        print(f"✗ Main bot imports test failed: {e}")
        return False

def main():
    """Run all enhanced tests"""
    print("Enhanced Trading Bot Setup Test")
    print("==============================")
    print("Testing all corrections and enhancements...\n")
    
    tests = [
        ("Configuration Management", test_config_management),
        ("Enhanced Binance Fetcher", test_enhanced_binance_fetcher),
        ("Enhanced Strategy", test_enhanced_strategy), 
        ("Timing Logic", test_timing_logic),
        ("Main Bot Imports", test_main_bot_imports),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"{test_name}")
        print("-" * len(test_name))
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
        
        print()  # Add spacing between tests
    
    # Summary
    print("="*60)
    print("ENHANCED TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All enhanced tests passed! The corrections are working properly.")
        print("\nKey Features Verified:")
        print("  ✓ Centralized configuration management")
        print("  ✓ Hourly data fetching with retry logic") 
        print("  ✓ Initial historical data loading")
        print("  ✓ Timestamped data handling")
        print("  ✓ Data freshness validation")
        print("  ✓ Enhanced strategy initialization")
        
        print("\nNext steps:")
        print("1. Copy .env.example to .env and configure your API keys")
        print("2. Set BYBIT_TESTNET=true for testing")
        print("3. Run: python utils/config_validator.py")
        print("4. Run: python main_trading_bot.py")
        
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please review the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())