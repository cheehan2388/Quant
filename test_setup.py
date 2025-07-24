#!/usr/bin/env python3
"""
Test script to verify the trading bot setup
"""

import sys
import os
import logging
from datetime import datetime

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_fetcher'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'trading'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from data_fetcher.binance_indicators import BinanceIndicatorsFetcher
        print("✓ BinanceIndicatorsFetcher imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import BinanceIndicatorsFetcher: {e}")
        return False
    
    try:
        from trading.strategy import ZScoreStrategy
        print("✓ ZScoreStrategy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ZScoreStrategy: {e}")
        return False
    
    try:
        from trading.position_manager import PositionManager
        print("✓ PositionManager imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PositionManager: {e}")
        return False
    
    try:
        from trading.order_manager import OrderManager
        print("✓ OrderManager imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import OrderManager: {e}")
        return False
    
    try:
        from utils.config_validator import ConfigValidator
        print("✓ ConfigValidator imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ConfigValidator: {e}")
        return False
    
    return True

def test_dependencies():
    """Test that required dependencies are available"""
    print("\nTesting dependencies...")
    
    required_modules = [
        'ccxt',
        'pandas',
        'numpy',
        'requests',
        'scipy',
        'dotenv'
    ]
    
    all_good = True
    for module in required_modules:
        try:
            if module == 'dotenv':
                import python_dotenv
                print(f"✓ {module} (python-dotenv) available")
            else:
                __import__(module)
                print(f"✓ {module} available")
        except ImportError:
            print(f"✗ {module} not available - install with: pip install {module}")
            all_good = False
    
    return all_good

def test_strategy_basic():
    """Test basic strategy functionality"""
    print("\nTesting strategy functionality...")
    
    try:
        from trading.strategy import ZScoreStrategy
        
        # Create strategy with small window for testing
        strategy = ZScoreStrategy(rolling_window=3, z_threshold=1.0)
        
        # Add some test data
        test_values = [100, 102, 98, 105, 110]
        for i, value in enumerate(test_values):
            success = strategy.add_indicator_value(value)
            if not success:
                print(f"✗ Failed to add indicator value: {value}")
                return False
        
        # Check if strategy is ready
        if not strategy.is_ready():
            print("✗ Strategy not ready after adding sufficient data")
            return False
        
        # Generate a signal
        signal = strategy.generate_signal()
        if signal is None:
            print("✗ Failed to generate signal")
            return False
        
        print(f"✓ Strategy test passed. Generated signal: {signal}")
        return True
        
    except Exception as e:
        print(f"✗ Strategy test failed: {e}")
        return False

def test_binance_public_data():
    """Test Binance public data access"""
    print("\nTesting Binance public data access...")
    
    try:
        from data_fetcher.binance_indicators import BinanceIndicatorsFetcher
        
        # Create fetcher without API keys (public data only)
        fetcher = BinanceIndicatorsFetcher()
        
        # Test fetching open interest
        oi = fetcher.get_open_interest('BTCUSDT')
        if oi is None:
            print("✗ Failed to fetch open interest")
            return False
        
        print(f"✓ Successfully fetched open interest: {oi}")
        
        # Test fetching long/short ratio
        ls_ratio = fetcher.get_long_short_ratio('BTCUSDT')
        if ls_ratio is None:
            print("✗ Failed to fetch long/short ratio")
            return False
        
        print(f"✓ Successfully fetched long/short ratio: {ls_ratio}")
        
        # Test combined indicator
        combined = fetcher.get_combined_indicator('BTCUSDT')
        if combined is None:
            print("✗ Failed to calculate combined indicator")
            return False
        
        print(f"✓ Successfully calculated combined indicator: {combined}")
        return True
        
    except Exception as e:
        print(f"✗ Binance data test failed: {e}")
        return False

def test_environment_file():
    """Test if .env file exists and has basic structure"""
    print("\nTesting environment configuration...")
    
    if not os.path.exists('.env'):
        print("⚠ .env file not found. You'll need to create it from .env.example")
        if os.path.exists('.env.example'):
            print("✓ .env.example found - copy it to .env and fill in your API keys")
        else:
            print("✗ .env.example not found")
            return False
        return True
    
    print("✓ .env file found")
    
    # Try to load environment
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✓ .env file loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error loading .env file: {e}")
        return False

def main():
    """Run all tests"""
    print("Trading Bot Setup Test")
    print("=====================")
    
    tests = [
        ("Import Test", test_imports),
        ("Dependencies Test", test_dependencies),
        ("Environment Test", test_environment_file),
        ("Strategy Test", test_strategy_basic),
        ("Binance Data Test", test_binance_public_data),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
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
        print("\n✓ All tests passed! Your setup looks good.")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and fill in your API keys")
        print("2. Run: python utils/config_validator.py")
        print("3. Run: python main_trading_bot.py (with sandbox=True first)")
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())