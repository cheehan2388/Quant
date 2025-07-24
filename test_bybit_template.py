#!/usr/bin/env python3
"""
Test script for Bybit CCXT template
This script tests the template functionality without requiring API credentials.
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class TestBybitCCXTTemplate(unittest.TestCase):
    """
    Test cases for the Bybit CCXT template.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'BYBIT_API_KEY': 'test_api_key',
            'BYBIT_SECRET': 'test_secret'
        })
        self.env_patcher.start()
        
        # Mock CCXT exchange
        self.exchange_mock = Mock()
        
    def tearDown(self):
        """Clean up after tests."""
        self.env_patcher.stop()
    
    @patch('ccxt.bybit')
    def test_initialization(self, mock_bybit):
        """Test template initialization."""
        from bybit_ccxt_template import BybitCCXTTemplate
        
        # Mock the exchange
        mock_bybit.return_value = self.exchange_mock
        
        # Test initialization
        template = BybitCCXTTemplate(sandbox=True)
        
        self.assertEqual(template.api_key, 'test_api_key')
        self.assertEqual(template.secret, 'test_secret')
        self.assertTrue(template.sandbox)
        
        # Verify CCXT was called correctly
        mock_bybit.assert_called_once()
        call_args = mock_bybit.call_args[0][0]  # First positional argument is the config dict
        self.assertEqual(call_args['apiKey'], 'test_api_key')
        self.assertEqual(call_args['secret'], 'test_secret')
        self.assertTrue(call_args['sandbox'])
        self.assertTrue(call_args['enableRateLimit'])
    
    @patch('ccxt.bybit')
    def test_connection_test_success(self, mock_bybit):
        """Test successful connection test."""
        from bybit_ccxt_template import BybitCCXTTemplate
        
        # Mock successful balance fetch
        self.exchange_mock.fetch_balance.return_value = {
            'total': {'USDT': 1000},
            'free': {'USDT': 1000},
            'used': {'USDT': 0},
            'timestamp': 1234567890,
            'datetime': '2023-01-01T00:00:00.000Z'
        }
        mock_bybit.return_value = self.exchange_mock
        
        template = BybitCCXTTemplate()
        result = template.test_connection()
        
        self.assertTrue(result)
        self.exchange_mock.fetch_balance.assert_called_once()
    
    @patch('ccxt.bybit')
    def test_connection_test_failure(self, mock_bybit):
        """Test failed connection test."""
        from bybit_ccxt_template import BybitCCXTTemplate
        
        # Mock failed balance fetch
        self.exchange_mock.fetch_balance.side_effect = Exception("Connection failed")
        mock_bybit.return_value = self.exchange_mock
        
        template = BybitCCXTTemplate()
        result = template.test_connection()
        
        self.assertFalse(result)
    
    @patch('ccxt.bybit')
    def test_get_account_info(self, mock_bybit):
        """Test getting account information."""
        from bybit_ccxt_template import BybitCCXTTemplate
        
        # Mock balance data
        balance_data = {
            'total': {'USDT': 1000, 'BTC': 0.1},
            'free': {'USDT': 800, 'BTC': 0.1},
            'used': {'USDT': 200, 'BTC': 0},
            'timestamp': 1234567890,
            'datetime': '2023-01-01T00:00:00.000Z'
        }
        self.exchange_mock.fetch_balance.return_value = balance_data
        mock_bybit.return_value = self.exchange_mock
        
        template = BybitCCXTTemplate()
        account_info = template.get_account_info()
        
        self.assertEqual(account_info['total_balance'], balance_data['total'])
        self.assertEqual(account_info['free_balance'], balance_data['free'])
        self.assertEqual(account_info['used_balance'], balance_data['used'])
    
    @patch('ccxt.bybit')
    def test_get_ticker(self, mock_bybit):
        """Test getting ticker information."""
        from bybit_ccxt_template import BybitCCXTTemplate
        
        # Mock ticker data
        ticker_data = {
            'symbol': 'BTC/USDT',
            'last': 50000.0,
            'bid': 49999.0,
            'ask': 50001.0,
            'baseVolume': 1000.0,
            'percentage': 2.5
        }
        self.exchange_mock.fetch_ticker.return_value = ticker_data
        mock_bybit.return_value = self.exchange_mock
        
        template = BybitCCXTTemplate()
        ticker = template.get_ticker('BTC/USDT')
        
        self.assertEqual(ticker['last'], 50000.0)
        self.assertEqual(ticker['bid'], 49999.0)
        self.assertEqual(ticker['ask'], 50001.0)
    
    @patch('ccxt.bybit')
    def test_get_ohlcv(self, mock_bybit):
        """Test getting OHLCV data."""
        from bybit_ccxt_template import BybitCCXTTemplate
        
        # Mock OHLCV data
        ohlcv_data = [
            [1234567890000, 50000.0, 51000.0, 49000.0, 50500.0, 100.0],
            [1234567920000, 50500.0, 52000.0, 50000.0, 51500.0, 150.0],
        ]
        self.exchange_mock.fetch_ohlcv.return_value = ohlcv_data
        mock_bybit.return_value = self.exchange_mock
        
        template = BybitCCXTTemplate()
        ohlcv = template.get_ohlcv('BTC/USDT', '1h', 2)
        
        self.assertIsInstance(ohlcv, pd.DataFrame)
        self.assertEqual(len(ohlcv), 2)
        self.assertIn('open', ohlcv.columns)
        self.assertIn('high', ohlcv.columns)
        self.assertIn('low', ohlcv.columns)
        self.assertIn('close', ohlcv.columns)
        self.assertIn('volume', ohlcv.columns)
    
    @patch('ccxt.bybit')
    def test_place_limit_order(self, mock_bybit):
        """Test placing a limit order."""
        from bybit_ccxt_template import BybitCCXTTemplate
        
        # Mock order data
        order_data = {
            'id': 'test_order_id',
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'amount': 0.001,
            'price': 50000.0,
            'status': 'open'
        }
        self.exchange_mock.create_limit_order.return_value = order_data
        mock_bybit.return_value = self.exchange_mock
        
        template = BybitCCXTTemplate()
        order = template.place_limit_order('BTC/USDT', 'buy', 0.001, 50000.0)
        
        self.assertEqual(order['id'], 'test_order_id')
        self.assertEqual(order['symbol'], 'BTC/USDT')
        self.assertEqual(order['side'], 'buy')
    
    @patch('ccxt.bybit')
    def test_calculate_position_size(self, mock_bybit):
        """Test position size calculation."""
        from bybit_ccxt_template import BybitCCXTTemplate
        
        # Mock balance and ticker data
        balance_data = {
            'total': {'USDT': 10000},
            'free': {'USDT': 10000},
            'used': {'USDT': 0},
            'timestamp': 1234567890,
            'datetime': '2023-01-01T00:00:00.000Z'
        }
        ticker_data = {
            'last': 50000.0,
            'bid': 49999.0,
            'ask': 50001.0
        }
        
        self.exchange_mock.fetch_balance.return_value = balance_data
        self.exchange_mock.fetch_ticker.return_value = ticker_data
        mock_bybit.return_value = self.exchange_mock
        
        template = BybitCCXTTemplate()
        position_size = template.calculate_position_size('BTC/USDT', 2.0, 5.0)
        
        # Expected calculation: (10000 * 0.02) / 0.05 = 4000 USDT worth
        expected_size = (10000 * 0.02) / 0.05
        self.assertAlmostEqual(position_size, expected_size, places=6)


class TestBybitConfig(unittest.TestCase):
    """
    Test cases for the Bybit configuration.
    """
    
    def test_validate_config_with_valid_data(self):
        """Test configuration validation with valid data."""
        with patch.dict(os.environ, {
            'BYBIT_API_KEY': 'test_key',
            'BYBIT_SECRET': 'test_secret'
        }):
            # Reload the module to get fresh environment variables
            import importlib
            import bybit_config
            importlib.reload(bybit_config)
            result = bybit_config.validate_config()
            self.assertTrue(result)
    
    def test_validate_config_with_missing_api_key(self):
        """Test configuration validation with missing API key."""
        with patch.dict(os.environ, {}, clear=True):
            # Reload the module to get fresh environment variables
            import importlib
            import bybit_config
            importlib.reload(bybit_config)
            result = bybit_config.validate_config()
            self.assertFalse(result)
    
    def test_get_config_summary(self):
        """Test getting configuration summary."""
        with patch.dict(os.environ, {
            'BYBIT_API_KEY': 'test_key',
            'BYBIT_SECRET': 'test_secret',
            'BYBIT_SANDBOX': 'true'
        }):
            from bybit_config import get_config_summary
            summary = get_config_summary()
            
            self.assertIn('api_configured', summary)
            self.assertIn('sandbox_mode', summary)
            self.assertIn('default_symbol', summary)
            self.assertTrue(summary['api_configured'])


def run_tests():
    """Run all tests."""
    print("Running Bybit CCXT Template Tests...")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestBybitCCXTTemplate))
    suite.addTests(loader.loadTestsFromTestCase(TestBybitConfig))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)