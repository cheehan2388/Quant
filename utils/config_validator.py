#!/usr/bin/env python3
"""
Configuration validator for the trading bot
"""

import os
import logging
from typing import Dict, Any, List, Tuple
import ccxt
from dotenv import load_dotenv


class ConfigValidator:
    """
    Validates trading bot configuration and API connections
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        load_dotenv()
    
    def validate_environment_variables(self) -> Tuple[bool, List[str]]:
        """
        Validate required environment variables
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Required variables
        required_vars = {
            'BYBIT_API_KEY': 'Bybit API Key',
            'BYBIT_SECRET': 'Bybit Secret Key'
        }
        
        # Optional but recommended
        optional_vars = {
            'BINANCE_API_KEY': 'Binance API Key (for data fetching)',
            'BINANCE_SECRET': 'Binance Secret Key (for data fetching)'
        }
        
        # Check required variables
        for var, description in required_vars.items():
            value = os.getenv(var)
            if not value:
                errors.append(f"Missing required environment variable: {var} ({description})")
            elif len(value.strip()) == 0:
                errors.append(f"Empty environment variable: {var} ({description})")
        
        # Check optional variables
        for var, description in optional_vars.items():
            value = os.getenv(var)
            if not value:
                self.logger.warning(f"Missing optional environment variable: {var} ({description})")
        
        # Validate trading parameters
        try:
            position_size = float(os.getenv('POSITION_SIZE', '0.001'))
            if position_size <= 0:
                errors.append("POSITION_SIZE must be greater than 0")
        except ValueError:
            errors.append("POSITION_SIZE must be a valid number")
        
        try:
            z_threshold = float(os.getenv('Z_SCORE_THRESHOLD', '2.1'))
            if z_threshold <= 0:
                errors.append("Z_SCORE_THRESHOLD must be greater than 0")
        except ValueError:
            errors.append("Z_SCORE_THRESHOLD must be a valid number")
        
        try:
            rolling_window = int(os.getenv('ROLLING_WINDOW', '15'))
            if rolling_window < 2:
                errors.append("ROLLING_WINDOW must be at least 2")
        except ValueError:
            errors.append("ROLLING_WINDOW must be a valid integer")
        
        return len(errors) == 0, errors
    
    def validate_bybit_connection(self) -> Tuple[bool, str]:
        """
        Validate Bybit API connection
        
        Returns:
            Tuple of (is_valid, error_message_or_success)
        """
        try:
            api_key = os.getenv('BYBIT_API_KEY')
            secret = os.getenv('BYBIT_SECRET')
            
            if not api_key or not secret:
                return False, "Bybit API credentials not found"
            
            # Create exchange instance
            exchange = ccxt.bybit({
                'apiKey': api_key,
                'secret': secret,
                'sandbox': False,  # Change to True for testnet
                'enableRateLimit': True,
            })
            
            # Test connection by fetching balance
            balance = exchange.fetch_balance()
            
            # Check if we have USDT balance info
            usdt_balance = balance.get('USDT', {})
            total_balance = usdt_balance.get('total', 0)
            
            return True, f"Bybit connection successful. USDT Balance: {total_balance}"
            
        except ccxt.AuthenticationError:
            return False, "Bybit authentication failed. Check API keys and permissions."
        except ccxt.NetworkError as e:
            return False, f"Bybit network error: {str(e)}"
        except Exception as e:
            return False, f"Bybit connection error: {str(e)}"
    
    def validate_binance_connection(self) -> Tuple[bool, str]:
        """
        Validate Binance API connection (optional)
        
        Returns:
            Tuple of (is_valid, error_message_or_success)
        """
        try:
            api_key = os.getenv('BINANCE_API_KEY', '')
            secret = os.getenv('BINANCE_SECRET', '')
            
            # For public data, API keys are optional
            exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': secret,
                'sandbox': False,
                'enableRateLimit': True,
            })
            
            # Test by fetching public market data
            ticker = exchange.fetch_ticker('BTC/USDT')
            
            if api_key and secret:
                # If we have API keys, test private endpoints
                try:
                    balance = exchange.fetch_balance()
                    return True, "Binance connection successful (with API keys)"
                except:
                    return True, "Binance public data access successful (API keys may be invalid for private data)"
            else:
                return True, "Binance public data access successful (no API keys provided)"
            
        except Exception as e:
            return False, f"Binance connection error: {str(e)}"
    
    def validate_symbol_availability(self, symbol: str = None) -> Tuple[bool, str]:
        """
        Validate that the trading symbol is available on both exchanges
        
        Args:
            symbol: Symbol to validate (defaults to SYMBOL env var)
            
        Returns:
            Tuple of (is_valid, error_message_or_success)
        """
        if symbol is None:
            symbol = os.getenv('SYMBOL', 'BTCUSDT')
        
        try:
            # Check Binance (for data)
            binance = ccxt.binance({'enableRateLimit': True})
            binance_markets = binance.load_markets()
            
            # Check if symbol exists on Binance futures
            binance_symbol_found = False
            for market_symbol in binance_markets:
                if (binance_markets[market_symbol]['base'] + binance_markets[market_symbol]['quote'] == symbol and
                    binance_markets[market_symbol]['type'] == 'future'):
                    binance_symbol_found = True
                    break
            
            if not binance_symbol_found:
                return False, f"Symbol {symbol} not found on Binance Futures"
            
            # Check Bybit (for trading)
            bybit_api_key = os.getenv('BYBIT_API_KEY')
            bybit_secret = os.getenv('BYBIT_SECRET')
            
            if bybit_api_key and bybit_secret:
                bybit = ccxt.bybit({
                    'apiKey': bybit_api_key,
                    'secret': bybit_secret,
                    'enableRateLimit': True,
                })
                bybit_markets = bybit.load_markets()
                
                # Convert symbol format for Bybit
                bybit_symbol = f"{symbol[:-4]}/USDT:USDT" if symbol.endswith('USDT') else symbol
                
                if bybit_symbol not in bybit_markets:
                    return False, f"Symbol {bybit_symbol} not found on Bybit"
                
                return True, f"Symbol {symbol} available on both exchanges"
            else:
                return True, f"Symbol {symbol} found on Binance (Bybit not checked - no API keys)"
            
        except Exception as e:
            return False, f"Error validating symbol availability: {str(e)}"
    
    def run_full_validation(self) -> Dict[str, Any]:
        """
        Run complete configuration validation
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'overall_valid': True,
            'errors': [],
            'warnings': [],
            'checks': {}
        }
        
        # Environment variables check
        env_valid, env_errors = self.validate_environment_variables()
        results['checks']['environment_variables'] = {
            'valid': env_valid,
            'errors': env_errors
        }
        
        if not env_valid:
            results['overall_valid'] = False
            results['errors'].extend(env_errors)
        
        # Bybit connection check
        bybit_valid, bybit_msg = self.validate_bybit_connection()
        results['checks']['bybit_connection'] = {
            'valid': bybit_valid,
            'message': bybit_msg
        }
        
        if not bybit_valid:
            results['overall_valid'] = False
            results['errors'].append(bybit_msg)
        
        # Binance connection check
        binance_valid, binance_msg = self.validate_binance_connection()
        results['checks']['binance_connection'] = {
            'valid': binance_valid,
            'message': binance_msg
        }
        
        if not binance_valid:
            results['warnings'].append(binance_msg)
        
        # Symbol availability check
        symbol_valid, symbol_msg = self.validate_symbol_availability()
        results['checks']['symbol_availability'] = {
            'valid': symbol_valid,
            'message': symbol_msg
        }
        
        if not symbol_valid:
            results['overall_valid'] = False
            results['errors'].append(symbol_msg)
        
        return results


def main():
    """Main function to run validation"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    print("Trading Bot Configuration Validator")
    print("===================================")
    
    validator = ConfigValidator()
    results = validator.run_full_validation()
    
    print("\nValidation Results:")
    print("==================")
    
    # Print individual checks
    for check_name, check_result in results['checks'].items():
        status = "✓ PASS" if check_result['valid'] else "✗ FAIL"
        print(f"{check_name.replace('_', ' ').title()}: {status}")
        
        if 'message' in check_result:
            print(f"  {check_result['message']}")
        
        if 'errors' in check_result and check_result['errors']:
            for error in check_result['errors']:
                print(f"  ERROR: {error}")
    
    # Print summary
    print(f"\nOverall Status: {'✓ VALID' if results['overall_valid'] else '✗ INVALID'}")
    
    if results['errors']:
        print("\nErrors that must be fixed:")
        for error in results['errors']:
            print(f"  - {error}")
    
    if results['warnings']:
        print("\nWarnings:")
        for warning in results['warnings']:
            print(f"  - {warning}")
    
    if results['overall_valid']:
        print("\n✓ Configuration is valid. You can run the trading bot.")
    else:
        print("\n✗ Configuration has errors. Please fix them before running the bot.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())