#!/usr/bin/env python3
"""
Example usage of the Bybit CCXT template
This script demonstrates basic functionality of the template.
"""

import os
import sys
from bybit_ccxt_template import BybitCCXTTemplate
from bybit_config import validate_config, get_config_summary

def main():
    """
    Example usage of the Bybit CCXT template.
    """
    print("=== Bybit CCXT Template Example ===\n")
    
    # Validate configuration
    print("1. Validating configuration...")
    if not validate_config():
        print("❌ Configuration validation failed!")
        print("Please set up your .env file with BYBIT_API_KEY and BYBIT_SECRET")
        return
    
    print("✅ Configuration is valid!")
    
    # Show configuration summary
    summary = get_config_summary()
    print(f"Configuration summary: {summary}\n")
    
    try:
        # Initialize the template
        print("2. Initializing Bybit CCXT template...")
        bybit = BybitCCXTTemplate(sandbox=True)  # Use sandbox for safety
        print("✅ Template initialized successfully!")
        
        # Test connection
        print("\n3. Testing connection to Bybit...")
        if not bybit.test_connection():
            print("❌ Failed to connect to Bybit!")
            return
        print("✅ Successfully connected to Bybit!")
        
        # Get account information
        print("\n4. Getting account information...")
        account_info = bybit.get_account_info()
        print(f"✅ Account information retrieved!")
        print(f"   Total balance: {account_info['total_balance']}")
        print(f"   Free balance: {account_info['free_balance']}")
        print(f"   Used balance: {account_info['used_balance']}")
        
        # Get markets
        print("\n5. Loading available markets...")
        markets = bybit.get_markets()
        print(f"✅ Loaded {len(markets)} markets")
        
        # Get ticker for BTC/USDT
        print("\n6. Getting BTC/USDT ticker...")
        ticker = bybit.get_ticker('BTC/USDT')
        print(f"✅ BTC/USDT ticker retrieved!")
        print(f"   Last price: ${ticker['last']:,.2f}")
        print(f"   Bid: ${ticker['bid']:,.2f}")
        print(f"   Ask: ${ticker['ask']:,.2f}")
        print(f"   Volume (24h): {ticker['baseVolume']:,.2f} BTC")
        print(f"   Change (24h): {ticker['percentage']:.2f}%")
        
        # Get OHLCV data
        print("\n7. Getting OHLCV data for BTC/USDT...")
        ohlcv = bybit.get_ohlcv('BTC/USDT', '1h', 24)
        print(f"✅ Retrieved {len(ohlcv)} hours of OHLCV data")
        print(f"   Latest close: ${ohlcv['close'].iloc[-1]:,.2f}")
        print(f"   Highest (24h): ${ohlcv['high'].max():,.2f}")
        print(f"   Lowest (24h): ${ohlcv['low'].min():,.2f}")
        
        # Get order book
        print("\n8. Getting order book for BTC/USDT...")
        order_book = bybit.get_order_book('BTC/USDT', 5)
        print(f"✅ Order book retrieved!")
        print("   Top 5 bids:")
        for i, (price, amount) in enumerate(order_book['bids'][:5]):
            print(f"     {i+1}. ${price:,.2f} - {amount:.4f} BTC")
        print("   Top 5 asks:")
        for i, (price, amount) in enumerate(order_book['asks'][:5]):
            print(f"     {i+1}. ${price:,.2f} - {amount:.4f} BTC")
        
        # Get open orders
        print("\n9. Getting open orders...")
        open_orders = bybit.get_open_orders()
        print(f"✅ You have {len(open_orders)} open orders")
        
        # Calculate position size example
        print("\n10. Calculating position size example...")
        try:
            position_size = bybit.calculate_position_size(
                symbol='BTC/USDT',
                risk_percentage=2.0,  # 2% risk
                stop_loss_percentage=5.0  # 5% stop loss
            )
            print(f"✅ Position size calculated!")
            print(f"   Recommended position size: {position_size:.4f} BTC")
            print(f"   This represents 2% risk with 5% stop loss")
        except Exception as e:
            print(f"⚠️  Could not calculate position size: {e}")
        
        # Get funding rate (for futures)
        print("\n11. Getting funding rate for BTC/USDT...")
        try:
            funding_rate = bybit.get_funding_rate('BTC/USDT')
            print(f"✅ Funding rate retrieved!")
            print(f"   Current funding rate: {funding_rate.get('fundingRate', 'N/A')}")
            print(f"   Next funding time: {funding_rate.get('nextFundingTime', 'N/A')}")
        except Exception as e:
            print(f"⚠️  Could not get funding rate: {e}")
        
        print("\n=== Example completed successfully! ===")
        print("\nNext steps:")
        print("1. Review the code to understand how each method works")
        print("2. Modify the configuration in bybit_config.py")
        print("3. Test with your own API credentials")
        print("4. Implement your trading strategy")
        print("\n⚠️  Remember: Always test with sandbox mode first!")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Verify your API credentials")
        print("3. Ensure you're using the correct sandbox/live environment")
        print("4. Check the logs for more detailed error information")


def demo_market_data():
    """
    Demonstrate market data functionality.
    """
    print("\n=== Market Data Demo ===")
    
    try:
        bybit = BybitCCXTTemplate(sandbox=True)
        
        # Get multiple tickers
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        
        for symbol in symbols:
            try:
                ticker = bybit.get_ticker(symbol)
                print(f"{symbol}: ${ticker['last']:,.2f} ({ticker['percentage']:+.2f}%)")
            except Exception as e:
                print(f"{symbol}: Error - {e}")
                
    except Exception as e:
        print(f"Error in market data demo: {e}")


def demo_risk_management():
    """
    Demonstrate risk management functionality.
    """
    print("\n=== Risk Management Demo ===")
    
    try:
        bybit = BybitCCXTTemplate(sandbox=True)
        
        # Different risk scenarios
        scenarios = [
            {'risk': 1.0, 'stop_loss': 3.0, 'name': 'Conservative'},
            {'risk': 2.0, 'stop_loss': 5.0, 'name': 'Moderate'},
            {'risk': 3.0, 'stop_loss': 7.0, 'name': 'Aggressive'},
        ]
        
        for scenario in scenarios:
            try:
                position_size = bybit.calculate_position_size(
                    symbol='BTC/USDT',
                    risk_percentage=scenario['risk'],
                    stop_loss_percentage=scenario['stop_loss']
                )
                print(f"{scenario['name']} ({scenario['risk']}% risk, {scenario['stop_loss']}% SL): {position_size:.4f} BTC")
            except Exception as e:
                print(f"{scenario['name']}: Error - {e}")
                
    except Exception as e:
        print(f"Error in risk management demo: {e}")


if __name__ == "__main__":
    # Run main example
    main()
    
    # Run additional demos
    demo_market_data()
    demo_risk_management()