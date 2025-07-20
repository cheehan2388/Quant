# Bybit CCXT Template

A comprehensive Python template for interacting with the Bybit cryptocurrency exchange using the CCXT library. This template provides a complete set of tools for market data retrieval, trading operations, and account management.

## Features

- 🔐 **Secure Authentication**: API key and secret management with environment variables
- 📊 **Market Data**: Real-time ticker, OHLCV, order book, and historical data
- 💱 **Trading Operations**: Market and limit orders with comprehensive error handling
- 📈 **Position Management**: Track and manage trading positions
- ⚡ **Rate Limiting**: Built-in rate limiting to comply with API restrictions
- 🛡️ **Risk Management**: Position sizing and risk calculation tools
- 📝 **Comprehensive Logging**: Detailed logging for debugging and monitoring
- 🔧 **Configurable**: Easy-to-use configuration system

## Prerequisites

- Python 3.7+
- CCXT library (already included in requirements.txt)
- Bybit API credentials

## Installation

1. **Clone or download the template files**:
   ```bash
   # The template files are already in your workspace
   ls -la bybit_ccxt_template.py bybit_config.py
   ```

2. **Install dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your environment variables**:
   Create a `.env` file in your project root:
   ```bash
   # .env file
   BYBIT_API_KEY=your_api_key_here
   BYBIT_SECRET=your_secret_here
   BYBIT_SANDBOX=true  # Set to false for live trading
   ```

## Quick Start

### Basic Usage

```python
from bybit_ccxt_template import BybitCCXTTemplate

# Initialize the template
bybit = BybitCCXTTemplate(sandbox=True)  # Use sandbox for testing

# Test connection
if bybit.test_connection():
    print("Connected to Bybit!")

# Get account information
account_info = bybit.get_account_info()
print(f"Account balance: {account_info['total_balance']}")

# Get current BTC price
ticker = bybit.get_ticker('BTC/USDT')
print(f"BTC price: ${ticker['last']}")
```

### Market Data Examples

```python
# Get OHLCV data
ohlcv = bybit.get_ohlcv('BTC/USDT', '1h', 24)
print(f"Retrieved {len(ohlcv)} hours of data")

# Get order book
order_book = bybit.get_order_book('BTC/USDT', 10)
print(f"Bid price: {order_book['bids'][0][0]}")
print(f"Ask price: {order_book['asks'][0][0]}")

# Get historical trades
trades = bybit.get_historical_trades('BTC/USDT', 50)
print(f"Retrieved {len(trades)} recent trades")
```

### Trading Examples

```python
# Place a limit order (commented out for safety)
# order = bybit.place_limit_order('BTC/USDT', 'buy', 0.001, 50000)
# print(f"Order placed: {order['id']}")

# Place a market order (commented out for safety)
# order = bybit.place_market_order('BTC/USDT', 'buy', 0.001)
# print(f"Market order placed: {order['id']}")

# Get open orders
open_orders = bybit.get_open_orders()
print(f"You have {len(open_orders)} open orders")

# Cancel an order
# bybit.cancel_order('order_id', 'BTC/USDT')
```

### Risk Management

```python
# Calculate position size based on risk
position_size = bybit.calculate_position_size(
    symbol='BTC/USDT',
    risk_percentage=2.0,  # 2% risk per trade
    stop_loss_percentage=5.0  # 5% stop loss
)
print(f"Recommended position size: {position_size}")
```

## Configuration

The template uses a comprehensive configuration system in `bybit_config.py`:

### API Configuration
```python
BYBIT_CONFIG = {
    'api_key': 'your_api_key',
    'secret': 'your_secret',
    'sandbox': True,  # Use sandbox for testing
    'enable_rate_limit': True,
    'default_type': 'spot',  # 'spot', 'linear', 'inverse'
}
```

### Trading Configuration
```python
TRADING_CONFIG = {
    'default_symbol': 'BTC/USDT',
    'default_timeframe': '1h',
    'default_limit': 100,
    'max_retries': 3,
    'retry_delay': 1,
    'order_timeout': 300,
}
```

### Risk Management
```python
RISK_CONFIG = {
    'max_risk_per_trade': 2.0,  # percentage
    'max_position_size': 10.0,  # percentage
    'default_stop_loss': 5.0,  # percentage
    'default_take_profit': 10.0,  # percentage
    'max_open_orders': 5,
}
```

## Available Methods

### Connection & Authentication
- `test_connection()` - Test API connection
- `get_account_info()` - Get account balances and information

### Market Data
- `get_markets()` - Get available trading pairs
- `get_ticker(symbol)` - Get current price and volume
- `get_order_book(symbol, limit)` - Get order book
- `get_ohlcv(symbol, timeframe, limit)` - Get OHLCV data
- `get_historical_trades(symbol, limit)` - Get recent trades
- `get_funding_rate(symbol)` - Get funding rate (futures)

### Trading Operations
- `place_market_order(symbol, side, amount)` - Place market order
- `place_limit_order(symbol, side, amount, price)` - Place limit order
- `cancel_order(order_id, symbol)` - Cancel order
- `get_order_status(order_id, symbol)` - Get order status
- `get_open_orders(symbol)` - Get all open orders
- `wait_for_order_completion(order_id, symbol, timeout)` - Wait for order completion

### Position Management
- `get_positions()` - Get current positions (futures)
- `calculate_position_size(symbol, risk_percentage, stop_loss_percentage)` - Calculate position size

## Error Handling

The template includes comprehensive error handling:

```python
try:
    ticker = bybit.get_ticker('BTC/USDT')
    print(f"Price: ${ticker['last']}")
except Exception as e:
    print(f"Error getting ticker: {e}")
    # Handle the error appropriately
```

Common error scenarios:
- **API Key/Secret missing**: Check your environment variables
- **Network issues**: Check your internet connection
- **Rate limiting**: Wait before making more requests
- **Insufficient balance**: Check your account balance
- **Invalid symbol**: Verify the trading pair exists

## Best Practices

### 1. Always Use Sandbox First
```python
# Test with sandbox before live trading
bybit = BybitCCXTTemplate(sandbox=True)
```

### 2. Implement Proper Error Handling
```python
try:
    order = bybit.place_limit_order('BTC/USDT', 'buy', 0.001, 50000)
except Exception as e:
    logger.error(f"Order failed: {e}")
    # Implement fallback strategy
```

### 3. Use Risk Management
```python
# Calculate position size based on risk
position_size = bybit.calculate_position_size('BTC/USDT', 2.0, 5.0)
```

### 4. Monitor Your Orders
```python
# Wait for order completion
completed_order = bybit.wait_for_order_completion(order_id, 'BTC/USDT')
```

### 5. Log Everything
```python
import logging
logging.basicConfig(level=logging.INFO)
# The template includes comprehensive logging
```

## Security Considerations

1. **Never hardcode API credentials** - Always use environment variables
2. **Use sandbox for testing** - Test thoroughly before live trading
3. **Implement proper error handling** - Don't let errors crash your application
4. **Monitor your API usage** - Stay within rate limits
5. **Secure your environment** - Keep your `.env` file secure

## Rate Limiting

Bybit has rate limits on API calls. The template includes built-in rate limiting:

```python
# The template automatically handles rate limiting
bybit = BybitCCXTTemplate(enable_rate_limit=True)
```

## Testing

Run the template to test your setup:

```bash
python bybit_ccxt_template.py
```

This will:
1. Test the connection to Bybit
2. Get account information
3. Retrieve market data
4. Display a summary

## Troubleshooting

### Common Issues

1. **"API key and secret are required"**
   - Check your `.env` file
   - Verify environment variables are set correctly

2. **"Connection test failed"**
   - Check your internet connection
   - Verify API credentials are correct
   - Ensure you're using the right sandbox/live environment

3. **"Rate limit exceeded"**
   - Wait before making more requests
   - Implement exponential backoff

4. **"Insufficient balance"**
   - Check your account balance
   - Verify you have enough funds for the trade

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Support

For issues with the template:
1. Check the error messages and logs
2. Verify your configuration
3. Test with sandbox mode first
4. Check Bybit's API documentation for updates

## License

This template is provided as-is for educational and development purposes. Use at your own risk when trading with real money.

## Disclaimer

This template is for educational purposes only. Cryptocurrency trading involves significant risk. Always:
- Test thoroughly with sandbox mode
- Start with small amounts
- Implement proper risk management
- Never invest more than you can afford to lose