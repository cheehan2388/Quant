# Z-Score Trading Bot for Bybit Futures

A comprehensive trading bot that trades Bybit futures contracts based on Z-score analysis of combined Open Interest and Long/Short Ratio indicators from Binance.

## Features

- **Multi-Exchange Data**: Fetches Open Interest and Long/Short Ratio from Binance
- **Advanced Strategy**: Uses Z-score analysis with rolling window calculations
- **Position Management**: Tracks positions and prevents overlapping trades
- **Risk Management**: Time-based position exits and comprehensive error handling
- **Comprehensive Logging**: Detailed logging with file and console output
- **Configuration Validation**: Built-in validator for API keys and settings
- **Stale Data Handling**: Detects and handles cases where data may not be updated

## Strategy Overview

1. **Data Collection**: Fetches Open Interest and Long/Short Ratio from Binance
2. **Combined Indicator**: Multiplies Open Interest × Long/Short Ratio
3. **Z-Score Calculation**: Calculates Z-score using a rolling window (default: 15 periods)
4. **Signal Generation**:
   - Z-score > 2.1 → Short signal
   - Z-score < -2.1 → Long signal
   - Otherwise → Hold
5. **Position Management**: Only one position at a time, with time-based exits

## Installation

1. **Clone the repository** (or create the project structure):
```bash
git clone <repository-url>
cd trading-bot
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Setup environment variables**:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Bybit API Keys (Required - for trading)
BYBIT_API_KEY=your_bybit_api_key_here
BYBIT_SECRET=your_bybit_secret_here

# Binance API Keys (Optional - for data fetching)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET=your_binance_secret_here

# Trading Configuration
SYMBOL=BTCUSDT
POSITION_SIZE=0.001
Z_SCORE_THRESHOLD=2.1
ROLLING_WINDOW=15
```

### API Key Setup

#### Bybit API Keys (Required)
1. Go to [Bybit API Management](https://www.bybit.com/app/user/api-management)
2. Create a new API key with the following permissions:
   - **Derivatives**: Read, Trade
   - **Wallet**: Read (for balance checking)
3. Add your IP address to the whitelist
4. **Important**: Start with testnet for testing

#### Binance API Keys (Optional)
1. Go to [Binance API Management](https://www.binance.com/en/my/settings/api-management)
2. Create a new API key
3. For data fetching only, you can use read-only permissions
4. The bot can work with public Binance data if no API keys are provided

## Usage

### 1. Validate Configuration

Before running the bot, validate your configuration:

```bash
python utils/config_validator.py
```

This will check:
- Environment variables
- API key validity
- Exchange connections
- Symbol availability

### 2. Test Individual Components

Test the Binance data fetcher:
```bash
python data_fetcher/binance_indicators.py
```

Test the trading strategy:
```bash
python trading/strategy.py
```

### 3. Run the Trading Bot

**For testing (recommended first):**
```bash
# Edit main_trading_bot.py and set sandbox=True in the Bybit exchange initialization
python main_trading_bot.py
```

**For live trading:**
```bash
# Ensure sandbox=False in main_trading_bot.py
python main_trading_bot.py
```

## Project Structure

```
trading-bot/
├── main_trading_bot.py              # Main bot orchestrator
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variables template
├── README.md                        # This file
├── data_fetcher/
│   ├── binance_indicators.py        # Binance data fetching
│   └── binance_df.py               # Original Binance data fetcher
├── trading/
│   ├── strategy.py                  # Z-score trading strategy
│   ├── position_manager.py          # Position tracking
│   └── order_manager.py             # Order execution
├── utils/
│   └── config_validator.py          # Configuration validation
├── logs/                            # Log files (created automatically)
└── live_trade/
    └── live_trade_sample.py         # Original sample code
```

## Key Components

### 1. BinanceIndicatorsFetcher (`data_fetcher/binance_indicators.py`)
- Fetches Open Interest data
- Fetches Long/Short Ratio data
- Combines indicators
- Handles API rate limits and errors

### 2. ZScoreStrategy (`trading/strategy.py`)
- Maintains rolling window of indicator values
- Calculates Z-scores
- Generates trading signals
- Handles stale data detection

### 3. PositionManager (`trading/position_manager.py`)
- Tracks current positions
- Prevents overlapping trades
- Manages position state

### 4. OrderManager (`trading/order_manager.py`)
- Executes market orders
- Handles order status tracking
- Manages position opening/closing

### 5. TradingBot (`main_trading_bot.py`)
- Orchestrates all components
- Main execution loop
- Comprehensive logging
- Graceful shutdown handling

## Logging

The bot creates comprehensive logs in the `logs/` directory:
- File: `logs/trading_bot_YYYYMMDD_HHMMSS.log`
- Console output with the same information
- Log levels: INFO for normal operations, ERROR for issues

## Safety Features

1. **Position Limits**: Only one position at a time
2. **Time-based Exits**: Positions are closed after 1 hour (configurable)
3. **Stale Data Detection**: Prevents trading on outdated data
4. **API Error Handling**: Robust error handling for network issues
5. **Graceful Shutdown**: Proper cleanup on exit signals

## Risk Warnings

⚠️ **Important Risk Disclaimers:**

1. **This is experimental software** - Use at your own risk
2. **Start with small position sizes** - Test thoroughly before scaling
3. **Use testnet first** - Always test on Bybit testnet before live trading
4. **Monitor continuously** - Don't leave the bot unattended for long periods
5. **Market conditions** - The strategy may not work in all market conditions
6. **API limits** - Be aware of exchange API rate limits
7. **No guarantees** - Past performance doesn't guarantee future results

## Customization

### Modifying the Strategy

To modify the trading strategy, edit `trading/strategy.py`:
- Change Z-score thresholds
- Modify rolling window calculations
- Add additional indicators
- Implement different signal logic

### Adding Risk Management

You can extend `trading/position_manager.py` or `main_trading_bot.py` to add:
- Stop-loss orders
- Take-profit levels
- Position sizing based on volatility
- Maximum drawdown limits

### Data Sources

To use different data sources, modify `data_fetcher/binance_indicators.py` or create new fetcher classes.

## Troubleshooting

### Common Issues

1. **API Authentication Errors**:
   - Verify API keys are correct
   - Check API permissions
   - Ensure IP whitelist is configured

2. **Network Errors**:
   - Check internet connection
   - Verify exchange status
   - Consider rate limiting

3. **Insufficient Balance**:
   - Ensure adequate USDT balance
   - Check position size configuration

4. **Symbol Not Found**:
   - Verify symbol format
   - Check if symbol is available on both exchanges

### Debug Mode

To enable more verbose logging, modify the logging level in `main_trading_bot.py`:
```python
# Change INFO to DEBUG
root_logger.setLevel(logging.DEBUG)
```

## Performance Monitoring

The bot logs key performance metrics:
- Total signals generated
- Total trades executed
- Position duration
- Z-score statistics

Monitor these metrics to evaluate strategy performance.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is provided as-is for educational purposes. Use at your own risk.

## Disclaimer

This software is for educational and research purposes only. Trading cryptocurrencies involves substantial risk of loss. The authors are not responsible for any financial losses incurred through the use of this software. Always do your own research and consider consulting with a financial advisor before trading.