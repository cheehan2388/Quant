class Config:
    """Configuration class for the trading system"""
    
    # Logging configuration
    LOG_LEVEL = "INFO"  # 日誌級別: DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FILE = "trading_system.log"  # 日誌文件名稱
    
    # Exchange API credentials
    BINANCE_API_KEY = "RGANDaKqgAe8hkj0uClGJ74wSJym2cNSex6EMfyDF4qEuM4D4gAYQZEK7fKA4Tnc"  # Binance API 密鑰
    BINANCE_SECRET_KEY = "Vbuvz9EXg9fadJH8ioaX4TIRwK2ORBPdH6WspFuYKYkbVfTIH33hZpnyNN9Xpyj8"  # Binance 秘密密鑰
    BINANCE_SANDBOX = False  # 是否使用 Binance 測試網 (True/False)
    
    BYBIT_API_KEY = "meq8oBzBjp9cYV64Do"  # Bybit API 密鑰
    BYBIT_SECRET_KEY = "oOOa4u4NISAjMxHblOhQcBG4n9MYq6b9cZ7H"  # Bybit 秘密密鑰
    BYBIT_SANDBOX = False  # 是否使用 Bybit 測試網 (True/False)
    
    # Trading parameters
    TRADING_SYMBOL = "BTCUSDT"  # 交易對，例如 "BTCUSDT"
    DATA_FETCH_INTERVAL = 300  # 數據抓取間隔（秒），例如 300 秒（5 分鐘）
    LOOKBACK_PERIOD = 20  # 歷史數據回看週期（用於計算 Z 分數等）
    
    # Z-Score strategy parameters
    ZSCORE_LONG_THRESHOLD = 2.0  # Z 分數做多閾值
    ZSCORE_SHORT_THRESHOLD = -2.0  # Z 分數做空閾值
    
    # Risk management parameters
    MAX_POSITION_SIZE = 0.01  # 最大倉位大小（例如 0.01 表示 1% 的帳戶資金）
    STOP_LOSS_PCT = 0.02  # 止損百分比（例如 0.02 表示 2%）
    TAKE_PROFIT_PCT = 0.04  # 止盈百分比（例如 0.04 表示 4%）

def validate_config():
    """Validate configuration parameters"""
    required_fields = [
        "BINASNCE_API_KEY", "BINANCE_SECRET_KEY",
        "BYBIT_API_KEY", "BYBIT_SECRET_KEY",
        "TRADING_SYMBOL", "LOG_LEVEL", "LOG_FILE"
    ]
    
    for field in required_fields:
        if not hasattr(Config, field) or not getattr(Config, field):
            raise ValueError(f"Configuration error: {field} is missing or empty")
    
    # Validate numeric parameters
    if Config.DATA_FETCH_INTERVAL <= 0:
        raise ValueError("DATA_FETCH_INTERVAL must be positive")
    if Config.LOOKBACK_PERIOD <= 0:
        raise ValueError("LOOKBACK_PERIOD must be positive")
    if Config.MAX_POSITION_SIZE <= 0:
        raise ValueError("MAX_POSITION_SIZE must be positive")
    if Config.STOP_LOSS_PCT <= 0:
        raise ValueError("STOP_LOSS_PCT must be positive")
    if Config.TAKE_PROFIT_PCT <= 0:
        raise ValueError("TAKE_PROFIT_PCT must be positive")