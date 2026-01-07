"""
Volatility arbitrage research and backtesting module.

Contents:
- har_rv: Realized volatility computations and HAR-RV forecasting utilities
- signals: Variance risk premium (VRP) signal and position sizing
- execution: Simplified variance-swap PnL simulation for straddle-like exposure
- backtest: Example end-to-end backtest runner
"""

from . import har_rv, signals, execution

__all__ = [
    "har_rv",
    "signals",
    "execution",
]


