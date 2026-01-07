# backtester_engine.py
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from itertools import product
import multiprocessing as mp
from functools import partial


@dataclass
class StrategyResult:
    window: int
    entry_threshold: float
    exit_threshold: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    annualized_return: float
    calmar_ratio: float
    total_trade: int


@dataclass
class BuyHoldResult:
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    interval: int
    composition: str


@dataclass
class BatchResult:
    window_range: Tuple[int, int]
    strategy_results: List[StrategyResult]
    indicators: Dict[str, pd.DataFrame]


class Backtester:
    def __init__(self, data: pd.DataFrame, strategy: Any, interval: int, initial_balance: float = 1000,
                 transaction_cost: float = 0.0005):
        self.data = data
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.interval = interval
        self.annualization_factor = self._calculate_annualization_factor(interval)
        self.results_df = None
        self.buy_hold_metrics = None

    def _calculate_annualization_factor(self, interval: int) -> float:
        intervals = {
            1: 365 * 24 * 60,  # 1-minute data
            5: 365 * 24 * 12,  # 5-minute data
            10: 365 * 24 * 6,  # 10-minute data
            15: 365 * 24 * 4,  # 15-minute data
            30: 365 * 24 * 2,  # 30-minute data
            60: 365 * 24,  # 1-hour data
            240: 365 * 6,  # 4-hour data
            1440: 365  # Daily data
        }
        return np.sqrt(intervals.get(interval, 365))

    def calculate_buy_hold_metrics(self) -> BuyHoldResult:
        if self.buy_hold_metrics is None:
            df = self.data.copy()
            df['returns'] = df['close'].pct_change()
            df['buy_hold_returns'] = df['returns']
            df['buy_hold_cumulative_returns'] = (1 + df['buy_hold_returns']).cumprod()
            df['buy_hold_equity_curve'] = self.initial_balance * df['buy_hold_cumulative_returns']
            df['buy_hold_drawdown'] = (df['buy_hold_equity_curve'].cummax() - df['buy_hold_equity_curve']) / df[
                'buy_hold_equity_curve'].cummax()

            buy_hold_total_return = (df['buy_hold_equity_curve'].iloc[-1] / self.initial_balance) - 1
            buy_hold_sharpe_ratio = self.annualization_factor * df['buy_hold_returns'].mean() / df[
                'buy_hold_returns'].std()
            buy_hold_max_drawdown = df['buy_hold_drawdown'].max()

            self.buy_hold_metrics = BuyHoldResult(
                sharpe_ratio=buy_hold_sharpe_ratio,
                max_drawdown=buy_hold_max_drawdown,
                total_return=buy_hold_total_return,
                interval=self.interval,
                composition=''
            )

        return self.buy_hold_metrics

    def run(self) -> Tuple[StrategyResult, BuyHoldResult]:
        df = self.strategy.calculate_signals(self.data.copy())
        df['returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['pos_t-1'] * df['returns'] - df['trade'] * self.transaction_cost
        df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
        df['equity_curve'] = self.initial_balance * df['cumulative_returns']
        df['drawdown'] = (df['equity_curve'].cummax() - df['equity_curve']) / df['equity_curve'].cummax()
        df['total_trades'] = df['trade'].cumsum()

        total_return = (df['equity_curve'].iloc[-1] / self.initial_balance) - 1
        sharpe_ratio = self.annualization_factor * df['strategy_returns'].mean() / df['strategy_returns'].std()
        max_drawdown = df['drawdown'].max()
        total_trade = df['total_trades'].iloc[-1]

        annualized_return = df['strategy_returns'].mean() * (self.annualization_factor ** 2)
        calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else np.inf

        self.results_df = df

        strategy_result = StrategyResult(
            window=self.strategy.window,
            entry_threshold=self.strategy.entry_threshold,
            exit_threshold=self.strategy.exit_threshold,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_return=total_return,
            annualized_return=annualized_return,
            calmar_ratio=calmar_ratio,
            total_trade=total_trade
        )

        return strategy_result, self.calculate_buy_hold_metrics()


def create_batches(window_start: int, window_end: int, batch_size: int = 50) -> List[Tuple[int, int]]:
    """Create window range batches."""
    ranges = []
    for start in range(window_start, window_end, batch_size):
        end = min(start + batch_size, window_end)
        ranges.append((start, end))
    return ranges


def process_batch(batch_params: Tuple[Tuple[int, int], List[Tuple[float, float, float]], pd.DataFrame, int, Any]) -> \
List[Tuple[StrategyResult, BuyHoldResult]]:
    window_range, param_combinations, data, interval, strategy_class = batch_params
    results = []

    # Pre-calculate indicators for the window range
    indicators = {}
    for window in range(window_range[0], window_range[1]):
        df = data.copy()
        # Store rolling calculations
        indicators[window] = {
            'mean': df['close'].rolling(window=window).mean(),
            'std': df['close'].rolling(window=window).std(),
            'min': df['close'].rolling(window=window).min(),
            'max': df['close'].rolling(window=window).max()
        }

    # Process parameter combinations for this window range
    for window, entry, exit in param_combinations:
        if window_range[0] <= window < window_range[1]:
            strategy = strategy_class(window, entry, exit)
            strategy._cached_indicators = indicators[window]  # Pass pre-calculated indicators
            backtester = Backtester(data, strategy, interval)
            results.append(backtester.run())

    return results


def run_grid_search(data: pd.DataFrame,
                    window_range: List[int],
                    entry_threshold_range: List[float],
                    exit_threshold_range: List[float],
                    interval: int,
                    strategy_class: Any,
                    n_processes: int = None,
                    batch_size: int = 50) -> Tuple[pd.DataFrame, BuyHoldResult]:
    if n_processes is None:
        n_processes = mp.cpu_count()

    # Create batches for windows
    window_batches = create_batches(min(window_range), max(window_range) + 1, batch_size)

    # Generate all parameter combinations
    all_params = list(product(window_range, entry_threshold_range, exit_threshold_range))

    # Prepare batch parameters
    batch_params = [(w_range, all_params, data, interval, strategy_class)
                    for w_range in window_batches]

    # Initialize the multiprocessing pool
    with mp.Pool(processes=n_processes) as pool:
        try:
            # Process batches in parallel
            batch_results = pool.map(process_batch, batch_params)

            # Flatten results
            all_results = [result for batch in batch_results for result in batch]

            # Split strategy results and buy-hold results
            strategy_results, buy_hold_results = zip(*all_results)

            # Convert to DataFrame
            strategy_df = pd.DataFrame(strategy_results)

            return strategy_df, buy_hold_results[0]

        except Exception as e:
            pool.terminate()
            raise Exception(f"Error during grid search: {str(e)}")

        finally:
            pool.close()
            pool.join()