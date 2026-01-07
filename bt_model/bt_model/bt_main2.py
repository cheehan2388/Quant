import pandas as pd
import time
import logging
from typing import List, Dict, Tuple
from strategy_module import (
    StrategyType,
    STRATEGY_GENERATORS,
    get_param_ranges
)
from backtester_engine1 import run_grid_search, Backtester#, create_batches
from data_composer import advanced_multi_compose_data
from functools import partial
from backtest_dashboard import run_dashboard
import multiprocessing as mp
from itertools import product

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COMPOSE_COLS: List[str] = ['close', 'close_premium']
BATCH_SIZE = 50


def create_strategy(strategy_type, window: int, entry_threshold: float, exit_threshold: float, composed_col: str):
    window = int(window)
    entry_threshold = float(entry_threshold)
    exit_threshold = float(exit_threshold)
    return STRATEGY_GENERATORS[strategy_type](
        window,
        entry_threshold,
        exit_threshold,
        [composed_col],
        'identity'
    )

def calculate_buy_hold_metrics(df: pd.DataFrame, interval: int) -> Dict:
    logger.info("Calculating Buy-and-Hold metrics...")
    dummy_backtester = Backtester(df.copy(), None, interval)
    buy_hold_metrics = dummy_backtester.calculate_buy_hold_metrics()
    return {
        'buy_hold_sharpe': float(buy_hold_metrics.sharpe_ratio),
        'buy_hold_max_drawdown': float(buy_hold_metrics.max_drawdown),
        'buy_hold_total_return': float(buy_hold_metrics.total_return)
    }

def optimize_result_for_dashboard(result: dict, strategy_type: str, composition_name: str) -> dict:
    try:
        return {
            'strategy_type': strategy_type,
            'composition': composition_name,
            'window': int(result.get('window', 0)),
            'entry_threshold': float(result.get('entry_threshold', 0)),
            'exit_threshold': float(result.get('exit_threshold', 0)),
            'sharpe_ratio': float(result.get('sharpe_ratio', 0)),
            'max_drawdown': float(result.get('max_drawdown', 0)),
            'total_return': float(result.get('total_return', 0)),
            'total_trade': int(result.get('total_trade', 0))
        }
    except Exception as e:
        logger.error(f"Error optimizing result: {str(e)}")
        logger.debug(f"Problem result: {result}")
        raise


def run_strategy_combination(params: Tuple[StrategyType, str, pd.DataFrame, int, int, float, float]) -> Dict:
    strategy_type, composition_name, data, interval, window, entry, exit = params
    try:
        strategy = create_strategy(strategy_type, window, entry, exit, composition_name)
        backtester = Backtester(data.copy(), strategy, interval)
        result, _ = backtester.run()
        return optimize_result_for_dashboard(
            result.__dict__,
            strategy_type.value,
            composition_name
        )
    except Exception as e:
        logger.error(f"Error in run_strategy_combination: {str(e)}")
        return None


def main():
    start_time = time.time()
    dashboard_results = []
    buy_hold_metrics = None

    try:
        # Load and prepare data
        logger.info("Loading data...")
        # df = pd.read_csv('C:\\Users\\User\\Desktop\\data\\BTC_binance_k_pi_15m.parquet')
        df = pd.read_parquet('C:\\Users\\User\\Desktop\\data\\BTC_premium_1h.parquet')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        interval = 60

        # Calculate buy-hold metrics once
        buy_hold_metrics = calculate_buy_hold_metrics(df, interval)
        logger.info(f"Buy-and-Hold metrics calculated: {buy_hold_metrics}")

        composed_dfs = advanced_multi_compose_data(df, COMPOSE_COLS)

        # Create all parameter combinations
        all_params = []
        for composition_name, composed_df in composed_dfs.items():
            for strategy_type in StrategyType:
                window_range, entry_range, exit_range = get_param_ranges(strategy_type)
                params = product(
                    [strategy_type],
                    [composition_name],
                    [composed_df],
                    [interval],
                    window_range,
                    entry_range,
                    exit_range
                )
                all_params.extend(params)

        # Process combinations in parallel
        n_processes = mp.cpu_count()
        with mp.Pool(processes=n_processes) as pool:
            results = pool.map(run_strategy_combination, all_params)
            dashboard_results = [r for r in results if r is not None]

        elapsed_time = time.time() - start_time
        logger.info(f"Backtest completed in {elapsed_time:.2f} seconds")

        if dashboard_results:
            logger.info(f"Launching dashboard with {len(dashboard_results)} results...")
            logger.debug(f"Sample of results: {dashboard_results[:2]}")
            logger.debug(f"Buy-hold metrics: {buy_hold_metrics}")
            run_dashboard(dashboard_results, buy_hold_metrics, port=8050)
        else:
            logger.warning("No results to display in dashboard")
            logger.debug("Results list is empty")
            logger.debug(f"Buy-hold metrics: {buy_hold_metrics}")

    except KeyboardInterrupt:
        logger.info("\nBacktest interrupted by user")
    except Exception as e:
        logger.error(f"Error during backtest: {str(e)}", exc_info=True)
        logger.error(f"Results length: {len(dashboard_results) if dashboard_results else 0}")
        logger.error(f"Buy-hold metrics: {buy_hold_metrics}")
        raise


if __name__ == "__main__":
    main()