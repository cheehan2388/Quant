import pandas as pd
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from enum import Enum


class StrategyType(Enum):
    # Mean Reversion Strategies
    MR_ZSCORE_LONG = "MR Z-Score Long"
    MR_ZSCORE_SHORT = "MR Z-Score Short"
    MR_MINMAX_LONG = "MR MinMax Long"
    MR_MINMAX_SHORT = "MR MinMax Short"
    MR_MINMAX_FLIP = "MR MinMax Flip"
    # Trend Following Strategies
    TF_ZSCORE_LONG = "TF Z-Score Long"
    TF_ZSCORE_SHORT = "TF Z-Score Short"
    TF_MINMAX_LONG = "TF MinMax Long"
    TF_MINMAX_SHORT = "TF MinMax Short"

@dataclass
class StrategyParams:
    window_start: int = 50
    window_end: int = 311
    window_step: int = 20
    entry_start: float = -2.5
    entry_end: float = -0.5
    entry_step: float = 0.2
    exit_start: float = 0.5
    exit_end: float = 2.5
    exit_step: float = 0.05

class StrategyBase:
    def __init__(self, window: int, entry_threshold: float, exit_threshold: float, compose_cols: List[str], compose_operation: str):
        self.window = window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.compose_cols = compose_cols
        self.compose_operation = compose_operation

    def calculate_zscore(self, series: pd.Series) -> pd.Series:
        rolling_mean = series.rolling(window=self.window).mean().shift()
        rolling_std = series.rolling(window=self.window).std().shift()
        return (series - rolling_mean) / rolling_std

    def calculate_minmax(self, series: pd.Series) -> pd.Series:
        rolling_min = series.rolling(window=self.window).min().shift()
        rolling_max = series.rolling(window=self.window).max().shift()
        denominator = (rolling_max - rolling_min)
        denominator = np.where(denominator == 0, 1, denominator)
        return pd.Series((series - rolling_min) / denominator, index=series.index)

    def check_crossover(self, series: pd.Series, threshold: float) -> pd.Series:
        series_prev = series.shift(1)
        return (series_prev <= threshold) & (series > threshold)

    def check_crossunder(self, series: pd.Series, threshold: float) -> pd.Series:
        series_prev = series.shift(1)
        return (series_prev >= threshold) & (series < threshold)

    def _process_signals(self, df: pd.DataFrame, signal_series: pd.Series) -> pd.DataFrame:
        df['signal'] = signal_series
        df['pos'] = df['signal'].ffill()
        df['pos_t-1'] = df['pos'].shift(1)
        df['trade'] = (df['pos'] != df['pos_t-1']).astype(int)
        return df

# Mean Reversion Strategies
# class MR_ZScoreLongStrategy(StrategyBase):
#     def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
#         composed_col = self.compose_cols[0]
#         df['z'] = self.calculate_zscore(df[composed_col])
#         signals = np.where(df['z'] < self.entry_threshold, 1,
#                           np.where(df['z'] > self.exit_threshold, 0, np.nan))
#         return self._process_signals(df, signals)

# class MR_ZScoreShortStrategy(StrategyBase):
#     def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
#         composed_col = self.compose_cols[0]
#         df['z'] = self.calculate_zscore(df[composed_col])
#         signals = np.where(df['z'] > self.entry_threshold, -1,
#                           np.where(df['z'] < self.exit_threshold, 0, np.nan))
#         return self._process_signals(df, signals)

# class MR_MinMaxLongStrategy(StrategyBase):
#     def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
#         composed_col = self.compose_cols[0]
#         df['minmax'] = self.calculate_minmax(df[composed_col])
#         signals = np.where(df['minmax'] < self.entry_threshold, 1,
#                           np.where(df['minmax'] > self.exit_threshold, 0, np.nan))
#         return self._process_signals(df, signals)

# class MR_MinMaxShortStrategy(StrategyBase):
#     def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
#         composed_col = self.compose_cols[0]
#         df['minmax'] = self.calculate_minmax(df[composed_col])
#         signals = np.where(df['minmax'] > self.entry_threshold, -1,
#                           np.where(df['minmax'] < self.exit_threshold, 0, np.nan))
#         return self._process_signals(df, signals)

# class MR_MinMaxFlipStrategy(StrategyBase):
#     def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
#         composed_col = self.compose_cols[0]
#         df['minmax'] = self.calculate_minmax(df[composed_col])
        
#         signals = np.where(df['minmax'] < self.entry_threshold, 1,
#                           np.where(df['minmax'] > self.exit_threshold, -1, np.nan))
        
#         return self._process_signals(df, signals)

# Mean Reversion Strategies
class MR_ZScoreLongStrategy(StrategyBase):
    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        composed_col = self.compose_cols[0]
        df['z'] = self.calculate_zscore(df[composed_col])
        signals = np.where(df['z'] < self.entry_threshold, 1,
                          np.where(df['z'] > self.exit_threshold, 0, np.nan))
        return self._process_signals(df, signals)

class MR_ZScoreShortStrategy(StrategyBase):
    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        composed_col = self.compose_cols[0]
        df['z'] = self.calculate_zscore(df[composed_col])
        signals = np.where(df['z'] > self.entry_threshold, -1,
                          np.where(df['z'] < self.exit_threshold, 0, np.nan))
        return self._process_signals(df, signals)

class MR_MinMaxLongStrategy(StrategyBase):
    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        composed_col = self.compose_cols[0]
        df['minmax'] = self.calculate_minmax(df[composed_col])
        signals = np.where(df['minmax'] < self.entry_threshold, 1,
                          np.where(df['minmax'] > self.exit_threshold, 0, np.nan))
        return self._process_signals(df, signals)

class MR_MinMaxShortStrategy(StrategyBase):
    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        composed_col = self.compose_cols[0]
        df['minmax'] = self.calculate_minmax(df[composed_col])
        signals = np.where(df['minmax'] > self.entry_threshold, -1,
                          np.where(df['minmax'] < self.exit_threshold, 0, np.nan))
        return self._process_signals(df, signals)

class MR_MinMaxFlipStrategy(StrategyBase):
    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        composed_col = self.compose_cols[0]
        df['minmax'] = self.calculate_minmax(df[composed_col])
        
        signals = np.where(df['minmax'] < self.entry_threshold, 1,
                          np.where(df['minmax'] > self.exit_threshold, -1, np.nan))
        
        return self._process_signals(df, signals)

# Trend Following Strategies
class TF_ZScoreLongStrategy(StrategyBase):
    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        composed_col = self.compose_cols[0]
        df['z'] = self.calculate_zscore(df[composed_col])

        # Keep entry as crossover
        entry_signals = self.check_crossover(df['z'], self.entry_threshold)
        # Change exit to threshold comparison
        # exit_signals = df['z'] >= self.exit_threshold
        exit_signals = self.check_crossunder(df['z'], self.exit_threshold)

        signals = pd.Series(np.nan, index=df.index)
        signals[entry_signals] = 1
        signals[exit_signals] = 0

        return self._process_signals(df, signals)

class TF_ZScoreShortStrategy(StrategyBase):
    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        composed_col = self.compose_cols[0]
        df['z'] = self.calculate_zscore(df[composed_col])

        # Keep entry as crossunder
        entry_signals = self.check_crossunder(df['z'], self.entry_threshold)
        # Change exit to threshold comparison
        # exit_signals = df['z'] <= self.exit_threshold
        exit_signals = self.check_crossover(df['z'], self.exit_threshold)

        signals = pd.Series(np.nan, index=df.index)
        signals[entry_signals] = -1
        signals[exit_signals] = 0

        return self._process_signals(df, signals)

class TF_MinMaxLongStrategy(StrategyBase):
    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        composed_col = self.compose_cols[0]
        df['minmax'] = self.calculate_minmax(df[composed_col])

        # Keep entry as crossover
        entry_signals = self.check_crossover(df['minmax'], self.entry_threshold)
        # Change exit to threshold comparison
        # exit_signals = df['minmax'] >= self.exit_threshold
        exit_signals = self.check_crossunder(df['minmax'], self.exit_threshold)

        signals = pd.Series(np.nan, index=df.index)
        signals[entry_signals] = 1
        signals[exit_signals] = 0

        return self._process_signals(df, signals)

class TF_MinMaxShortStrategy(StrategyBase):
    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        composed_col = self.compose_cols[0]
        df['minmax'] = self.calculate_minmax(df[composed_col])

        # Keep entry as crossunder
        entry_signals = self.check_crossunder(df['minmax'], self.entry_threshold)
        # Change exit to threshold comparison
        # exit_signals = df['minmax'] <= self.exit_threshold
        exit_signals = self.check_crossover(df['minmax'], self.exit_threshold)

        signals = pd.Series(np.nan, index=df.index)
        signals[entry_signals] = -1
        signals[exit_signals] = 0

        return self._process_signals(df, signals)

# Strategy generator functions
def generate_mr_zscore_long(window: int, entry_threshold: float, exit_threshold: float, compose_cols: List[str], compose_operation: str):
    return MR_ZScoreLongStrategy(window, entry_threshold, exit_threshold, compose_cols, compose_operation)

def generate_mr_zscore_short(window: int, entry_threshold: float, exit_threshold: float, compose_cols: List[str], compose_operation: str):
    return MR_ZScoreShortStrategy(window, entry_threshold, exit_threshold, compose_cols, compose_operation)

def generate_mr_minmax_long(window: int, entry_threshold: float, exit_threshold: float, compose_cols: List[str], compose_operation: str):
    return MR_MinMaxLongStrategy(window, entry_threshold, exit_threshold, compose_cols, compose_operation)

def generate_mr_minmax_short(window: int, entry_threshold: float, exit_threshold: float, compose_cols: List[str], compose_operation: str):
    return MR_MinMaxShortStrategy(window, entry_threshold, exit_threshold, compose_cols, compose_operation)

def generate_mr_minmax_flip(window: int, entry_threshold: float, exit_threshold: float, compose_cols: List[str], compose_operation: str):
    return MR_MinMaxFlipStrategy(window, entry_threshold, exit_threshold, compose_cols, compose_operation)

def generate_tf_zscore_long(window: int, entry_threshold: float, exit_threshold: float, compose_cols: List[str], compose_operation: str):
    return TF_ZScoreLongStrategy(window, entry_threshold, exit_threshold, compose_cols, compose_operation)

def generate_tf_zscore_short(window: int, entry_threshold: float, exit_threshold: float, compose_cols: List[str], compose_operation: str):
    return TF_ZScoreShortStrategy(window, entry_threshold, exit_threshold, compose_cols, compose_operation)

def generate_tf_minmax_long(window: int, entry_threshold: float, exit_threshold: float, compose_cols: List[str], compose_operation: str):
    return TF_MinMaxLongStrategy(window, entry_threshold, exit_threshold, compose_cols, compose_operation)

def generate_tf_minmax_short(window: int, entry_threshold: float, exit_threshold: float, compose_cols: List[str], compose_operation: str):
    return TF_MinMaxShortStrategy(window, entry_threshold, exit_threshold, compose_cols, compose_operation)

STRATEGY_GENERATORS = {
    # Mean Reversion Strategies
    StrategyType.MR_ZSCORE_LONG: generate_mr_zscore_long,
    StrategyType.MR_ZSCORE_SHORT: generate_mr_zscore_short,
    StrategyType.MR_MINMAX_LONG: generate_mr_minmax_long,
    StrategyType.MR_MINMAX_SHORT: generate_mr_minmax_short,
    StrategyType.MR_MINMAX_FLIP: generate_mr_minmax_flip,
    # Trend Following Strategies
    StrategyType.TF_ZSCORE_LONG: generate_tf_zscore_long,
    StrategyType.TF_ZSCORE_SHORT: generate_tf_zscore_short,
    StrategyType.TF_MINMAX_LONG: generate_tf_minmax_long,
    StrategyType.TF_MINMAX_SHORT: generate_tf_minmax_short
}

def get_strategy_params(strategy_type: StrategyType) -> StrategyParams:
    params_map = {
        # Mean Reversion Parameters
        StrategyType.MR_ZSCORE_LONG: StrategyParams(
            entry_start=-2.7, entry_end=-0.5, entry_step=0.2,
            exit_start=0.5, exit_end=2.7, exit_step=0.2
        ),
        StrategyType.MR_ZSCORE_SHORT: StrategyParams(
            entry_start=0.5, entry_end=2.7, entry_step=0.2,
            exit_start=-2.7, exit_end=-0.5, exit_step=0.2
        ),
        StrategyType.MR_MINMAX_LONG: StrategyParams(
            entry_start=0.1, entry_end=0.4, entry_step=0.02,
            exit_start=0.6, exit_end=0.9, exit_step=0.02
        ),
        StrategyType.MR_MINMAX_SHORT: StrategyParams(
            entry_start=0.6, entry_end=0.9, entry_step=0.02,
            exit_start=0.1, exit_end=0.4, exit_step=0.02
        ),
        StrategyType.MR_MINMAX_FLIP: StrategyParams(
            entry_start=0.1, entry_end=0.4, entry_step=0.02,
            exit_start=0.6, exit_end=0.9, exit_step=0.02
        ),
        # Trend Following Parameters
        StrategyType.TF_ZSCORE_LONG: StrategyParams(
            entry_start=-2.7, entry_end=0.5, entry_step=0.2,
            exit_start=1.0, exit_end=2.7, exit_step=0.2
        ),
        StrategyType.TF_ZSCORE_SHORT: StrategyParams(
            entry_start=0.5, entry_end=2.7, entry_step=0.2,
            exit_start=-2.7, exit_end=0.0, exit_step=0.2
        ),
        StrategyType.TF_MINMAX_LONG: StrategyParams(
            entry_start=0.1, entry_end=0.4, entry_step=0.02,
            exit_start=0.5, exit_end=0.9, exit_step=0.02
        ),
        StrategyType.TF_MINMAX_SHORT: StrategyParams(
            entry_start=0.5, entry_end=0.9, entry_step=0.02,
            exit_start=0.1, exit_end=0.4, exit_step=0.02
        )
    }
    return params_map[strategy_type]

def get_param_ranges(strategy_type: StrategyType) -> tuple:
    params = get_strategy_params(strategy_type)
    window_range = range(params.window_start, params.window_end, params.window_step)
    entry_threshold_range = np.arange(params.entry_start, params.entry_end, params.entry_step)
    exit_threshold_range = np.arange(params.exit_start, params.exit_end, params.exit_step)
    return window_range, entry_threshold_range, exit_threshold_range

def print_results_with_title(result_df: pd.DataFrame, strategy_name: str):
    separator = "=" * 100
    print(f"\n{separator}")
    print(f"{strategy_name} - Top 10 Results".center(100))
    print(separator)
    print(result_df.head(10))

    print(f"\n{separator}")
    print(f"{strategy_name} - Bottom 10 Results".center(100))
    print(separator)
    print(result_df.tail(10))