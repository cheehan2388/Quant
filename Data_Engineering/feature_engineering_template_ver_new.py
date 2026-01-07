import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
import random
from scipy.stats import pearsonr, spearmanr
warnings.filterwarnings('ignore')

###############################################################################
# Alpha101-style Operators and Alpha Templates
###############################################################################

class AlphaOps:
    """Alpha101-style primitive operators for pandas Series/DataFrame.

    Notes:
    - For cross-sectional ops (rank, indneutralize, scale): if given a Series, we
      approximate via time-series variants.
    - All functions are vectorized and return pandas Series aligned with input.
    """

    @staticmethod
    def to_series(x):
        if isinstance(x, pd.Series):
            return x
        return pd.Series(x)

    @staticmethod
    def abs(x):
        x = AlphaOps.to_series(x)
        return x.abs()

    @staticmethod
    def log(x):
        x = AlphaOps.to_series(x)
        return np.log(x.replace({0: np.nan})).replace({np.inf: np.nan, -np.inf: np.nan}).fillna(0)

    @staticmethod
    def sign(x):
        x = AlphaOps.to_series(x)
        return np.sign(x)

    @staticmethod
    def rank(x):
        """Cross-sectional rank; for Series fallback to time-series rank over entire series."""
        if isinstance(x, pd.DataFrame):
            # Rank each timestamp across columns
            return x.rank(axis=1, pct=True)
        s = AlphaOps.to_series(x)
        return s.rank(pct=True)

    @staticmethod
    def rank_ts(x, d=None):
        """Time-series rank. If d is None, rank over entire history; else rank of last
        value within rolling window d, normalized to [0,1]."""
        s = AlphaOps.to_series(x)
        if d is None:
            return s.rank(pct=True)
        def _rank_last(window):
            arr = pd.Series(window)
            return arr.rank(pct=True).iloc[-1]
        return s.rolling(int(np.floor(d)), min_periods=1).apply(_rank_last, raw=False)

    @staticmethod
    def delay(x, d):
        s = AlphaOps.to_series(x)
        return s.shift(int(np.floor(d)))

    @staticmethod
    def delta(x, d):
        s = AlphaOps.to_series(x)
        d = int(np.floor(d))
        return s - s.shift(d)

    @staticmethod
    def correlation(x, y, d):
        sx = AlphaOps.to_series(x)
        sy = AlphaOps.to_series(y)
        return sx.rolling(int(np.floor(d)), min_periods=2).corr(sy).fillna(0)

    @staticmethod
    def covariance(x, y, d):
        sx = AlphaOps.to_series(x)
        sy = AlphaOps.to_series(y)
        return sx.rolling(int(np.floor(d)), min_periods=2).cov(sy).fillna(0)

    @staticmethod
    def scale(x, a=1.0):
        """Cross-sectional scale to sum(abs(x)) = a. For Series, approximate over time."""
        if isinstance(x, pd.DataFrame):
            denom = x.abs().sum(axis=1).replace(0, np.nan)
            return x.div(denom, axis=0).fillna(0) * a
        s = AlphaOps.to_series(x)
        denom = s.abs().sum()
        if denom == 0:
            return s * 0
        return s * (a / denom)

    @staticmethod
    def signedpower(x, a):
        s = AlphaOps.to_series(x)
        return np.sign(s) * (np.abs(s) ** a)

    @staticmethod
    def decay_linear(x, d):
        s = AlphaOps.to_series(x)
        d = int(np.floor(d))
        weights = np.arange(1, d + 1, dtype=float)
        weights /= weights.sum()
        def _wma(window):
            w = weights[-len(window):]
            return np.dot(window, w)
        return s.rolling(d, min_periods=1).apply(_wma, raw=True)

    @staticmethod
    def ts_min(x, d):
        s = AlphaOps.to_series(x)
        return s.rolling(int(np.floor(d)), min_periods=1).min()

    @staticmethod
    def ts_max(x, d):
        s = AlphaOps.to_series(x)
        return s.rolling(int(np.floor(d)), min_periods=1).max()

    @staticmethod
    def ts_argmax(x, d):
        s = AlphaOps.to_series(x)
        def _argmax(window):
            return int(np.argmax(window))
        return s.rolling(int(np.floor(d)), min_periods=1).apply(_argmax, raw=True)

    @staticmethod
    def ts_argmin(x, d):
        s = AlphaOps.to_series(x)
        def _argmin(window):
            return int(np.argmin(window))
        return s.rolling(int(np.floor(d)), min_periods=1).apply(_argmin, raw=True)

    @staticmethod
    def ts_rank(x, d):
        return AlphaOps.rank_ts(x, d)

    @staticmethod
    def ts_sum(x, d):
        s = AlphaOps.to_series(x)
        return s.rolling(int(np.floor(d)), min_periods=1).sum()

    @staticmethod
    def ts_product(x, d):
        s = AlphaOps.to_series(x)
        return s.rolling(int(np.floor(d)), min_periods=1).apply(np.prod, raw=True)

    @staticmethod
    def stddev(x, d):
        s = AlphaOps.to_series(x)
        return s.rolling(int(np.floor(d)), min_periods=2).std().fillna(0)

    # Aliases from the paper
    min = ts_min
    max = ts_max
    sum = ts_sum
    product = ts_product

    # Logical and ternary helpers
    @staticmethod
    def gt(x, y):
        return (AlphaOps.to_series(x) > AlphaOps.to_series(y)).astype(float)

    @staticmethod
    def lt(x, y):
        return (AlphaOps.to_series(x) < AlphaOps.to_series(y)).astype(float)

    @staticmethod
    def eq(x, y):
        return (AlphaOps.to_series(x) == AlphaOps.to_series(y)).astype(float)

    @staticmethod
    def logical_or(a, b):
        return ((AlphaOps.to_series(a) > 0) | (AlphaOps.to_series(b) > 0)).astype(float)

    @staticmethod
    def ternary(condition, y, z):
        c = AlphaOps.to_series(condition)
        return pd.Series(np.where(c > 0, AlphaOps.to_series(y), AlphaOps.to_series(z)), index=c.index)

    @staticmethod
    def indneutralize(x, g=None):
        """Cross-sectional demeaning by group g. If Series or no groups, return x."""
        if isinstance(x, pd.Series) or g is None:
            return AlphaOps.to_series(x)
        # x: DataFrame [time, instruments], g: dict of column->group or Series indexed by columns
        groups = pd.Series({col: g.get(col, None) for col in x.columns}) if isinstance(g, dict) else pd.Series(g)
        out = []
        for t, row in x.iterrows():
            df = pd.DataFrame({'value': row, 'group': groups})
            demeaned = df.groupby('group')['value'].transform(lambda v: v - v.mean())
            out.append(demeaned.values)
        return pd.DataFrame(out, index=x.index, columns=x.columns)


class MarketDataContext:
    """Helper to map typical market field names and derive commonly-used series."""
    def __init__(self, df: pd.DataFrame):
        self.df = df
        cols_map = {c.lower(): c for c in df.columns}
        def pick(*names):
            for n in names:
                if n in cols_map:
                    return df[cols_map[n]]
            return None
        self.open = pick('open')
        self.high = pick('high')
        self.low = pick('low')
        self.close = pick('close', 'price', 'last')
        self.volume = pick('volume', 'vol')
        self.vwap = pick('vwap', 'volume_weighted_average_price')
        # Derived series
        self.returns = self.close.pct_change() if self.close is not None else None
        self.adv20 = self.volume.rolling(20, min_periods=1).mean() if self.volume is not None else None


def compute_alpha_samples(ctx: MarketDataContext, use_time_series_rank: bool = True):
    """Compute a subset of Alpha101 sample formulas (8-12, 14-17).

    Returns: dict[name -> Series]
    """
    A = AlphaOps  # alias
    out = {}

    def r(x, d=None):
        return A.rank_ts(x, d) if use_time_series_rank else A.rank(x)

    try:
        s = A.sum(ctx.open, 5) * A.sum(ctx.returns, 5)
        out['alpha_8'] = -r(s - A.delay(s, 10))
    except Exception:
        pass

    try:
        d1 = A.delta(ctx.close, 1)
        cond1 = A.gt(0, A.ts_min(d1, 5))
        cond2 = A.lt(A.ts_max(d1, 5), 0)
        # nested ternary
        out['alpha_9'] = A.ternary(cond1, d1, A.ternary(cond2, d1, -d1))
    except Exception:
        pass

    try:
        d1 = A.delta(ctx.close, 1)
        cond1 = A.gt(0, A.ts_min(d1, 4))
        cond2 = A.lt(A.ts_max(d1, 4), 0)
        out['alpha_10'] = r(A.ternary(cond1, d1, A.ternary(cond2, d1, -d1)))
    except Exception:
        pass

    try:
        x = ctx.vwap - ctx.close
        out['alpha_11'] = (r(A.ts_max(x, 3)) + r(A.ts_min(x, 3))) * r(A.delta(ctx.volume, 3))
    except Exception:
        pass

    try:
        out['alpha_12'] = A.sign(A.delta(ctx.volume, 1)) * (-A.delta(ctx.close, 1))
    except Exception:
        pass

    try:
        out['alpha_14'] = (-r(A.delta(ctx.returns, 3))) * A.correlation(ctx.open, ctx.volume, 10)
    except Exception:
        pass

    try:
        out['alpha_15'] = -A.sum(r(A.correlation(r(ctx.high), r(ctx.volume), 3)), 3)
    except Exception:
        pass

    try:
        out['alpha_16'] = -r(A.covariance(r(ctx.high), r(ctx.volume), 5))
    except Exception:
        pass

    try:
        tsr_close_10 = A.ts_rank(ctx.close, 10)
        ddc = A.delta(A.delta(ctx.close, 1), 1)
        vol_adv = ctx.volume / ctx.adv20
        out['alpha_17'] = ((-r(tsr_close_10)) * r(ddc)) * r(A.ts_rank(vol_adv, 5))
    except Exception:
        pass

    return out

def load_data(file_path='../Data/All_data/merged/BTC_merged_by_datetime.csv'):
    """Load and preprocess the merged data"""
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df = df.ffill().bfill()  # Forward fill then backward fill missing values
    return df

def calculate_future_returns(df, close_dir='../Data/close', shifts=[-2, -8, -24]):
    """Calculate future returns for different shift windows"""
    future_returns = {}
    
    # Use the first close file for future returns calculation
    close_path = os.path.join(close_dir, 'Binance_1h_BTCUSDT_2020-01-01_2025-07-07.csv')
    close_df = pd.read_csv(close_path)
    close_df['datetime'] = pd.to_datetime(close_df['datetime'])
    
    # Merge with main dataframe to get close prices
    merged = pd.merge(df, close_df[['datetime', 'Close']], on='datetime', how='left')
    
    # Calculate future returns for different shifts
    for shift in shifts:
        future_return_col = f'future_return_{abs(shift)}'
        merged[future_return_col] = merged['Close'].pct_change(periods=abs(shift)).shift(shift)
        future_returns[future_return_col] = merged[future_return_col]
    
    # Add future returns to original dataframe
    for shift in shifts:
        future_return_col = f'future_return_{abs(shift)}'
        df[future_return_col] = future_returns[future_return_col]
    
    return df, future_returns

def apply_rolling_mean(series, window):
    """Apply rolling mean transformation"""
    return series.rolling(window=window, min_periods=1).mean()

def apply_rolling_minmax_scaling(series, window):
    """Apply rolling min-max scaling"""
    def rolling_minmax(x):
        if len(x) < 2:
            return x.iloc[-1] if hasattr(x, 'iloc') else x[-1]
        min_val = x.min()
        max_val = x.max()
        if max_val == min_val:
            return 0.0
        # Return only the scaled value of the last point in the window
        return (x.iloc[-1] - min_val) / (max_val - min_val) if hasattr(x, 'iloc') else (x[-1] - min_val) / (max_val - min_val)
    
    return series.rolling(window=window, min_periods=1).apply(rolling_minmax, raw=False)

def apply_rolling_zscore(series, window):
    """Apply rolling z-score transformation"""
    def rolling_zscore(x):
        if len(x) < 2:
            return 0.0
        mean_val = x.mean()
        std_val = x.std()
        if std_val == 0:
            return 0.0
        # Return only the z-score of the last point in the window
        return (x.iloc[-1] - mean_val) / std_val if hasattr(x, 'iloc') else (x[-1] - mean_val) / std_val
    
    return series.rolling(window=window, min_periods=1).apply(rolling_zscore, raw=False)

def calculate_ic(feature_series, target_series):
    """Calculate Information Coefficient (Spearman correlation)"""
    # Remove NaN values
    valid_mask = ~(pd.isna(feature_series) | pd.isna(target_series))
    if valid_mask.sum() < 10:  # Need at least 10 valid observations
        return 0.0
    
    feature_clean = feature_series[valid_mask]
    target_clean = target_series[valid_mask]
    
    # Calculate correlation
    correlation = feature_clean.corr(target_clean, method="spearman")
    return correlation if not pd.isna(correlation) else 0.0

def generate_all_features(df):
    """Generate all transformed features according to the specification"""
    
    # Get all numeric columns (exclude datetime and future returns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['datetime'] + [col for col in df.columns if 'future_return' in col]
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"Processing {len(feature_cols)} base features...")
    
    # Define transformation tools
    transformations = {
        'rolling_mean_12': lambda x: apply_rolling_mean(x, 12),
        'rolling_mean_24': lambda x: apply_rolling_mean(x, 24),
        'rolling_mean_52': lambda x: apply_rolling_mean(x, 52),
        'rolling_minmax_12': lambda x: apply_rolling_minmax_scaling(x, 12),
        'rolling_minmax_24': lambda x: apply_rolling_minmax_scaling(x, 24),
        'rolling_minmax_52': lambda x: apply_rolling_minmax_scaling(x, 52),
        'rolling_zscore_12': lambda x: apply_rolling_zscore(x, 12),
        'rolling_zscore_24': lambda x: apply_rolling_zscore(x, 24),
        'rolling_zscore_52': lambda x: apply_rolling_zscore(x, 52)
    }
    
    # Store all transformed features
    transformed_features = {}
    feature_metadata = {}
    
    # Apply all transformations to all columns
    for col in feature_cols:
        print(f"Processing column: {col}")
        
        for transform_name, transform_func in transformations.items():
            feature_name = f"{col}_{transform_name}"
            try:
                transformed_series = transform_func(np.log1p(df[col]))
                transformed_features[feature_name] = transformed_series
                
                # Store metadata
                feature_metadata[feature_name] = {
                    'base_column': col,
                    'transformation': transform_name,
                    'ic_values': {},
                    'max_ic': 0.0,
                    'best_shift': None
                }
                
            except Exception as e:
                print(f"Error processing {feature_name}: {str(e)}")
                continue
    
    return transformed_features, feature_metadata

def generate_alpha101_features(df, use_time_series_rank: bool = True):
    """Generate Alpha101 sample features (8-12,14-17) and metadata."""
    ctx = MarketDataContext(df)
    alpha_features = compute_alpha_samples(ctx, use_time_series_rank=use_time_series_rank)
    alpha_metadata = {}
    for name in alpha_features.keys():
        alpha_metadata[name] = {
            'base_column': 'alpha101',
            'transformation': name,
            'ic_values': {},
            'max_ic': 0.0,
            'best_shift': None
        }
    return alpha_features, alpha_metadata

def calculate_ics_for_all_features(df, transformed_features, feature_metadata, shifts=[-2, -8, -24]):
    """Calculate IC values for all transformed features"""
    
    print(f"Calculating IC values for {len(transformed_features)} transformed features...")
    
    # Get future return columns
    future_return_cols = [f'future_return_{abs(shift)}' for shift in shifts]
    
    for feature_name, feature_series in transformed_features.items():
        ic_values = {}
        
        # Calculate IC for each shift window
        for shift in shifts:
            future_return_col = f'future_return_{abs(shift)}'
            if future_return_col in df.columns:
                ic_value = calculate_ic(feature_series, df[future_return_col])
                ic_values[f'shift_{shift}'] = ic_value
            else:
                ic_values[f'shift_{shift}'] = 0.0
        
        # Find maximum absolute IC and corresponding shift
        abs_ic_values = {k: abs(v) for k, v in ic_values.items()}
        best_shift_key = max(abs_ic_values.keys(), key=abs_ic_values.get)
        max_ic = ic_values[best_shift_key]
        
        # Update metadata
        feature_metadata[feature_name]['ic_values'] = ic_values
        feature_metadata[feature_name]['max_ic'] = max_ic
        feature_metadata[feature_name]['best_shift'] = best_shift_key
        
        # Add transformed feature to dataframe
        df[feature_name] = feature_series
    
    return df, feature_metadata

def generate_report(feature_metadata, output_path='feature_engineering_report_N_S_1.json'):
    """Generate JSON report with all feature information"""
    
    # Sort features by maximum absolute IC (descending)
    sorted_features = sorted(
        feature_metadata.items(),
        key=lambda x: abs(x[1]['max_ic']),
        reverse=True
    )
    
    report = {
        'summary': {
            'total_features': len(feature_metadata),
            'transformation_types': [
                'rolling_mean_12', 'rolling_mean_24', 'rolling_mean_52',
                'rolling_minmax_12', 'rolling_minmax_24', 'rolling_minmax_52',
                'rolling_zscore_12', 'rolling_zscore_24', 'rolling_zscore_52',
                'alpha101_samples'
            ],
            'shift_windows': [-2, -8, -24],
            'top_10_features': [name for name, _ in sorted_features[:10]]
        },
        'features': {}
    }
    
    # Add detailed information for each feature
    for feature_name, metadata in sorted_features:
        report['features'][feature_name] = {
            'base_column': metadata['base_column'],
            'transformation': metadata['transformation'],
            'ic_values': {
                'shift_-2': metadata['ic_values'].get('shift_-2', 0.0),
                'shift_-8': metadata['ic_values'].get('shift_-8', 0.0),
                'shift_-24': metadata['ic_values'].get('shift_-24', 0.0)
            },
            'max_ic': metadata['max_ic'],
            'best_shift': metadata['best_shift'],
            'abs_max_ic': abs(metadata['max_ic'])
        }
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to {output_path}")
    return report

def elite_check_function(df, feature_metadata, ic_threshold=0.05):
    """Filter and save elite features with IC > threshold"""
    print(f"\n=== ELITE FEATURE SELECTION (IC > {ic_threshold}) ===")
    
    # Filter elite features
    elite_features = {}
    elite_metadata = {}
    
    for feature_name, metadata in feature_metadata.items():
        if abs(metadata['max_ic']) > ic_threshold:
            elite_features[feature_name] = df[feature_name]
            elite_metadata[feature_name] = metadata
    
    print(f"Found {len(elite_features)} elite features out of {len(feature_metadata)} total features")
    
    if len(elite_features) == 0:
        print("No elite features found with the specified threshold.")
        return elite_features, elite_metadata
    
    # Create elite dataframe with datetime
    elite_df = pd.DataFrame({'datetime': df['datetime']})
    for feature_name, feature_series in elite_features.items():
        elite_df[feature_name] = feature_series
    
    # Save elite features to CSV
    elite_csv_path = 'elite_features.csv'
    elite_df.to_csv(elite_csv_path, index=False)
    print(f"Elite features saved to {elite_csv_path}")
    
    # Create elite JSON report
    elite_report = {
        'summary': {
            'total_elite_features': len(elite_features),
            'ic_threshold': ic_threshold,
            'average_ic': np.mean([abs(metadata['max_ic']) for metadata in elite_metadata.values()]),
            'max_ic': max([abs(metadata['max_ic']) for metadata in elite_metadata.values()]),
            'elite_feature_names': list(elite_features.keys())
        },
        'features': {}
    }
    
    # Sort elite features by absolute IC
    sorted_elite = sorted(
        elite_metadata.items(),
        key=lambda x: abs(x[1]['max_ic']),
        reverse=True
    )
    
    for feature_name, metadata in sorted_elite:
        elite_report['features'][feature_name] = {
            'base_column': metadata['base_column'],
            'transformation': metadata['transformation'],
            'ic_values': metadata['ic_values'],
            'max_ic': metadata['max_ic'],
            'best_shift': metadata['best_shift'],
            'abs_max_ic': abs(metadata['max_ic'])
        }
    
    # Save elite JSON report
    elite_json_path = 'elite_features.json'
    with open(elite_json_path, 'w') as f:
        json.dump(elite_report, f, indent=2)
    print(f"Elite features report saved to {elite_json_path}")
    
    return elite_features, elite_metadata

def split_feature_groups(df, feature_metadata, group_ic_threshold=0.01):
    """Split features into short-midterm and mid-longterm groups"""
    print(f"\n=== FEATURE GROUP SPLITTING (IC > {group_ic_threshold}) ===")
    
    short_midterm_features = {}
    mid_longterm_features = {}
    short_midterm_metadata = {}
    mid_longterm_metadata = {}
    
    for feature_name, metadata in feature_metadata.items():
        ic_values = metadata['ic_values']
        
        # Check for short-midterm: shift(-2) OR shift(-8)
        shift_2_ic = abs(ic_values.get('shift_-2', 0.0))
        shift_8_ic = abs(ic_values.get('shift_-8', 0.0))
        shift_24_ic = abs(ic_values.get('shift_-24', 0.0))
        
        # Short-midterm group: IC > threshold for shift(-2) OR shift(-8) or both
        if shift_2_ic > group_ic_threshold or shift_8_ic > group_ic_threshold:
            short_midterm_features[feature_name] = df[feature_name]
            short_midterm_metadata[feature_name] = metadata
        
        # Mid-longterm group: IC > threshold for shift(-8) OR shift(-24) or both
        if shift_8_ic > group_ic_threshold or shift_24_ic > group_ic_threshold:
            mid_longterm_features[feature_name] = df[feature_name]
            mid_longterm_metadata[feature_name] = metadata
    
    print(f"Short-Midterm Group: {len(short_midterm_features)} features")
    print(f"Mid-Longterm Group: {len(mid_longterm_features)} features")
    
    # Save short-midterm group
    if len(short_midterm_features) > 0:
        short_midterm_df = pd.DataFrame({'datetime': df['datetime']})
        for feature_name, feature_series in short_midterm_features.items():
            short_midterm_df[feature_name] = feature_series
        
        short_midterm_csv = 'short_midterm_features.csv'
        short_midterm_df.to_csv(short_midterm_csv, index=False)
        print(f"Short-midterm features saved to {short_midterm_csv}")
        
        # Short-midterm JSON report
        short_midterm_report = create_group_report(short_midterm_metadata, "Short-Midterm", group_ic_threshold)
        with open('short_midterm_features.json', 'w') as f:
            json.dump(short_midterm_report, f, indent=2)
        print("Short-midterm features report saved to short_midterm_features.json")
    
    # Save mid-longterm group
    if len(mid_longterm_features) > 0:
        mid_longterm_df = pd.DataFrame({'datetime': df['datetime']})
        for feature_name, feature_series in mid_longterm_features.items():
            mid_longterm_df[feature_name] = feature_series
        
        mid_longterm_csv = 'mid_longterm_features.csv'
        mid_longterm_df.to_csv(mid_longterm_csv, index=False)
        print(f"Mid-longterm features saved to {mid_longterm_csv}")
        
        # Mid-longterm JSON report
        mid_longterm_report = create_group_report(mid_longterm_metadata, "Mid-Longterm", group_ic_threshold)
        with open('mid_longterm_features.json', 'w') as f:
            json.dump(mid_longterm_report, f, indent=2)
        print("Mid-longterm features report saved to mid_longterm_features.json")
    
    return short_midterm_features, mid_longterm_features, short_midterm_metadata, mid_longterm_metadata

def create_group_report(metadata, group_name, threshold):
    """Create a standardized report for feature groups"""
    sorted_features = sorted(
        metadata.items(),
        key=lambda x: abs(x[1]['max_ic']),
        reverse=True
    )
    
    report = {
        'summary': {
            'group_name': group_name,
            'total_features': len(metadata),
            'ic_threshold': threshold,
            'average_ic': np.mean([abs(meta['max_ic']) for meta in metadata.values()]) if metadata else 0,
            'max_ic': max([abs(meta['max_ic']) for meta in metadata.values()]) if metadata else 0,
            'top_10_features': [name for name, _ in sorted_features[:10]]
        },
        'features': {}
    }
    
    for feature_name, meta in sorted_features:
        report['features'][feature_name] = {
            'base_column': meta['base_column'],
            'transformation': meta['transformation'],
            'ic_values': meta['ic_values'],
            'max_ic': meta['max_ic'],
            'best_shift': meta['best_shift'],
            'abs_max_ic': abs(meta['max_ic'])
        }
    
    return report

# =============================================================================
# ENHANCED GENETIC PROGRAMMING WITH GENERATION-BY-GENERATION TRACKING
# =============================================================================

class GeneticOperators:
    """Genetic programming operators for feature combination"""
    
    @staticmethod
    def safe_divide(a, b):
        """Safe division to avoid division by zero"""
        return np.where(np.abs(b) < 1e-10, np.ones_like(a), a / b)
    
    @staticmethod
    def correlation_op(a, b):
        """Calculate rolling correlation between two series"""
        # Use pandas rolling correlation
        if len(a) != len(b):
            min_len = min(len(a), len(b))
            a = a[:min_len]
            b = b[:min_len]
        
        # Create pandas series for correlation calculation
        series_a = pd.Series(a)
        series_b = pd.Series(b)
        
        # Calculate rolling correlation with window=24
        corr = series_a.rolling(window=24, min_periods=1).corr(series_b)
        return corr.fillna(0).values
    
    @staticmethod
    def apply_operator(feature1, feature2, operator):
        """Apply genetic operator to two features"""
        try:
            if operator == '+':
                return feature1 + feature2
            elif operator == '-':
                return feature1 - feature2
            elif operator == '*':
                return feature1 * feature2
            elif operator == '/':
                return GeneticOperators.safe_divide(feature1, feature2)
            elif operator == 'corr':
                return GeneticOperators.correlation_op(feature1, feature2)
            else:
                return feature1 + feature2  # Default fallback
        except Exception as e:
            print(f"Error in operator {operator}: {e}")
            return feature1 + feature2  # Safe fallback

class EnhancedGeneticFeatureEvolution:
    """Enhanced Genetic Programming with Generation Tracking"""
    
    def __init__(self, short_midterm_features, mid_longterm_features, df, 
                 operators=['+', '-', '*', '/', 'corr'], generations=12, 
                 population_size=50, mutation_rate=0.3, ic_threshold=0.05):
        self.short_midterm_features = short_midterm_features
        self.mid_longterm_features = mid_longterm_features
        self.df = df
        self.operators = operators
        self.generations = generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.ic_threshold = ic_threshold
        
        # Available transformations for mutation
        self.transformations = {
            'rolling_mean_12': lambda x: apply_rolling_mean(x, 12),
            'rolling_mean_24': lambda x: apply_rolling_mean(x, 24),
            'rolling_mean_52': lambda x: apply_rolling_mean(x, 52),
            'rolling_minmax_12': lambda x: apply_rolling_minmax_scaling(x, 12),
            'rolling_minmax_24': lambda x: apply_rolling_minmax_scaling(x, 24),
            'rolling_minmax_52': lambda x: apply_rolling_minmax_scaling(x, 52),
            'rolling_zscore_12': lambda x: apply_rolling_zscore(x, 12),
            'rolling_zscore_24': lambda x: apply_rolling_zscore(x, 24),
            'rolling_zscore_52': lambda x: apply_rolling_zscore(x, 52)
        }
        
        self.population = []
        self.fitness_scores = []
        self.generation_data = []  # Store each generation's elite features
        
    def create_individual(self):
        """Create a single genetic programming individual"""
        # Choose 2 features from pools (can be from same or different pools)
        all_features = {**self.short_midterm_features, **self.mid_longterm_features}
        feature_names = list(all_features.keys())
        
        if len(feature_names) < 2:
            return None
            
        feature1_name = random.choice(feature_names)
        feature2_name = random.choice(feature_names)
        
        # Choose 1 operator
        operator = random.choice(self.operators)
        
        individual = {
            'feature1': feature1_name,
            'feature2': feature2_name,
            'operator': operator,
            'fitness': 0.0,
            'feature_data': None,
            'ic_values': {}  # Store all IC values for all shifts
        }
        
        return individual
    
    def evaluate_individual_detailed(self, individual):
        """Evaluate individual with detailed IC tracking for all shifts"""
        try:
            all_features = {**self.short_midterm_features, **self.mid_longterm_features}
            
            # Get feature data
            feature1_data = all_features[individual['feature1']].values
            feature2_data = all_features[individual['feature2']].values
            
            # Apply single operator: feature1 operator feature2
            final_feature = GeneticOperators.apply_operator(
                feature1_data, feature2_data, individual['operator']
            )
            
            # Store the evolved feature data
            individual['feature_data'] = pd.Series(final_feature, index=self.df.index)
            
            # Calculate IC against all future returns and store ALL values
            ic_values = {}
            for shift in [-2, -8, -24]:
                future_return_col = f'future_return_{abs(shift)}'
                if future_return_col in self.df.columns:
                    ic = calculate_ic(individual['feature_data'], self.df[future_return_col])
                    ic_values[f'shift_{shift}'] = ic
            
            individual['ic_values'] = ic_values
            
            # Use maximum absolute IC as fitness
            abs_ics = [abs(v) for v in ic_values.values()]
            fitness = max(abs_ics) if abs_ics else 0.0
            individual['fitness'] = fitness
            
            return fitness
            
        except Exception as e:
            print(f"Error evaluating individual: {e}")
            individual['fitness'] = 0.0
            individual['ic_values'] = {'shift_-2': 0.0, 'shift_-8': 0.0, 'shift_-24': 0.0}
            return 0.0
    
    def save_generation_elite(self, generation, elite_individuals):
        """Save elite features from current generation to CSV and JSON"""
        if not elite_individuals:
            return
        
        # Create CSV with elite features
        generation_df = pd.DataFrame({'datetime': self.df['datetime']})
        generation_metadata = {}
        
        for i, individual in enumerate(elite_individuals):
            feature_name = f"gen_{generation}_elite_{i+1}"
            
            if individual['feature_data'] is not None:
                generation_df[feature_name] = individual['feature_data']
                
                # Create detailed metadata
                generation_metadata[feature_name] = {
                    'generation': generation,
                    'rank': i + 1,
                    'fitness': individual['fitness'],
                    'feature1': individual['feature1'],
                    'feature2': individual['feature2'],
                    'operator': individual['operator'],
                    'expression': f"{individual['feature1']} {individual['operator']} {individual['feature2']}",
                    'ic_values': {
                        'shift_-2': individual['ic_values'].get('shift_-2', 0.0),
                        'shift_-8': individual['ic_values'].get('shift_-8', 0.0),
                        'shift_-24': individual['ic_values'].get('shift_-24', 0.0)
                    },
                    'best_shift': f"shift_{[k for k, v in individual['ic_values'].items() if abs(v) == individual['fitness']][0].split('_')[1]}" if individual['ic_values'] else 'none'
                }
        
        # Save generation CSV
        csv_filename = f"generation_{generation}_elite_features.csv"
        generation_df.to_csv(csv_filename, index=False)
        print(f"  📁 Generation {generation} elite features saved to {csv_filename}")
        
        # Save generation JSON
        generation_report = {
            'generation_info': {
                'generation_number': generation,
                'elite_count': len(elite_individuals),
                'ic_threshold': self.ic_threshold,
                'king_fitness': elite_individuals[0]['fitness'] if elite_individuals else 0,
                'avg_elite_fitness': np.mean([ind['fitness'] for ind in elite_individuals])
            },
            'king_of_generation': {
                'feature_name': f"gen_{generation}_elite_1",
                'fitness': elite_individuals[0]['fitness'],
                'expression': generation_metadata[f"gen_{generation}_elite_1"]['expression'],
                'ic_values': generation_metadata[f"gen_{generation}_elite_1"]['ic_values']
            } if elite_individuals else {},
            'all_elite_features': generation_metadata
        }
        
        json_filename = f"generation_{generation}_elite_report.json"
        with open(json_filename, 'w') as f:
            json.dump(generation_report, f, indent=2)
        print(f"  📊 Generation {generation} report saved to {json_filename}")
        
        # Store for overall tracking
        self.generation_data.append({
            'generation': generation,
            'elite_individuals': elite_individuals,
            'metadata': generation_metadata,
            'csv_file': csv_filename,
            'json_file': json_filename
        })
    
    def evolve_with_tracking(self):
        """Run evolution with detailed generation tracking"""
        print(f"\n🧬 ENHANCED GENETIC PROGRAMMING with Generation Tracking")
        print(f"Saving all elite features (IC > {self.ic_threshold}) from each generation")
        print("="*80)
        
        # Initialize population
        print(f"Initializing population of {self.population_size} individuals...")
        self.population = []
        
        for _ in range(self.population_size):
            individual = self.create_individual()
            if individual is not None:
                self.population.append(individual)
        
        print(f"Created {len(self.population)} individuals")
        
        if len(self.population) == 0:
            print("❌ Failed to create initial population")
            return []
        
        for generation in range(self.generations):
            print(f"\n🔬 Generation {generation + 1}/{self.generations}")
            
            # Evaluate population with detailed tracking
            valid_individuals = []
            for individual in self.population:
                fitness = self.evaluate_individual_detailed(individual)
                if fitness > self.ic_threshold:
                    valid_individuals.append(individual)
            
            # Sort by fitness (descending)
            valid_individuals.sort(key=lambda x: x['fitness'], reverse=True)
            
            print(f"📊 Elite individuals (IC > {self.ic_threshold}): {len(valid_individuals)}")
            
            if valid_individuals:
                king_fitness = valid_individuals[0]['fitness']
                avg_fitness = np.mean([ind['fitness'] for ind in valid_individuals])
                print(f"👑 King fitness: {king_fitness:.6f}")
                print(f"📈 Average elite fitness: {avg_fitness:.6f}")
                
                # Save this generation's elite features
                self.save_generation_elite(generation + 1, valid_individuals)
            else:
                print("⚠️  No elite individuals in this generation")
                # Still save empty generation for completeness
                self.save_generation_elite(generation + 1, [])
            
            # Continue evolution if not last generation
            if generation < self.generations - 1:
                # Selection and reproduction
                if len(valid_individuals) >= 2:
                    selected = valid_individuals  # Use all valid individuals
                    next_generation = []
                    
                    # Perform 5 crossover operations as requested
                    for _ in range(5):
                        parent1 = random.choice(selected)
                        parent2 = random.choice(selected)
                        
                        child1 = self.crossover(parent1, parent2)
                        child2 = self.crossover(parent2, parent1)
                        
                        # Apply mutation
                        self.mutate(child1)
                        self.mutate(child2)
                        
                        next_generation.extend([child1, child2])
                    
                    # Add elite individuals (best 20)
                    elite_size = min(20, len(selected))
                    elite = selected[:elite_size]
                    next_generation.extend([ind.copy() for ind in elite])
                    
                    # Fill remaining slots with new random individuals
                    while len(next_generation) < self.population_size:
                        new_individual = self.create_individual()
                        if new_individual is not None:
                            next_generation.append(new_individual)
                    
                    self.population = next_generation[:self.population_size]
                else:
                    # Reinitialize if no valid individuals
                    print("🔄 Reinitializing population...")
                    self.population = []
                    for _ in range(self.population_size):
                        individual = self.create_individual()
                        if individual is not None:
                            self.population.append(individual)
        
        print(f"\n✅ Evolution with tracking completed!")
        return self.generation_data
    
    def crossover(self, parent1, parent2):
        """Create offspring by combining parents"""
        child = parent1.copy()
        
        # Randomly inherit features and operator from parents
        if random.random() < 0.5:
            child['feature1'] = parent2['feature1']
        if random.random() < 0.5:
            child['feature2'] = parent2['feature2']
        if random.random() < 0.5:
            child['operator'] = parent2['operator']
        
        child['fitness'] = 0.0
        child['feature_data'] = None
        child['ic_values'] = {}
        
        return child
    
    def mutate(self, individual):
        """Apply mutation to an individual"""
        if random.random() < self.mutation_rate:
            # Choose what to mutate
            mutation_type = 'transform' #random.choice(['feature1', 'feature2', 'operator', 'transform'])
            
            # if mutation_type == 'feature1' or mutation_type == 'feature2':
            #     # Mutate feature selection
            #     all_features = {**self.short_midterm_features, **self.mid_longterm_features}
            #     feature_names = list(all_features.keys())
            #     individual[mutation_type] = random.choice(feature_names)
                
            # elif mutation_type == 'operator':
            #     # Mutate operator
            #     individual['operator'] = random.choice(self.operators)
                
            # elif mutation_type == 'transform':
                # Apply transformation to current feature data
            if individual['feature_data'] is not None:
                transform_name = random.choice(list(self.transformations.keys()))
                transform_func = self.transformations[transform_name]
                try:
                    individual['feature_data'] = transform_func(individual['feature_data'])
                except:
                    pass  # Skip if transformation fails
            
            individual['fitness'] = 0.0  # Reset fitness after mutation
            individual['ic_values'] = {}

def run_enhanced_genetic_programming(df, short_midterm_features, mid_longterm_features, 
                                   generations=12, population_size=50, mutation_rate=0.3, ic_threshold=0.05):
    """Run enhanced genetic programming with generation tracking"""
    
    if len(short_midterm_features) == 0 and len(mid_longterm_features) == 0:
        print("⚠️  No features available for genetic programming")
        return []
    
    print(f"\n🧬 ENHANCED GENETIC PROGRAMMING EVOLUTION")
    print(f"Short-midterm pool: {len(short_midterm_features)} features")
    print(f"Mid-longterm pool: {len(mid_longterm_features)} features")
    print(f"Parameters: {generations} generations, {population_size} population, {mutation_rate:.1%} mutation rate")
    print(f"Elite threshold: IC > {ic_threshold}")
    
    # Initialize enhanced genetic programming
    gp_evolution = EnhancedGeneticFeatureEvolution(
        short_midterm_features=short_midterm_features,
        mid_longterm_features=mid_longterm_features,
        df=df,
        generations=generations,
        population_size=population_size,
        mutation_rate=mutation_rate,
        ic_threshold=ic_threshold
    )
    
    # Run evolution with tracking
    generation_data = gp_evolution.evolve_with_tracking()
    
    # Create comprehensive summary
    summary_report = {
        'evolution_summary': {
            'total_generations': generations,
            'population_size': population_size,
            'mutation_rate': mutation_rate,
            'ic_threshold': ic_threshold,
            'short_midterm_pool_size': len(short_midterm_features),
            'mid_longterm_pool_size': len(mid_longterm_features)
        },
        'generation_summaries': [],
        'all_kings': [],
        'evolution_files_generated': []
    }
    
    # Process each generation's data
    for gen_data in generation_data:
        generation = gen_data['generation']
        elite_individuals = gen_data['elite_individuals']
        
        if elite_individuals:
            king = elite_individuals[0]
            summary_report['generation_summaries'].append({
                'generation': generation,
                'elite_count': len(elite_individuals),
                'king_fitness': king['fitness'],
                'avg_fitness': np.mean([ind['fitness'] for ind in elite_individuals])
            })
            
            summary_report['all_kings'].append({
                'generation': generation,
                'fitness': king['fitness'],
                'expression': f"{king['feature1']} {king['operator']} {king['feature2']}",
                'ic_values': king['ic_values']
            })
        
        summary_report['evolution_files_generated'].extend([
            gen_data['csv_file'],
            gen_data['json_file']
        ])
    
    # Save comprehensive summary
    with open('genetic_programming_evolution_summary.json', 'w') as f:
        json.dump(summary_report, f, indent=2)
    print(f"\nEvolution summary saved to genetic_programming_evolution_summary.json")
    
    return generation_data

def save_transformed_features(df, output_path='transformed_features_V_S_2.csv'):
    """Save all transformed features to CSV"""
    # Select columns to save (exclude temporary future return columns if needed)
    df.to_csv(output_path, index=False)
    print(f"Transformed features saved to {output_path}")

def main():
    """Main pipeline execution with enhanced genetic programming"""
    print("=== ENHANCED FEATURE ENGINEERING PIPELINE WITH GENETIC PROGRAMMING ===")
    
    # Step 1: Load data
    print("\n1. Loading data...")
    df = load_data()
    print(f"Loaded data with shape: {df.shape}")
    
    # Step 2: Calculate future returns
    print("\n2. Calculating future returns...")
    df, future_returns = calculate_future_returns(df)
    
    # Step 3: Generate all transformed features
    print("\n3. Applying transformations to all features...")
    transformed_features, feature_metadata = generate_all_features(df)
    
    # Step 3b: Generate Alpha101 sample features
    print("\n3b. Generating Alpha101 sample features (8-12, 14-17)...")
    alpha_features, alpha_metadata = generate_alpha101_features(df)
    transformed_features.update(alpha_features)
    feature_metadata.update(alpha_metadata)
    
    # Step 4: Calculate IC values for all features
    print("\n4. Calculating Information Coefficients...")
    df, feature_metadata = calculate_ics_for_all_features(df, transformed_features, feature_metadata)
    
    # Step 5: Generate main report
    print("\n5. Generating main report...")
    report = generate_report(feature_metadata)
    
    # Step 6: Elite feature selection (IC > 0.05)
    print("\n6. Elite feature selection...")
    elite_features, elite_metadata = elite_check_function(df, feature_metadata, ic_threshold=0.05)
    
    # Step 7: Split features into groups
    print("\n7. Splitting features into groups...")
    short_midterm_features, mid_longterm_features, short_midterm_metadata, mid_longterm_metadata = split_feature_groups(
        df, feature_metadata, group_ic_threshold=0.01
    )
    
    # Step 8: Enhanced Genetic Programming Evolution with Generation Tracking
    print("\n8. Enhanced Genetic Programming Evolution with Generation Tracking...")
    generation_data = run_enhanced_genetic_programming(
        df=df,
        short_midterm_features=short_midterm_features,
        mid_longterm_features=mid_longterm_features,
        generations=12,
        population_size=50,
        mutation_rate=0.3,
        ic_threshold=0.05
    )
    
    # Step 9: Save all results
    print("\n9. Saving all results...")
    save_transformed_features(df)
    
    # Display comprehensive summary
    print("\n" + "="*80)
    print("COMPREHENSIVE FEATURE ANALYSIS SUMMARY WITH GENETIC PROGRAMMING")
    print("="*80)
    
    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"Total features generated: {len(feature_metadata)}")
    print(f"Elite features (IC > 0.05): {len(elite_features)}")
    print(f"Short-midterm features (IC > 0.01): {len(short_midterm_features)}")
    print(f"Mid-longterm features (IC > 0.01): {len(mid_longterm_features)}")
    print(f"🧬 GP Generations processed: {len(generation_data)}")
    
    # Show GP evolution summary
    if generation_data:
        print(f"\n🧬 GENETIC PROGRAMMING EVOLUTION SUMMARY:")
        total_elite_across_generations = sum(len(gen['elite_individuals']) for gen in generation_data)
        print(f"Total elite features across all generations: {total_elite_across_generations}")
        
        # Show top 3 generations
        generations_by_king = sorted(generation_data, 
                                   key=lambda x: x['elite_individuals'][0]['fitness'] if x['elite_individuals'] else 0, 
                                   reverse=True)[:3]
        
        print(f"🏆 TOP 3 GENERATIONS BY KING FITNESS:")
        for i, gen_data in enumerate(generations_by_king, 1):
            if gen_data['elite_individuals']:
                king = gen_data['elite_individuals'][0]
                print(f"{i}. Generation {gen_data['generation']}")
                print(f"   King Fitness: {king['fitness']:.6f}")
                print(f"   Expression: {king['feature1']} {king['operator']} {king['feature2']}")
                print(f"   IC Values: {king['ic_values']}")
        
        print(f"\n📁 GENETIC PROGRAMMING FILES GENERATED:")
        print(f"  - genetic_programming_evolution_summary.json: Complete evolution overview")
        for gen_data in generation_data:
            print(f"  - {gen_data['csv_file']}: Generation {gen_data['generation']} elite features data")
            print(f"  - {gen_data['json_file']}: Generation {gen_data['generation']} detailed report")
    
    # Files generated
    print(f"\n📁 ALL FILES GENERATED:")
    print("Main outputs:")
    print("  - feature_engineering_report_N_S_1.json: Complete feature analysis")
    print("  - transformed_features_V_S_2.csv: All transformed features")
    
    if len(elite_features) > 0:
        print("Elite features:")
        print("  - elite_features.csv: Elite features data")
        print("  - elite_features.json: Elite features report")
    
    if len(short_midterm_features) > 0:
        print("Short-midterm group:")
        print("  - short_midterm_features.csv: Short-midterm features data")
        print("  - short_midterm_features.json: Short-midterm features report")
    
    if len(mid_longterm_features) > 0:
        print("Mid-longterm group:")
        print("  - mid_longterm_features.csv: Mid-longterm features data")
        print("  - mid_longterm_features.json: Mid-longterm features report")
    
    print("\n✅ Enhanced pipeline with generation tracking completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()