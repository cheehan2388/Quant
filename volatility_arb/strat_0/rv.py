import pandas as pd
import numpy as np


def load_hourly_close(file_path: str) -> pd.DataFrame:
    """Load hourly BTC close data with columns ['timestamp','close'].

    Expects CSV with headers like 'Date,Closeprice' as seen in `1hbtc.csv`.
    Returns UTC-naive timestamps; caller can localize if needed.
    """
    df = pd.read_csv(file_path)
    # Standardize column names
    if 'Date' in df.columns:
        df = df.rename(columns={'Date': 'timestamp'})
    if 'Closeprice' in df.columns:
        df = df.rename(columns={'Closeprice': 'close'})

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df[['timestamp', 'close']].dropna()
    return df


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add hourly log returns column 'ret' to dataframe."""
    out = df.copy()
    out['ret'] = np.log(out['close']).diff()
    out = out.dropna().reset_index(drop=True)
    return out


def realized_variance_hourly(df: pd.DataFrame, hours: int = 24 * 7) -> pd.DataFrame:
    """Compute rolling realized variance over a horizon in hours using hourly log returns.

    - Returns a DataFrame with ['timestamp', 'rv'] where rv is annualized variance (per year),
      assuming 365*24 trading hours.
    """
    hourly_per_year = 365 * 24
    ret = df['ret'].to_numpy()
    # rolling sum of squared returns
    sq = ret ** 2
    window = hours
    # use convolution for efficiency
    rolling_sum = np.convolve(sq, np.ones(window, dtype=float), 'valid')
    rv = (rolling_sum) * hourly_per_year / window

    rv_series = pd.Series(rv, index=df.index[window - 1:])
    out = pd.DataFrame({
        'timestamp': df.loc[df.index[window - 1:], 'timestamp'].values,
        'rv': rv_series.values,
    })
    return out


def realized_vol_from_variance(rv: pd.Series) -> pd.Series:
    """Convert variance to volatility (stdev)."""
    return np.sqrt(rv)


