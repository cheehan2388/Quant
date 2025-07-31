import numpy  as    np
import pandas as    pd
import talib  as    tb
# -------------------------
def compute_ewma_diff(series, fast_span, slow_span, alpha):
    return series.ewm(span=fast_span, adjust=False).mean() \
         - series.ewm(span=slow_span, adjust=False).mean()

def compute_RSI(series, window):
    arr = series.values
    return tb.RSI(arr, timeperiod = window)

def compute_zscore(series, window):
    m = series.rolling(window).mean()
    s = series.rolling(window).std()
    return (series - m) / s

def compute_minmax(series, window):
    lo = series.rolling(window).min()
    hi = series.rolling(window).max()
    return (2*(series - lo) / (hi - lo + 1e-9))- 1

def compute_ma_double(series, short_w, long_w):
    return series.rolling(short_w).mean() - series.rolling(long_w).mean()

def compute_ma_diff(series, window):
    sma = series.rolling(window).mean()
    
    return sma

def compute_exp_sum(series, window):
    
    decay = np.exp(np.arange(window))
    weight = decay/ decay.sum()
    exp_sum = np.dot(series ,weight )
    return exp_sum

def compute_percentile(series, window):
    def _rank(arr):
        return np.searchsorted(np.sort(arr), arr[-1]) / len(arr)
    return series.rolling(window).apply(_rank, raw=True)

#havent 

def macd(series, short_w, long_w, alpha):
    fast_ema = series.ewm(span=long_w, adjust=False).mean()
    slow_ema = series.ewm(span=short_w, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=alpha, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

#Average True Range
def atr(df, window):
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=window, adjust=False).mean()

# Volume Weighted Average Price
def vwap(df):
    tpv = df['close'] * df['volume']
    return tpv.cumsum() / df['volume'].cumsum()

#Stochastic Oscillator
def stochastic(df, k_period, d_period):
    low_min = df['low'].rolling(k_period).min()
    high_max = df['high'].rolling(k_period).max()
    k = 100 * (df['close'] - low_min) / (high_max - low_min)
    d = k.rolling(d_period).mean()
    return k, d

# on balance volume
def obv(df):
    sign = np.sign(df['close'].diff()).fillna(0)
    return (sign * df['volume']).cumsum()

#
def cci(df, period=20):
    tp = (df['high'] + df['low'] + df['close'])/3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.fabs(x - x.mean()).mean())
    return (tp - sma_tp) / (0.015 * mad)

#
def mfi(df, period=14):
    tp = (df['high'] + df['low'] + df['close'])/3
    mf = tp * df['volume']
    plus = mf.where(tp > tp.shift(1), 0)
    minus = mf.where(tp < tp.shift(1), 0)
    mfr = plus.rolling(period).sum() / minus.rolling(period).sum()
    return 100 - 100/(1 + mfr)

def cmf(df, period=20):
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / \
           (df['high'] - df['low']) * df['volume']
    return mfv.rolling(period).sum() / df['volume'].rolling(period).sum()

def roc(price, period=12):
    return 100 * (price - price.shift(period)) / price.shift(period)

def williams_r(df, period=14):
    hh = df['high'].rolling(period).max()
    ll = df['low'].rolling(period).min()
    return -100 * (hh - df['close']) / (hh - ll)

def donchian(df, period=20):
    upper = df['high'].rolling(period).max()
    lower = df['low'].rolling(period).min()
    return upper, lower

def keltner(df, period=20, k=2):
    typical = (df['high'] + df['low'] + df['close'])/3
    mid = typical.ewm(span=period, adjust=False).mean()
    range_ = atr(df, period)
    return mid + k*range_, mid - k*range_

def ichimoku(df):
    nine_high = df['high'].rolling(9).max()
    nine_low  = df['low'].rolling(9).min()
    tenkan = (nine_high + nine_low)/2
    period26_high = df['high'].rolling(26).max()
    period26_low  = df['low'].rolling(26).min()
    kijun = (period26_high + period26_low)/2
    span_a = ((tenkan + kijun)/2).shift(26)
    period52_high = df['high'].rolling(52).max()
    period52_low  = df['low'].rolling(52).min()
    span_b = ((period52_high + period52_low)/2).shift(26)
    chikou = df['close'].shift(-26)
    return tenkan, kijun, span_a, span_b, chikou

def rolling_std(price, period=20):
    return price.rolling(period).std(ddof=0)

def elder_ray(df, period=13):
    ema13 = df['close'].ewm(span=period, adjust=False).mean()
    bull = df['high'] - ema13
    bear = df['low'] - ema13
    return bull, bear