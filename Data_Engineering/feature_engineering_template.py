import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
import random
from scipy.stats import pearsonr, spearmanr
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import welch, periodogram
import pywt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy.stats import jarque_bera, normaltest
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

def load_data(file_path='../Data/All_data/merged/BTC_merged_V.csv'):
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
    close_path = os.path.join(close_dir, 'Binance_1Hour_BTCUSD_T.csv')
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

# =============================================================================
# SOPHISTICATED FEATURE ENGINEERING FUNCTIONS
# =============================================================================

def fourier_features(series, window=52):
    """Extract Fourier transform features for frequency domain analysis"""
    def rolling_fourier(x):
        if len(x) < window:
            return {
                'dominant_freq': 0.0,
                'power_spectrum_peak': 0.0,
                'spectral_entropy': 0.0,
                'low_freq_power': 0.0,
                'high_freq_power': 0.0
            }
        
        # Apply FFT
        fft_vals = fft(x.values if hasattr(x, 'values') else x)
        freqs = fftfreq(len(x))
        
        # Power spectrum
        power_spectrum = np.abs(fft_vals)**2
        
        # Find dominant frequency (excluding DC component)
        dominant_freq_idx = np.argmax(power_spectrum[1:]) + 1
        dominant_freq = np.abs(freqs[dominant_freq_idx])
        
        # Power spectrum peak
        power_spectrum_peak = np.max(power_spectrum[1:])
        
        # Spectral entropy
        normalized_power = power_spectrum[1:] / np.sum(power_spectrum[1:])
        normalized_power = normalized_power[normalized_power > 0]
        spectral_entropy = -np.sum(normalized_power * np.log2(normalized_power))
        
        # Low and high frequency power
        mid_point = len(freqs) // 2
        low_freq_power = np.sum(power_spectrum[1:mid_point//2])
        high_freq_power = np.sum(power_spectrum[mid_point//2:mid_point])
        
        return {
            'dominant_freq': dominant_freq,
            'power_spectrum_peak': power_spectrum_peak,
            'spectral_entropy': spectral_entropy,
            'low_freq_power': low_freq_power,
            'high_freq_power': high_freq_power
        }
    
    results = series.rolling(window=window, min_periods=window//2).apply(
        lambda x: rolling_fourier(x), raw=False
    )
    
    # Extract individual feature series
    features = {}
    for i, key in enumerate(['dominant_freq', 'power_spectrum_peak', 'spectral_entropy', 'low_freq_power', 'high_freq_power']):
        features[f'{key}'] = pd.Series([rolling_fourier(series.iloc[max(0, i-window+1):i+1])[key] 
                                       for i in range(len(series))], index=series.index)
    
    return features

def wavelet_features(series, wavelet='db4', levels=5):
    """Extract wavelet decomposition features for time-frequency analysis"""
    def rolling_wavelet(x, window=52):
        if len(x) < window:
            return {f'wavelet_energy_L{i}': 0.0 for i in range(levels)}
        
        # Pad signal to power of 2 for efficient computation
        signal = x.values if hasattr(x, 'values') else x
        signal_length = len(signal)
        next_power_2 = 2**int(np.ceil(np.log2(signal_length)))
        padded_signal = np.pad(signal, (0, next_power_2 - signal_length), mode='edge')
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(padded_signal, wavelet, level=levels)
        
        # Calculate energy at each level
        features = {}
        for i in range(levels + 1):  # +1 for approximation coefficients
            if i == 0:
                # Approximation coefficients
                energy = np.sum(coeffs[0]**2)
                features[f'wavelet_approx_energy'] = energy
            else:
                # Detail coefficients
                energy = np.sum(coeffs[i]**2)
                features[f'wavelet_detail_L{i}_energy'] = energy
        
        # Additional wavelet features
        total_energy = sum(features.values())
        for key in features:
            features[key] = features[key] / total_energy if total_energy > 0 else 0
            
        return features
    
    # Apply rolling wavelet analysis
    results = {}
    window = 52
    
    for i in range(len(series)):
        start_idx = max(0, i - window + 1)
        window_data = series.iloc[start_idx:i+1]
        
        if len(window_data) >= 8:  # Minimum length for wavelet
            wavelet_result = rolling_wavelet(window_data)
        else:
            wavelet_result = {f'wavelet_approx_energy': 0.0}
            for j in range(1, levels + 1):
                wavelet_result[f'wavelet_detail_L{j}_energy'] = 0.0
        
        for key, value in wavelet_result.items():
            if key not in results:
                results[key] = []
            results[key].append(value)
    
    # Convert to pandas Series
    for key in results:
        results[key] = pd.Series(results[key], index=series.index)
    
    return results

def sentiment_features(df, sentiment_proxy_columns=None):
    """Process sentiment data and create sentiment-based features"""
    if sentiment_proxy_columns is None:
        # Look for potential sentiment proxy columns
        potential_sentiment_cols = [col for col in df.columns 
                                  if any(keyword in col.lower() 
                                       for keyword in ['premium', 'flow', 'volume', 'funding'])]
        sentiment_proxy_columns = potential_sentiment_cols[:3]  # Use first 3 found
    
    sentiment_features = {}
    
    if not sentiment_proxy_columns:
        # Create synthetic sentiment if no proxy columns found
        print("Warning: No sentiment proxy columns found, creating synthetic sentiment")
        sentiment_proxy_columns = ['synthetic_sentiment']
        # Create synthetic sentiment based on price volatility
        if 'Close' in df.columns:
            returns = df['Close'].pct_change()
            sentiment_features['synthetic_sentiment'] = returns.rolling(24).std()
        else:
            # Use first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                sentiment_features['synthetic_sentiment'] = df[numeric_cols[0]].rolling(24).std()
    
    for col in sentiment_proxy_columns:
        if col in df.columns:
            series = df[col]
            
            # Sentiment momentum
            sentiment_features[f'{col}_sentiment_momentum'] = series.pct_change(12)
            
            # Sentiment volatility
            sentiment_features[f'{col}_sentiment_volatility'] = series.rolling(24).std()
            
            # Sentiment regime (above/below rolling median)
            rolling_median = series.rolling(52).median()
            sentiment_features[f'{col}_sentiment_regime'] = (series > rolling_median).astype(float)
            
            # Sentiment z-score
            rolling_mean = series.rolling(52).mean()
            rolling_std = series.rolling(52).std()
            sentiment_features[f'{col}_sentiment_zscore'] = (series - rolling_mean) / rolling_std
            
            # Sentiment divergence (current vs long-term trend)
            long_term_ma = series.rolling(168).mean()  # Weekly MA
            short_term_ma = series.rolling(24).mean()   # Daily MA
            sentiment_features[f'{col}_sentiment_divergence'] = (short_term_ma - long_term_ma) / long_term_ma
    
    return sentiment_features

def regime_detection_features(df, price_col='Close', window=168):
    """Implement regime detection for adaptive feature selection"""
    if price_col not in df.columns:
        # Use first numeric column if Close is not available
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            price_col = numeric_cols[0]
        else:
            return {}
    
    price_series = df[price_col]
    returns = price_series.pct_change()
    
    regime_features = {}
    
    # 1. Volatility Regime Detection
    volatility = returns.rolling(24).std()
    vol_threshold_high = volatility.rolling(window).quantile(0.75)
    vol_threshold_low = volatility.rolling(window).quantile(0.25)
    
    volatility_regime = pd.Series(np.nan, index=df.index)
    volatility_regime[volatility > vol_threshold_high] = 2  # High vol
    volatility_regime[volatility < vol_threshold_low] = 0   # Low vol
    volatility_regime = volatility_regime.fillna(1)         # Medium vol
    
    regime_features['volatility_regime'] = volatility_regime
    
    # 2. Trend Regime Detection using dual moving averages
    short_ma = price_series.rolling(24).mean()
    long_ma = price_series.rolling(168).mean()
    
    trend_regime = pd.Series(np.nan, index=df.index)
    trend_regime[short_ma > long_ma * 1.02] = 1   # Uptrend
    trend_regime[short_ma < long_ma * 0.98] = -1  # Downtrend
    trend_regime = trend_regime.fillna(0)         # Sideways
    
    regime_features['trend_regime'] = trend_regime
    
    # 3. Mean Reversion vs Momentum Regime
    # Use autocorrelation to detect persistence
    def rolling_autocorr(x, lag=1):
        if len(x) < lag + 1:
            return 0
        return pd.Series(x).autocorr(lag=lag)
    
    autocorr = returns.rolling(window).apply(lambda x: rolling_autocorr(x, lag=1))
    
    momentum_regime = pd.Series(np.nan, index=df.index)
    momentum_regime[autocorr > 0.1] = 1    # Momentum regime
    momentum_regime[autocorr < -0.1] = -1  # Mean reversion regime
    momentum_regime = momentum_regime.fillna(0)  # Neutral
    
    regime_features['momentum_regime'] = momentum_regime
    
    # 4. Market Stress Regime (using multiple indicators)
    # Combine volatility, large moves, and trend strength
    large_moves = (np.abs(returns) > returns.rolling(window).quantile(0.95)).rolling(24).sum()
    trend_strength = np.abs(short_ma - long_ma) / long_ma
    
    stress_indicator = (volatility / volatility.rolling(window).median() + 
                       large_moves / 24 + 
                       trend_strength) / 3
    
    stress_threshold = stress_indicator.rolling(window).quantile(0.8)
    market_stress_regime = (stress_indicator > stress_threshold).astype(float)
    regime_features['market_stress_regime'] = market_stress_regime
    
    # 5. Liquidity Regime (if volume data available)
    volume_cols = [col for col in df.columns if 'volume' in col.lower()]
    if volume_cols:
        volume_series = df[volume_cols[0]]
        volume_ma = volume_series.rolling(24).mean()
        volume_threshold = volume_series.rolling(window).quantile(0.3)
        
        liquidity_regime = (volume_ma > volume_threshold).astype(float)
        regime_features['liquidity_regime'] = liquidity_regime
    
    return regime_features

def portfolio_optimization_features(df, feature_columns, returns_columns, window=168):
    """Apply Modern Portfolio Theory for feature weighting and selection"""
    
    # Calculate feature returns (or changes) for portfolio optimization
    feature_returns = pd.DataFrame()
    
    for col in feature_columns:
        if col in df.columns:
            feature_returns[col] = df[col].pct_change()
    
    if len(feature_returns.columns) < 2:
        return {}
    
    portfolio_features = {}
    
    def calculate_portfolio_weights(returns_matrix, method='max_sharpe'):
        """Calculate portfolio weights using different methods"""
        if returns_matrix.empty or returns_matrix.shape[1] < 2:
            return np.array([])
        
        # Remove NaN values
        clean_returns = returns_matrix.dropna()
        if len(clean_returns) < 10:  # Need minimum observations
            return np.array([])
        
        # Calculate mean returns and covariance matrix
        mean_returns = clean_returns.mean()
        
        # Use Ledoit-Wolf shrinkage estimator for robust covariance
        try:
            lw = LedoitWolf()
            cov_matrix = pd.DataFrame(
                lw.fit(clean_returns).covariance_,
                index=clean_returns.columns,
                columns=clean_returns.columns
            )
        except:
            cov_matrix = clean_returns.cov()
        
        n_assets = len(mean_returns)
        
        if method == 'max_sharpe':
            # Maximum Sharpe ratio portfolio
            def negative_sharpe(weights):
                portfolio_return = np.sum(mean_returns * weights)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return -(portfolio_return / portfolio_vol) if portfolio_vol > 0 else -np.inf
            
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            try:
                result = minimize(negative_sharpe, 
                                np.array([1/n_assets]*n_assets), 
                                method='SLSQP', 
                                bounds=bounds, 
                                constraints=constraints)
                return result.x if result.success else np.array([1/n_assets]*n_assets)
            except:
                return np.array([1/n_assets]*n_assets)
        
        elif method == 'min_variance':
            # Minimum variance portfolio
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            try:
                result = minimize(portfolio_variance, 
                                np.array([1/n_assets]*n_assets), 
                                method='SLSQP', 
                                bounds=bounds, 
                                constraints=constraints)
                return result.x if result.success else np.array([1/n_assets]*n_assets)
            except:
                return np.array([1/n_assets]*n_assets)
        
        else:  # equal weight
            return np.array([1/n_assets]*n_assets)
    
    # Rolling portfolio optimization
    for i in range(window, len(df)):
        start_idx = i - window
        
        # Get window data
        window_returns = feature_returns.iloc[start_idx:i]
        
        if len(window_returns.dropna()) < window//2:
            continue
        
        # Calculate different portfolio weights
        max_sharpe_weights = calculate_portfolio_weights(window_returns, 'max_sharpe')
        min_var_weights = calculate_portfolio_weights(window_returns, 'min_variance')
        equal_weights = calculate_portfolio_weights(window_returns, 'equal')
        
        # Store weights as features
        if len(max_sharpe_weights) == len(feature_columns):
            for j, col in enumerate(feature_columns[:len(max_sharpe_weights)]):
                if f'mpt_max_sharpe_weight_{col}' not in portfolio_features:
                    portfolio_features[f'mpt_max_sharpe_weight_{col}'] = pd.Series(np.nan, index=df.index)
                portfolio_features[f'mpt_max_sharpe_weight_{col}'].iloc[i] = max_sharpe_weights[j]
        
        if len(min_var_weights) == len(feature_columns):
            for j, col in enumerate(feature_columns[:len(min_var_weights)]):
                if f'mpt_min_var_weight_{col}' not in portfolio_features:
                    portfolio_features[f'mpt_min_var_weight_{col}'] = pd.Series(np.nan, index=df.index)
                portfolio_features[f'mpt_min_var_weight_{col}'].iloc[i] = min_var_weights[j]
    
    # Forward fill weights
    for key in portfolio_features:
        portfolio_features[key] = portfolio_features[key].fillna(method='ffill').fillna(0)
    
    # Calculate portfolio-weighted composite features
    if len(max_sharpe_weights) == len(feature_columns):
        weighted_feature_max_sharpe = pd.Series(0, index=df.index)
        weighted_feature_min_var = pd.Series(0, index=df.index)
        
        for i, col in enumerate(feature_columns[:len(max_sharpe_weights)]):
            if col in df.columns:
                # Max Sharpe weighted feature
                weight_col_sharpe = f'mpt_max_sharpe_weight_{col}'
                if weight_col_sharpe in portfolio_features:
                    weighted_feature_max_sharpe += (df[col] * portfolio_features[weight_col_sharpe])
                
                # Min variance weighted feature
                weight_col_var = f'mpt_min_var_weight_{col}'
                if weight_col_var in portfolio_features:
                    weighted_feature_min_var += (df[col] * portfolio_features[weight_col_var])
        
        portfolio_features['mpt_composite_max_sharpe_feature'] = weighted_feature_max_sharpe
        portfolio_features['mpt_composite_min_var_feature'] = weighted_feature_min_var
    
    return portfolio_features

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
    """Generate all transformed features including sophisticated financial analysis"""
    
    # Get all numeric columns (exclude datetime and future returns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['datetime'] + [col for col in df.columns if 'future_return' in col]
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"Processing {len(feature_cols)} base features with sophisticated transformations...")
    
    # Define basic transformation tools
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
    
    # 1. Apply basic transformations to all columns
    print("\n🔧 Applying basic transformations...")
    for col in feature_cols:
        print(f"  Processing column: {col}")
        
        for transform_name, transform_func in transformations.items():
            feature_name = f"{col}_{transform_name}"
            try:
                transformed_series = transform_func(df[col])
                transformed_features[feature_name] = transformed_series
                
                # Store metadata
                feature_metadata[feature_name] = {
                    'base_column': col,
                    'transformation': transform_name,
                    'feature_type': 'basic',
                    'ic_values': {},
                    'max_ic': 0.0,
                    'best_shift': None
                }
                
            except Exception as e:
                print(f"    Error processing {feature_name}: {str(e)}")
                continue
    
    # 2. Apply sophisticated features to selected columns
    print("\n🧠 Applying sophisticated features...")
    
    # Select top columns for sophisticated analysis (to manage computational load)
    selected_cols = feature_cols[:min(10, len(feature_cols))]  # Limit to first 10 columns
    
    for col in selected_cols:
        print(f"  Sophisticated analysis for: {col}")
        
        # Fourier Transform Features
        try:
            fourier_feats = fourier_features(df[col])
            for fourier_name, fourier_series in fourier_feats.items():
                feature_name = f"{col}_{fourier_name}"
                transformed_features[feature_name] = fourier_series
                feature_metadata[feature_name] = {
                    'base_column': col,
                    'transformation': fourier_name,
                    'feature_type': 'fourier',
                    'ic_values': {},
                    'max_ic': 0.0,
                    'best_shift': None
                }
            print(f"    ✓ Added {len(fourier_feats)} Fourier features")
        except Exception as e:
            print(f"    ⚠️ Fourier analysis failed for {col}: {str(e)}")
        
        # Wavelet Features
        try:
            wavelet_feats = wavelet_features(df[col])
            for wavelet_name, wavelet_series in wavelet_feats.items():
                feature_name = f"{col}_{wavelet_name}"
                transformed_features[feature_name] = wavelet_series
                feature_metadata[feature_name] = {
                    'base_column': col,
                    'transformation': wavelet_name,
                    'feature_type': 'wavelet',
                    'ic_values': {},
                    'max_ic': 0.0,
                    'best_shift': None
                }
            print(f"    ✓ Added {len(wavelet_feats)} Wavelet features")
        except Exception as e:
            print(f"    ⚠️ Wavelet analysis failed for {col}: {str(e)}")
    
    # 3. Add Sentiment Features
    print("\n💭 Adding sentiment features...")
    try:
        sentiment_feats = sentiment_features(df)
        for sentiment_name, sentiment_series in sentiment_feats.items():
            transformed_features[sentiment_name] = sentiment_series
            feature_metadata[sentiment_name] = {
                'base_column': 'multiple',
                'transformation': 'sentiment_analysis',
                'feature_type': 'sentiment',
                'ic_values': {},
                'max_ic': 0.0,
                'best_shift': None
            }
        print(f"  ✓ Added {len(sentiment_feats)} sentiment features")
    except Exception as e:
        print(f"  ⚠️ Sentiment analysis failed: {str(e)}")
    
    # 4. Add Regime Detection Features  
    print("\n🎯 Adding regime detection features...")
    try:
        regime_feats = regime_detection_features(df)
        for regime_name, regime_series in regime_feats.items():
            transformed_features[regime_name] = regime_series
            feature_metadata[regime_name] = {
                'base_column': 'multiple',
                'transformation': 'regime_detection',
                'feature_type': 'regime',
                'ic_values': {},
                'max_ic': 0.0,
                'best_shift': None
            }
        print(f"  ✓ Added {len(regime_feats)} regime detection features")
    except Exception as e:
        print(f"  ⚠️ Regime detection failed: {str(e)}")
    
    # 5. Add Portfolio Optimization Features
    print("\n📈 Adding portfolio optimization features...")
    try:
        # Use a subset of features for portfolio optimization
        portfolio_feature_cols = selected_cols[:min(5, len(selected_cols))]  # Top 5 for MPT
        returns_cols = [col for col in df.columns if 'future_return' in col]
        
        portfolio_feats = portfolio_optimization_features(df, portfolio_feature_cols, returns_cols)
        for portfolio_name, portfolio_series in portfolio_feats.items():
            transformed_features[portfolio_name] = portfolio_series
            feature_metadata[portfolio_name] = {
                'base_column': 'multiple',
                'transformation': 'portfolio_optimization',
                'feature_type': 'portfolio',
                'ic_values': {},
                'max_ic': 0.0,
                'best_shift': None
            }
        print(f"  ✓ Added {len(portfolio_feats)} portfolio optimization features")
    except Exception as e:
        print(f"  ⚠️ Portfolio optimization failed: {str(e)}")
    
    print(f"\n🎉 Total features generated: {len(transformed_features)}")
    print(f"   - Basic transformations: {sum(1 for meta in feature_metadata.values() if meta.get('feature_type') == 'basic')}")
    print(f"   - Fourier features: {sum(1 for meta in feature_metadata.values() if meta.get('feature_type') == 'fourier')}")
    print(f"   - Wavelet features: {sum(1 for meta in feature_metadata.values() if meta.get('feature_type') == 'wavelet')}")
    print(f"   - Sentiment features: {sum(1 for meta in feature_metadata.values() if meta.get('feature_type') == 'sentiment')}")
    print(f"   - Regime features: {sum(1 for meta in feature_metadata.values() if meta.get('feature_type') == 'regime')}")
    print(f"   - Portfolio features: {sum(1 for meta in feature_metadata.values() if meta.get('feature_type') == 'portfolio')}")
    
    return transformed_features, feature_metadata

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
                'rolling_zscore_12', 'rolling_zscore_24', 'rolling_zscore_52'
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
            'feature_type': meta.get('feature_type', 'unknown'),
            'ic_values': meta['ic_values'],
            'max_ic': meta['max_ic'],
            'best_shift': meta['best_shift'],
            'abs_max_ic': abs(meta['max_ic'])
        }
    
    return report

def adaptive_feature_selection_with_regimes(df, feature_metadata, regime_threshold=0.01):
    """Perform regime-aware adaptive feature selection"""
    print(f"\n🎯 REGIME-AWARE ADAPTIVE FEATURE SELECTION")
    print(f"Threshold: IC > {regime_threshold}")
    
    # Get regime features
    regime_features = [name for name, meta in feature_metadata.items() 
                      if meta.get('feature_type') == 'regime']
    
    if not regime_features:
        print("⚠️ No regime features found, performing standard selection")
        return standard_feature_selection(df, feature_metadata, regime_threshold)
    
    regime_adaptive_features = {}
    regime_metadata = {}
    
    # Define regime conditions
    regime_conditions = {}
    
    # Check if we have regime features
    if 'volatility_regime' in df.columns:
        regime_conditions['high_volatility'] = df['volatility_regime'] == 2
        regime_conditions['low_volatility'] = df['volatility_regime'] == 0
    
    if 'trend_regime' in df.columns:
        regime_conditions['uptrend'] = df['trend_regime'] == 1
        regime_conditions['downtrend'] = df['trend_regime'] == -1
        regime_conditions['sideways'] = df['trend_regime'] == 0
    
    if 'momentum_regime' in df.columns:
        regime_conditions['momentum'] = df['momentum_regime'] == 1
        regime_conditions['mean_reversion'] = df['momentum_regime'] == -1
    
    if 'market_stress_regime' in df.columns:
        regime_conditions['market_stress'] = df['market_stress_regime'] == 1
        regime_conditions['normal_market'] = df['market_stress_regime'] == 0
    
    print(f"Identified {len(regime_conditions)} regime conditions")
    
    # For each feature, calculate regime-conditional IC
    for feature_name, metadata in feature_metadata.items():
        if feature_name not in df.columns:
            continue
            
        feature_series = df[feature_name]
        regime_ics = {}
        
        # Calculate IC for each regime condition
        for regime_name, regime_mask in regime_conditions.items():
            if regime_mask.sum() < 50:  # Need minimum observations
                continue
                
            # Calculate IC for each target under this regime
            regime_ic_values = {}
            for shift in [-2, -8, -24]:
                future_return_col = f'future_return_{abs(shift)}'
                if future_return_col in df.columns:
                    # Calculate IC only for regime periods
                    regime_feature = feature_series[regime_mask]
                    regime_target = df[future_return_col][regime_mask]
                    
                    if len(regime_feature.dropna()) > 10:
                        ic = calculate_ic(regime_feature, regime_target)
                        regime_ic_values[f'shift_{shift}'] = ic
                    else:
                        regime_ic_values[f'shift_{shift}'] = 0.0
            
            if regime_ic_values:
                # Get max absolute IC for this regime
                abs_ics = {k: abs(v) for k, v in regime_ic_values.items()}
                best_shift_key = max(abs_ics.keys(), key=abs_ics.get) if abs_ics else 'shift_-2'
                max_ic = regime_ic_values[best_shift_key]
                regime_ics[regime_name] = {
                    'ic_values': regime_ic_values,
                    'max_ic': max_ic,
                    'best_shift': best_shift_key
                }
        
        # Check if feature is significant in any regime
        significant_regimes = [regime for regime, data in regime_ics.items() 
                             if abs(data['max_ic']) > regime_threshold]
        
        if significant_regimes:
            # Calculate overall regime-weighted IC
            total_observations = len(df)
            weighted_ic = 0
            total_weight = 0
            
            for regime_name in significant_regimes:
                regime_mask = regime_conditions[regime_name]
                regime_weight = regime_mask.sum() / total_observations
                regime_ic = regime_ics[regime_name]['max_ic']
                
                weighted_ic += regime_weight * abs(regime_ic)
                total_weight += regime_weight
            
            if total_weight > 0:
                final_weighted_ic = weighted_ic / total_weight
                
                regime_adaptive_features[feature_name] = feature_series
                regime_metadata[feature_name] = {
                    'base_column': metadata['base_column'],
                    'transformation': metadata['transformation'],
                    'feature_type': metadata.get('feature_type', 'unknown'),
                    'regime_ics': regime_ics,
                    'significant_regimes': significant_regimes,
                    'weighted_ic': final_weighted_ic,
                    'original_ic_values': metadata['ic_values'],
                    'original_max_ic': metadata['max_ic']
                }
    
    print(f"Found {len(regime_adaptive_features)} regime-adaptive features")
    
    # Save regime-adaptive features
    if regime_adaptive_features:
        regime_df = pd.DataFrame({'datetime': df['datetime']})
        for feature_name, feature_series in regime_adaptive_features.items():
            regime_df[feature_name] = feature_series
        
        regime_csv = 'regime_adaptive_features.csv'
        regime_df.to_csv(regime_csv, index=False)
        print(f"Regime-adaptive features saved to {regime_csv}")
        
        # Create regime-adaptive report
        sorted_regime_features = sorted(
            regime_metadata.items(),
            key=lambda x: x[1]['weighted_ic'],
            reverse=True
        )
        
        regime_report = {
            'summary': {
                'total_features': len(regime_adaptive_features),
                'regime_threshold': regime_threshold,
                'average_weighted_ic': np.mean([meta['weighted_ic'] for meta in regime_metadata.values()]),
                'max_weighted_ic': max([meta['weighted_ic'] for meta in regime_metadata.values()]) if regime_metadata else 0,
                'regimes_analyzed': list(regime_conditions.keys()),
                'top_10_features': [name for name, _ in sorted_regime_features[:10]]
            },
            'features': {}
        }
        
        for feature_name, meta in sorted_regime_features:
            regime_report['features'][feature_name] = {
                'base_column': meta['base_column'],
                'transformation': meta['transformation'],
                'feature_type': meta['feature_type'],
                'weighted_ic': meta['weighted_ic'],
                'significant_regimes': meta['significant_regimes'],
                'regime_ics': meta['regime_ics'],
                'original_max_ic': meta['original_max_ic']
            }
        
        with open('regime_adaptive_features.json', 'w') as f:
            json.dump(regime_report, f, indent=2)
        print("Regime-adaptive features report saved to regime_adaptive_features.json")
    
    return regime_adaptive_features, regime_metadata

def standard_feature_selection(df, feature_metadata, threshold):
    """Standard feature selection fallback"""
    selected_features = {}
    selected_metadata = {}
    
    for feature_name, metadata in feature_metadata.items():
        if abs(metadata['max_ic']) > threshold:
            selected_features[feature_name] = df[feature_name]
            selected_metadata[feature_name] = metadata
    
    return selected_features, selected_metadata

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
            mutation_type = random.choice(['feature1', 'feature2', 'operator', 'transform'])
            
            if mutation_type == 'feature1' or mutation_type == 'feature2':
                # Mutate feature selection
                all_features = {**self.short_midterm_features, **self.mid_longterm_features}
                feature_names = list(all_features.keys())
                individual[mutation_type] = random.choice(feature_names)
                
            elif mutation_type == 'operator':
                # Mutate operator
                individual['operator'] = random.choice(self.operators)
                
            elif mutation_type == 'transform':
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
    """Main pipeline execution with sophisticated feature engineering and regime-aware selection"""
    print("=== SOPHISTICATED FEATURE ENGINEERING PIPELINE WITH ADVANCED ANALYTICS ===")
    
    # Step 1: Load data
    print("\n1. 📊 Loading data...")
    df = load_data()
    print(f"Loaded data with shape: {df.shape}")
    
    # Step 2: Calculate future returns
    print("\n2. 📈 Calculating future returns...")
    df, future_returns = calculate_future_returns(df)
    
    # Step 3: Generate all sophisticated features
    print("\n3. 🧠 Generating sophisticated features (Fourier, Wavelet, Sentiment, Regime, Portfolio)...")
    transformed_features, feature_metadata = generate_all_features(df)
    
    # Step 4: Calculate IC values for all features
    print("\n4. 🔍 Calculating Information Coefficients...")
    df, feature_metadata = calculate_ics_for_all_features(df, transformed_features, feature_metadata)
    
    # Step 5: Generate comprehensive report
    print("\n5. 📋 Generating comprehensive feature report...")
    report = generate_report(feature_metadata, output_path='sophisticated_feature_engineering_report.json')
    
    # Step 6: Elite feature selection (IC > 0.05)
    print("\n6. 🏆 Elite feature selection...")
    elite_features, elite_metadata = elite_check_function(df, feature_metadata, ic_threshold=0.05)
    
    # Step 7: Split features into groups
    print("\n7. 📂 Splitting features into groups...")
    short_midterm_features, mid_longterm_features, short_midterm_metadata, mid_longterm_metadata = split_feature_groups(
        df, feature_metadata, group_ic_threshold=0.01
    )
    
    # Step 8: Regime-Aware Adaptive Feature Selection
    print("\n8. 🎯 Regime-aware adaptive feature selection...")
    regime_adaptive_features, regime_adaptive_metadata = adaptive_feature_selection_with_regimes(
        df, feature_metadata, regime_threshold=0.02
    )
    
    # Step 9: Enhanced Genetic Programming Evolution with Generation Tracking
    print("\n9. 🧬 Enhanced Genetic Programming Evolution with Generation Tracking...")
    generation_data = run_enhanced_genetic_programming(
        df=df,
        short_midterm_features=short_midterm_features,
        mid_longterm_features=mid_longterm_features,
        generations=12,
        population_size=50,
        mutation_rate=0.3,
        ic_threshold=0.05
    )
    
    # Step 10: Save all sophisticated results
    print("\n10. 💾 Saving all sophisticated results...")
    save_transformed_features(df, output_path='sophisticated_transformed_features.csv')
    
    # Display comprehensive summary
    print("\n" + "="*80)
    print("🚀 SOPHISTICATED FEATURE ENGINEERING ANALYSIS SUMMARY")
    print("="*80)
    
    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"Total features generated: {len(feature_metadata)}")
    
    # Feature type breakdown
    feature_type_counts = {}
    for meta in feature_metadata.values():
        ftype = meta.get('feature_type', 'unknown')
        feature_type_counts[ftype] = feature_type_counts.get(ftype, 0) + 1
    
    print(f"\n🎯 FEATURE TYPE BREAKDOWN:")
    for ftype, count in feature_type_counts.items():
        print(f"   {ftype.capitalize()} features: {count}")
    
    print(f"\n🏆 FEATURE SELECTION RESULTS:")
    print(f"Elite features (IC > 0.05): {len(elite_features)}")
    print(f"Short-midterm features (IC > 0.01): {len(short_midterm_features)}")
    print(f"Mid-longterm features (IC > 0.01): {len(mid_longterm_features)}")
    print(f"Regime-adaptive features: {len(regime_adaptive_features)}")
    print(f"🧬 GP Generations processed: {len(generation_data)}")
    
    # Show sophisticated feature performance
    if feature_metadata:
        print(f"\n⚡ SOPHISTICATED FEATURES PERFORMANCE:")
        
        # Get best features by type
        best_by_type = {}
        for feature_name, meta in feature_metadata.items():
            ftype = meta.get('feature_type', 'unknown')
            if ftype not in best_by_type or abs(meta['max_ic']) > abs(best_by_type[ftype]['max_ic']):
                best_by_type[ftype] = {
                    'name': feature_name,
                    'max_ic': meta['max_ic'],
                    'base_column': meta['base_column']
                }
        
        for ftype, best_info in best_by_type.items():
            print(f"   Best {ftype} feature: {best_info['name']} (IC: {best_info['max_ic']:.6f})")
    
    # Show regime-adaptive results
    if regime_adaptive_features:
        print(f"\n🎯 REGIME-ADAPTIVE SELECTION RESULTS:")
        top_regime_features = sorted(regime_adaptive_metadata.items(), 
                                   key=lambda x: x[1]['weighted_ic'], reverse=True)[:5]
        
        for i, (feature_name, meta) in enumerate(top_regime_features, 1):
            print(f"{i}. {feature_name}")
            print(f"   Weighted IC: {meta['weighted_ic']:.6f}")
            print(f"   Significant regimes: {', '.join(meta['significant_regimes'])}")
    
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
    
    # Files generated
    print(f"\n📁 SOPHISTICATED FILES GENERATED:")
    print("📊 Main outputs:")
    print("  - sophisticated_feature_engineering_report.json: Complete sophisticated feature analysis")
    print("  - sophisticated_transformed_features.csv: All sophisticated transformed features")
    
    if len(elite_features) > 0:
        print("🏆 Elite features:")
        print("  - elite_features.csv: Elite features data")
        print("  - elite_features.json: Elite features report")
    
    if len(short_midterm_features) > 0:
        print("⏰ Short-midterm group:")
        print("  - short_midterm_features.csv: Short-midterm features data")
        print("  - short_midterm_features.json: Short-midterm features report")
    
    if len(mid_longterm_features) > 0:
        print("📈 Mid-longterm group:")
        print("  - mid_longterm_features.csv: Mid-longterm features data")
        print("  - mid_longterm_features.json: Mid-longterm features report")
    
    if len(regime_adaptive_features) > 0:
        print("🎯 Regime-adaptive features:")
        print("  - regime_adaptive_features.csv: Regime-adaptive features data")
        print("  - regime_adaptive_features.json: Regime-adaptive features report")
    
    if generation_data:
        print("🧬 Genetic programming outputs:")
        print("  - genetic_programming_evolution_summary.json: Complete evolution overview")
        print(f"  - generation_X_elite_features.csv: Elite features for each generation")
        print(f"  - generation_X_elite_report.json: Detailed reports for each generation")
    
    print(f"\n🎊 SOPHISTICATED FEATURE ENGINEERING CAPABILITIES ADDED:")
    print("  ✅ Fourier Transform Analysis - Frequency domain features")
    print("  ✅ Wavelet Analysis - Time-frequency decomposition")
    print("  ✅ Sentiment Feature Engineering - Market sentiment indicators")
    print("  ✅ Regime Detection - Adaptive market regime identification")
    print("  ✅ Modern Portfolio Theory - Optimal feature weighting")
    print("  ✅ Regime-Aware Feature Selection - Context-dependent feature filtering")
    print("  ✅ Enhanced Genetic Programming - Advanced feature combination")
    
    print("\n🚀 Sophisticated pipeline with advanced financial analytics completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()