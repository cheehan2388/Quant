"""
Feature Engineering module for processing raw market data and generating derived features.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available, using manual calculations.")


# === Abstract Base ===
class FeatureEngineer(ABC):
    @abstractmethod
    def calculate_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        pass


# === Statistical Features ===
class StatisticalFeatureEngineer(FeatureEngineer):
    def __init__(self, window_size: int = 20, min_periods: int = 5):
        self.window_size = window_size
        self.min_periods = min_periods

    def get_feature_names(self) -> List[str]:
        return [
            'zscore', 'zscore_mean', 'zscore_std', 'zscore_current_value',
            'percentile_rank', 'rolling_mean', 'rolling_std', 'rolling_min', 'rolling_max'
        ]

    def calculate_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            series = next(
                (data[col] for col in ['open_interest', 'close', 'value']
                 if col in data.columns),
                data.select_dtypes(include=[np.number]).iloc[:, 0] if not data.empty else None
            )
            if series is None or len(series) < self.min_periods:
                return {'valid': False, 'error': 'Insufficient or missing numeric data'}

            rolling = series.rolling(self.window_size, min_periods=self.min_periods)
            mean, std = rolling.mean(), rolling.std()
            min_, max_ = rolling.min(), rolling.max()

            current = series.iloc[-1]
            m, s = mean.iloc[-1], std.iloc[-1]
            z = 0.0 if s == 0 or np.isnan(s) else (current - m) / s
            pctl = (series <= current).mean() * 100

            return {
                'zscore': float(z),
                'zscore_mean': float(m),
                'zscore_std': float(s),
                'zscore_current_value': float(current),
                'percentile_rank': float(pctl),
                'rolling_mean': float(m),
                'rolling_std': float(s),
                'rolling_min': float(min_.iloc[-1]),
                'rolling_max': float(max_.iloc[-1]),
                'valid': True,
            }
        except Exception as e:
            logger.exception("Statistical feature error:")
            return {'valid': False, 'error': str(e)}


# === Technical Features ===
class TechnicalFeatureEngineer(FeatureEngineer):
    def __init__(self, periods: Optional[Dict[str, int]] = None):
        self.periods = periods or {
            'sma_fast': 5, 'sma_slow': 20,
            'ema_fast': 12, 'ema_slow': 26,
            'rsi': 14, 'bb': 20
        }

    def get_feature_names(self) -> List[str]:
        return [
            'sma_fast', 'sma_slow', 'sma_cross', 'sma_distance',
            'ema_fast', 'ema_slow', 'ema_cross', 'ema_distance',
            'rsi', 'rsi_overbought', 'rsi_oversold',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_position', 'bb_squeeze',
            'price_change', 'price_change_pct', 'volatility', 'current_price'
        ]

    def calculate_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            series = next(
                (data[col] for col in ['close', 'open_interest']
                 if col in data.columns),
                data.select_dtypes(include=[np.number]).iloc[:, 0] if not data.empty else None
            )
            if series is None or len(series) < max(self.periods.values()):
                return {'valid': False, 'error': 'Insufficient price data'}

            prices = series.values

            if TALIB_AVAILABLE:
                sma_f = talib.SMA(prices, timeperiod=self.periods['sma_fast'])
                sma_s = talib.SMA(prices, timeperiod=self.periods['sma_slow'])
                ema_f = talib.EMA(prices, timeperiod=self.periods['ema_fast'])
                ema_s = talib.EMA(prices, timeperiod=self.periods['ema_slow'])
                rsi = talib.RSI(prices, timeperiod=self.periods['rsi'])
                bb_u, bb_m, bb_l = talib.BBANDS(prices, timeperiod=self.periods['bb'])
            else:
                sma_f = series.rolling(self.periods['sma_fast']).mean().values
                sma_s = series.rolling(self.periods['sma_slow']).mean().values
                ema_f = series.ewm(span=self.periods['ema_fast']).mean().values
                ema_s = series.ewm(span=self.periods['ema_slow']).mean().values
                rsi = self._calculate_rsi(series, self.periods['rsi']).values
                bb_m = series.rolling(self.periods['bb']).mean().values
                bb_std = series.rolling(self.periods['bb']).std().values
                bb_u, bb_l = bb_m + 2 * bb_std, bb_m - 2 * bb_std

            cp = prices[-1]
            f = lambda x: float(x[-1]) if len(x) and not np.isnan(x[-1]) else 0
            rsival = f(rsi)

            return {
                'sma_fast': f(sma_f),
                'sma_slow': f(sma_s),
                'sma_cross': int(np.sign(f(sma_f) - f(sma_s))),
                'sma_distance': (f(sma_f) - f(sma_s)) / f(sma_s) if f(sma_s) != 0 else 0,
                'ema_fast': f(ema_f),
                'ema_slow': f(ema_s),
                'ema_cross': int(np.sign(f(ema_f) - f(ema_s))),
                'ema_distance': (f(ema_f) - f(ema_s)) / f(ema_s) if f(ema_s) != 0 else 0,
                'rsi': rsival,
                'rsi_overbought': int(rsival > 70),
                'rsi_oversold': int(rsival < 30),
                'bb_upper': f(bb_u),
                'bb_middle': f(bb_m),
                'bb_lower': f(bb_l),
                'bb_position': (cp - f(bb_l)) / (f(bb_u) - f(bb_l)) if f(bb_u) != f(bb_l) else 0.5,
                'bb_squeeze': int((f(bb_u) - f(bb_l)) < np.std(prices[-20:]) * 0.1),
                'price_change': float(cp - prices[-2]) if len(prices) > 1 else 0,
                'price_change_pct': float((cp - prices[-2]) / prices[-2]) if len(prices) > 1 and prices[-2] != 0 else 0,
                'volatility': float(np.std(prices[-min(20, len(prices)):])),
                'current_price': float(cp),
                'valid': True
            }
        except Exception as e:
            logger.exception("Technical feature error:")
            return {'valid': False, 'error': str(e)}

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


# === Volume Features ===
class VolumeFeatureEngineer(FeatureEngineer):
    def __init__(self, window_size: int = 20):
        self.window_size = window_size

    def get_feature_names(self) -> List[str]:
        return [
            'volume_mean', 'volume_std', 'volume_ratio', 'volume_spike',
            'vwap', 'volume_trend', 'volume_momentum', 'current_volume'
        ]

    def calculate_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            if 'volume' not in data.columns or len(data['volume']) < 2:
                return {'valid': False, 'error': 'Insufficient volume data'}

            volume = data['volume']
            current = volume.iloc[-1]
            mean = volume.rolling(self.window_size).mean().iloc[-1]
            std = volume.rolling(self.window_size).std().iloc[-1]

            ratio = current / mean if mean != 0 else 1
            spike = int(ratio > 2.0)

            vwap = (
                (data['close'] * volume).sum() / volume.sum()
                if 'close' in data.columns and volume.sum() != 0
                else 0
            )

            recent = volume.iloc[-5:].mean()
            older = volume.iloc[-10:-5].mean() if len(volume) >= 10 else volume.iloc[:-5].mean()
            trend = int(np.sign(recent - older))
            momentum = (recent - older) / older if older != 0 else 0

            return {
                'volume_mean': float(mean),
                'volume_std': float(std),
                'volume_ratio': float(ratio),
                'volume_spike': spike,
                'vwap': float(vwap),
                'volume_trend': trend,
                'volume_momentum': float(momentum),
                'current_volume': float(current),
                'valid': True
            }
        except Exception as e:
            logger.exception("Volume feature error:")
            return {'valid': False, 'error': str(e)}


# === Manager ===
class FeatureEngineeringManager:
    def __init__(self):
        self.engineers: Dict[str, FeatureEngineer] = {}
        self.feature_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        self.cache_ttl = 60  # seconds

    def add_engineer(self, name: str, engineer: FeatureEngineer):
        self.engineers[name] = engineer
        logger.info(f"Added engineer: {name}")

    def calculate_all_features(self, data: pd.DataFrame, use_cache: bool = True) -> Dict[str, Any]:
        if data.empty:
            return {'valid': False, 'error': 'Empty data input'}

        cache_key = f"{len(data)}_{hash(data.iloc[-1].to_json())}"

        if use_cache and cache_key in self.feature_cache:
            if (datetime.now() - self.cache_timestamps[cache_key]).total_seconds() < self.cache_ttl:
                return self.feature_cache[cache_key]

        results = {
            'timestamp': datetime.now().isoformat(),
            'data_points': len(data),
            'engineers_used': list(self.engineers.keys())
        }

        for name, eng in self.engineers.items():
            results[name] = eng.calculate_features(data)

        if use_cache:
            self.feature_cache[cache_key] = results
            self.cache_timestamps[cache_key] = datetime.now()

        return results

    def get_unified_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        features = self.calculate_all_features(data)
        if not features:
            return pd.DataFrame()

        unified = {}
        for eng_name, feat_dict in features.items():
            if isinstance(feat_dict, dict) and feat_dict.get('valid', False):
                for k, v in feat_dict.items():
                    if k not in ['valid', 'error']:
                        unified[f"{eng_name}_{k}"] = v

        unified['timestamp'] = features.get('timestamp')
        unified['data_points'] = features.get('data_points')
        return pd.DataFrame([unified])


# === Factory ===
def create_default_feature_engineering_manager() -> FeatureEngineeringManager:
    mgr = FeatureEngineeringManager()
    mgr.add_engineer("statistical", StatisticalFeatureEngineer())
    mgr.add_engineer("technical", TechnicalFeatureEngineer())
    mgr.add_engineer("volume", VolumeFeatureEngineer())
    return mgr
