"""
Data preprocessing module for transforming raw market data.
Supports various preprocessing methods including z-score normalization.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class Preprocessor(ABC):
    """Abstract base class for data preprocessors"""
    
    @abstractmethod
    def process(self, data: Union[pd.DataFrame, List[Dict], np.ndarray]) -> Dict[str, Any]:
        """Process the input data and return processed results"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the preprocessor"""
        pass

class ZScorePreprocessor(Preprocessor):
    """Z-score normalization preprocessor"""
    
    def __init__(self, window_size: int = 20, min_periods: int = 5):
        self.window_size = window_size
        self.min_periods = min_periods
        
    def get_name(self) -> str:
        return "ZScore"
    
    def process(self, data: Union[pd.DataFrame, List[Dict], np.ndarray]) -> Dict[str, Any]:
        """
        Calculate z-score for the given data
        
        Args:
            data: Input data (DataFrame, list of dicts, or numpy array)
            
        Returns:
            Dictionary containing z-score and related statistics
        """
        try:
            # Convert input to pandas Series
            if isinstance(data, list):
                if len(data) == 0:
                    return {'zscore': 0.0, 'mean': 0.0, 'std': 0.0, 'value': 0.0, 'valid': False}
                
                # Extract values from list of dicts
                if isinstance(data[0], dict):
                    values = [item.get('open_interest', 0) for item in data if 'open_interest' in item]
                else:
                    values = data
                    
                series = pd.Series(values)
                
            elif isinstance(data, pd.DataFrame):
                # Use the first numeric column or 'open_interest' if available
                if 'open_interest' in data.columns:
                    series = data['open_interest']
                else:
                    # Use first numeric column
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        series = data[numeric_cols[0]]
                    else:
                        return {'zscore': 0.0, 'mean': 0.0, 'std': 0.0, 'value': 0.0, 'valid': False}
                        
            elif isinstance(data, np.ndarray):
                series = pd.Series(data)
            else:
                logger.error(f"Unsupported data type: {type(data)}")
                return {'zscore': 0.0, 'mean': 0.0, 'std': 0.0, 'value': 0.0, 'valid': False}
            
            # Check if we have enough data
            if len(series) < self.min_periods:
                logger.warning(f"Not enough data points: {len(series)} < {self.min_periods}")
                return {'zscore': 0.0, 'mean': 0.0, 'std': 0.0, 'value': 0.0, 'valid': False}
            
            # Calculate rolling statistics
            if len(series) >= self.window_size:
                # Use rolling window
                rolling_mean = series.rolling(window=self.window_size, min_periods=self.min_periods).mean()
                rolling_std = series.rolling(window=self.window_size, min_periods=self.min_periods).std()
                
                current_mean = rolling_mean.iloc[-1]
                current_std = rolling_std.iloc[-1]
            else:
                # Use all available data
                current_mean = series.mean()
                current_std = series.std()
            
            current_value = series.iloc[-1]
            
            # Calculate z-score
            if current_std == 0 or np.isnan(current_std):
                zscore = 0.0
            else:
                zscore = (current_value - current_mean) / current_std
            
            result = {
                'zscore': float(zscore),
                'mean': float(current_mean),
                'std': float(current_std),
                'value': float(current_value),
                'valid': True,
                'data_points': len(series),
                'window_size': self.window_size
            }
            
            logger.debug(f"Z-score calculation: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating z-score: {e}")
            return {'zscore': 0.0, 'mean': 0.0, 'std': 0.0, 'value': 0.0, 'valid': False}

class PercentilePreprocessor(Preprocessor):
    """Percentile-based normalization preprocessor"""
    
    def __init__(self, window_size: int = 20, percentiles: List[float] = [25, 50, 75]):
        self.window_size = window_size
        self.percentiles = percentiles
        
    def get_name(self) -> str:
        return "Percentile"
    
    def process(self, data: Union[pd.DataFrame, List[Dict], np.ndarray]) -> Dict[str, Any]:
        """Calculate percentile-based statistics"""
        try:
            # Convert input to pandas Series (similar to ZScorePreprocessor)
            if isinstance(data, list):
                if len(data) == 0:
                    return {'percentile_rank': 0.0, 'valid': False}
                
                if isinstance(data[0], dict):
                    values = [item.get('open_interest', 0) for item in data if 'open_interest' in item]
                else:
                    values = data
                    
                series = pd.Series(values)
                
            elif isinstance(data, pd.DataFrame):
                if 'open_interest' in data.columns:
                    series = data['open_interest']
                else:
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        series = data[numeric_cols[0]]
                    else:
                        return {'percentile_rank': 0.0, 'valid': False}
                        
            elif isinstance(data, np.ndarray):
                series = pd.Series(data)
            else:
                return {'percentile_rank': 0.0, 'valid': False}
            
            if len(series) < 2:
                return {'percentile_rank': 0.0, 'valid': False}
            
            # Calculate percentile rank of current value
            current_value = series.iloc[-1]
            percentile_rank = stats.percentileofscore(series, current_value)
            
            # Calculate percentile values
            percentile_values = {}
            for p in self.percentiles:
                percentile_values[f'p{p}'] = np.percentile(series, p)
            
            result = {
                'percentile_rank': percentile_rank,
                'current_value': float(current_value),
                'valid': True,
                **percentile_values
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating percentiles: {e}")
            return {'percentile_rank': 0.0, 'valid': False}

class MovingAveragePreprocessor(Preprocessor):
    """Moving average preprocessor"""
    
    def __init__(self, window_sizes: List[int] = [5, 10, 20]):
        self.window_sizes = window_sizes
        
    def get_name(self) -> str:
        return "MovingAverage"
    
    def process(self, data: Union[pd.DataFrame, List[Dict], np.ndarray]) -> Dict[str, Any]:
        """Calculate moving averages"""
        try:
            # Convert input to pandas Series
            if isinstance(data, list):
                if len(data) == 0:
                    return {'valid': False}
                
                if isinstance(data[0], dict):
                    values = [item.get('open_interest', 0) for item in data if 'open_interest' in item]
                else:
                    values = data
                    
                series = pd.Series(values)
                
            elif isinstance(data, pd.DataFrame):
                if 'open_interest' in data.columns:
                    series = data['open_interest']
                else:
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        series = data[numeric_cols[0]]
                    else:
                        return {'valid': False}
                        
            elif isinstance(data, np.ndarray):
                series = pd.Series(data)
            else:
                return {'valid': False}
            
            # Calculate moving averages
            result = {'valid': True, 'current_value': float(series.iloc[-1])}
            
            for window in self.window_sizes:
                if len(series) >= window:
                    ma = series.rolling(window=window).mean().iloc[-1]
                    result[f'ma_{window}'] = float(ma)
                    
                    # Calculate deviation from moving average
                    deviation = (series.iloc[-1] - ma) / ma if ma != 0 else 0
                    result[f'ma_{window}_deviation'] = float(deviation)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")
            return {'valid': False}

class PreprocessorManager:
    """Manages multiple preprocessors"""
    
    def __init__(self):
        self.preprocessors: Dict[str, Preprocessor] = {}
        
    def add_preprocessor(self, name: str, preprocessor: Preprocessor):
        """Add a preprocessor"""
        self.preprocessors[name] = preprocessor
        logger.info(f"Added preprocessor: {name}")
        
    def process_data(self, data: Union[pd.DataFrame, List[Dict], np.ndarray], 
                    preprocessor_names: List[str] = None) -> Dict[str, Any]:
        """Process data with specified preprocessors"""
        if preprocessor_names is None:
            preprocessor_names = list(self.preprocessors.keys())
            
        results = {}
        for name in preprocessor_names:
            if name in self.preprocessors:
                try:
                    result = self.preprocessors[name].process(data)
                    results[name] = result
                except Exception as e:
                    logger.error(f"Error processing with {name}: {e}")
                    results[name] = {'valid': False, 'error': str(e)}
            else:
                logger.warning(f"Preprocessor {name} not found")
                
        return results
    
    def get_zscore(self, data: Union[pd.DataFrame, List[Dict], np.ndarray]) -> float:
        """Convenience method to get z-score"""
        if 'zscore' in self.preprocessors:
            result = self.preprocessors['zscore'].process(data)
            return result.get('zscore', 0.0)
        return 0.0
