import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Tuple
from collections import deque
from scipy import stats
from datetime import datetime
import sys
import os

# Add config path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.settings import get_config


class ZScoreStrategy:
    """
    Trading strategy based on Z-score of combined Open Interest and Long/Short Ratio
    """
    
    def __init__(self, rolling_window: int = None, z_threshold: float = None, config_manager=None):
        """
        Initialize the Z-score strategy
        
        Args:
            rolling_window: Number of periods for rolling mean calculation (uses config if None)
            z_threshold: Z-score threshold for trading signals (uses config if None)
            config_manager: Configuration manager instance
        """
        self.logger = logging.getLogger(__name__)
        
        # Get configuration
        self.config = config_manager or get_config()
        
        # Use config values if not provided
        self.rolling_window = rolling_window or self.config.trading.rolling_window
        self.z_threshold = z_threshold or self.config.trading.z_score_threshold
        
        # Data storage for rolling calculations
        self.indicator_history = deque(maxlen=self.rolling_window * 2)  # Keep extra for safety
        self.z_scores = deque(maxlen=100)  # Keep history of z-scores
        self.signals = deque(maxlen=100)  # Keep history of signals
        
        # Last data tracking to handle stale data
        self.last_indicator_value = None
        self.last_indicator_timestamp = None
        self.stale_data_count = 0
        
        # Strategy state
        self.is_initialized = False
        
        self.logger.info(f"Initialized Z-Score Strategy: window={self.rolling_window}, threshold=±{self.z_threshold}")
    
    def initialize_with_historical_data(self, historical_data: List[Tuple[float, datetime]]) -> bool:
        """
        Initialize strategy with historical data
        
        Args:
            historical_data: List of (indicator_value, timestamp) tuples
            
        Returns:
            True if initialization successful, False otherwise
        """
        if not historical_data:
            self.logger.error("No historical data provided for initialization")
            return False
        
        if len(historical_data) < self.rolling_window:
            self.logger.error(f"Insufficient historical data: {len(historical_data)} < {self.rolling_window}")
            return False
        
        self.logger.info(f"Initializing strategy with {len(historical_data)} historical data points")
        
        # Clear existing data
        self.indicator_history.clear()
        self.z_scores.clear()
        self.signals.clear()
        
        # Add historical data
        for indicator_value, timestamp in historical_data:
            self.indicator_history.append({
                'value': indicator_value,
                'timestamp': timestamp
            })
        
        # Update tracking
        if historical_data:
            self.last_indicator_value = historical_data[-1][0]
            self.last_indicator_timestamp = historical_data[-1][1]
        
        self.is_initialized = True
        self.stale_data_count = 0
        
        self.logger.info(f"Strategy initialized successfully. Ready for trading signals.")
        return True
    
    def add_indicator_value(self, indicator_value: float, timestamp: datetime = None) -> bool:
        """
        Add a new indicator value to the history
        
        Args:
            indicator_value: Combined indicator value (OI * LS_Ratio)
            timestamp: Timestamp of the data point
            
        Returns:
            True if value was added, False if it's stale data
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Check for stale data
        if self.last_indicator_value is not None:
            if abs(indicator_value - self.last_indicator_value) < 1e-10:  # Essentially the same value
                self.stale_data_count += 1
                self.logger.warning(f"Stale data detected: {indicator_value} "
                                  f"(count: {self.stale_data_count})")
                
                if self.stale_data_count >= self.config.data.max_stale_data_tolerance:
                    self.logger.error(f"Too many consecutive stale data points: {self.stale_data_count}")
                    return False
            else:
                self.stale_data_count = 0  # Reset stale data count
        
        # Check for timestamp freshness if provided
        if timestamp and self.last_indicator_timestamp:
            if timestamp <= self.last_indicator_timestamp:
                self.logger.warning(f"Data timestamp is not newer than previous: {timestamp} <= {self.last_indicator_timestamp}")
                # Don't return False here as this might be expected in some cases
        
        # Add the new value
        self.indicator_history.append({
            'value': indicator_value,
            'timestamp': timestamp
        })
        
        self.last_indicator_value = indicator_value
        self.last_indicator_timestamp = timestamp
        
        self.logger.info(f"Added indicator value: {indicator_value:.6f} at {timestamp}")
        return True
    
    def calculate_z_score(self) -> Optional[float]:
        """
        Calculate the Z-score of the latest indicator value
        
        Returns:
            Z-score or None if insufficient data
        """
        if len(self.indicator_history) < self.rolling_window:
            self.logger.warning(f"Insufficient data for Z-score calculation: "
                              f"{len(self.indicator_history)}/{self.rolling_window}")
            return None
        
        try:
            # Get the last rolling_window values
            recent_values = [item['value'] for item in list(self.indicator_history)[-self.rolling_window:]]
            
            # Calculate rolling mean and std
            rolling_mean = np.mean(recent_values[:-1])  # Mean of all but the latest value
            rolling_std = np.std(recent_values[:-1], ddof=1)  # Standard deviation
            
            if rolling_std == 0:
                self.logger.warning("Rolling standard deviation is zero, cannot calculate Z-score")
                return None
            
            # Calculate Z-score of the latest value
            latest_value = recent_values[-1]
            z_score = (latest_value - rolling_mean) / rolling_std
            
            self.z_scores.append({
                'z_score': z_score,
                'timestamp': self.last_indicator_timestamp,
                'indicator_value': latest_value,
                'rolling_mean': rolling_mean,
                'rolling_std': rolling_std
            })
            
            self.logger.info(f"Z-Score calculated: {z_score:.4f} "
                           f"(value: {latest_value:.6f}, mean: {rolling_mean:.6f}, std: {rolling_std:.6f})")
            
            return z_score
            
        except Exception as e:
            self.logger.error(f"Error calculating Z-score: {e}")
            return None
    
    def generate_signal(self) -> Optional[str]:
        """
        Generate trading signal based on Z-score
        
        Returns:
            'LONG' for long signal, 'SHORT' for short signal, 'HOLD' for no signal, None if error
        """
        z_score = self.calculate_z_score()
        
        if z_score is None:
            return None
        
        signal = 'HOLD'
        
        if z_score > self.z_threshold:
            signal = 'SHORT'
            self.logger.info(f"SHORT signal generated: Z-score {z_score:.4f} > {self.z_threshold}")
        elif z_score < -self.z_threshold:
            signal = 'LONG'
            self.logger.info(f"LONG signal generated: Z-score {z_score:.4f} < -{self.z_threshold}")
        else:
            self.logger.info(f"HOLD signal: Z-score {z_score:.4f} within threshold ±{self.z_threshold}")
        
        # Store signal history
        self.signals.append({
            'signal': signal,
            'z_score': z_score,
            'timestamp': self.last_indicator_timestamp
        })
        
        return signal
    
    def get_latest_z_score(self) -> Optional[float]:
        """
        Get the latest Z-score
        
        Returns:
            Latest Z-score or None if not available
        """
        if self.z_scores:
            return self.z_scores[-1]['z_score']
        return None
    
    def get_signal_history(self, n: int = 10) -> List[dict]:
        """
        Get recent signal history
        
        Args:
            n: Number of recent signals to return
            
        Returns:
            List of recent signals
        """
        return list(self.signals)[-n:] if self.signals else []
    
    def get_z_score_history(self, n: int = 10) -> List[dict]:
        """
        Get recent Z-score history
        
        Args:
            n: Number of recent Z-scores to return
            
        Returns:
            List of recent Z-scores
        """
        return list(self.z_scores)[-n:] if self.z_scores else []
    
    def get_indicator_history(self, n: int = 20) -> List[dict]:
        """
        Get recent indicator history
        
        Args:
            n: Number of recent indicators to return
            
        Returns:
            List of recent indicator values
        """
        return list(self.indicator_history)[-n:] if self.indicator_history else []
    
    def is_ready(self) -> bool:
        """
        Check if strategy has enough data to generate signals
        
        Returns:
            True if ready, False otherwise
        """
        return (self.is_initialized and 
                len(self.indicator_history) >= self.rolling_window)
    
    def get_strategy_stats(self) -> dict:
        """
        Get strategy statistics
        
        Returns:
            Dictionary with strategy statistics
        """
        stats = {
            'data_points': len(self.indicator_history),
            'z_scores_calculated': len(self.z_scores),
            'signals_generated': len(self.signals),
            'is_ready': self.is_ready(),
            'rolling_window': self.rolling_window,
            'z_threshold': self.z_threshold,
            'stale_data_count': self.stale_data_count,
            'last_indicator_value': self.last_indicator_value,
            'last_timestamp': self.last_indicator_timestamp.isoformat() if self.last_indicator_timestamp else None
        }
        
        if self.z_scores:
            latest_z = self.z_scores[-1]
            stats.update({
                'latest_z_score': latest_z['z_score'],
                'latest_rolling_mean': latest_z['rolling_mean'],
                'latest_rolling_std': latest_z['rolling_std']
            })
        
        if self.signals:
            latest_signal = self.signals[-1]
            stats['latest_signal'] = latest_signal['signal']
        
        return stats
    
    def reset_strategy(self):
        """
        Reset the strategy state (use with caution)
        """
        self.logger.warning("Resetting strategy state")
        self.indicator_history.clear()
        self.z_scores.clear()
        self.signals.clear()
        self.last_indicator_value = None
        self.last_indicator_timestamp = None
        self.stale_data_count = 0


def test_z_score_strategy():
    """Test function for the Z-score strategy"""
    logging.basicConfig(level=logging.INFO)
    
    strategy = ZScoreStrategy(rolling_window=5, z_threshold=1.5)  # Smaller values for testing
    
    # Simulate some indicator data
    test_data = [100, 102, 98, 105, 110, 95, 120, 85, 130, 80, 140, 75]
    
    for i, value in enumerate(test_data):
        timestamp = datetime.now().replace(second=i)
        strategy.add_indicator_value(value, timestamp)
        
        if strategy.is_ready():
            signal = strategy.generate_signal()
            print(f"Step {i}: Value={value}, Signal={signal}, Z-Score={strategy.get_latest_z_score():.4f}")
    
    # Print strategy stats
    print("\nStrategy Stats:")
    stats = strategy.get_strategy_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    test_z_score_strategy()