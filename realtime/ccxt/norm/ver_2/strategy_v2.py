
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import logging
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Trading signal types"""
    LONG = "LONG"
    SHORT = "SHORT"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    HOLD = "HOLD"
    NO_SIGNAL = "NO_SIGNAL"

class TradingSignal:
    """Represents a trading signal"""
    
    def __init__(self, signal_type: SignalType, strength: float = 1.0, 
                 reason: str = "", metadata: Dict = None):
        self.signal_type = signal_type
        self.strength = strength  # Signal strength from 0.0 to 1.0
        self.reason = reason
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    def __str__(self):
        return f"{self.signal_type.value} (strength: {self.strength:.2f}) - {self.reason}"

class BaseStrategy(ABC):
    """Enhanced abstract base class for trading strategies"""
    
    @abstractmethod
    def generate_signal(self, features: Dict[str, Any], 
                       current_position: Dict[str, Any] = None) -> TradingSignal:
        """Generate a trading signal based on features"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the strategy"""
        pass
    
    @abstractmethod
    def get_required_features(self) -> List[str]:
        """Get list of required features for this strategy"""
        pass

class MeanReversionV1Strategy(BaseStrategy):
    """
    Mean Reversion Strategy Version 1:
    - If Z-score > threshold → Short
    - If Z-score < -threshold → Long
    - Exit conditions:
      - If in short position → exit when Z-score < 0
      - If in long position → exit when Z-score > 0
    """
    
    def __init__(self, threshold: float = 1.3, min_signal_strength: float = 0.5):
        self.threshold = threshold
        self.min_signal_strength = min_signal_strength
        
    def get_name(self) -> str:
        return "MeanReversionV1"
    
    def get_required_features(self) -> List[str]:
        return ['statistical_zscore']
    
    def generate_signal(self, features: Dict[str, Any], 
                       current_position: Dict[str, Any] = None) -> TradingSignal:
        """Generate signal based on Mean Reversion V1 logic"""
        
        # Extract z-score from features
        statistical_features = features.get('statistical', {})
        if not statistical_features.get('valid', False):
            return TradingSignal(SignalType.NO_SIGNAL, 0.0, "No valid statistical features")
        
        zscore = statistical_features.get('zscore', 0.0)
        current_value = statistical_features.get('zscore_current_value', 0.0)
        
        # Get current position info
        current_side = None
        if current_position:
            current_side = current_position.get('side', None)
        
        # Exit conditions first
        if current_side == 'short' and zscore < 0:
            strength = min(1.0, abs(zscore) / self.threshold + 0.3)
            return TradingSignal(
                SignalType.CLOSE_SHORT,
                strength,
                f"MeanRev V1: Exit short, Z-score returned to negative: {zscore:.3f}",
                {'zscore': zscore, 'threshold': self.threshold}
            )
        
        if current_side == 'long' and zscore > 0:
            strength = min(1.0, abs(zscore) / self.threshold + 0.3)
            return TradingSignal(
                SignalType.CLOSE_LONG,
                strength,
                f"MeanRev V1: Exit long, Z-score returned to positive: {zscore:.3f}",
                {'zscore': zscore, 'threshold': self.threshold}
            )
        
        # Entry conditions
        if zscore >= self.threshold:
            # Mean reversion: high z-score suggests price will revert down
            strength = min(1.0, (zscore - self.threshold) / self.threshold + 0.5)
            if strength >= self.min_signal_strength:
                return TradingSignal(
                    SignalType.SHORT,
                    strength,
                    f"MeanRev V1: Z-score high, expect reversion down: {zscore:.3f} >= {self.threshold}",
                    {'zscore': zscore, 'threshold': self.threshold}
                )
        
        elif zscore <= -self.threshold:
            # Mean reversion: low z-score suggests price will revert up
            strength = min(1.0, (abs(zscore) - self.threshold) / self.threshold + 0.5)
            if strength >= self.min_signal_strength:
                return TradingSignal(
                    SignalType.LONG,
                    strength,
                    f"MeanRev V1: Z-score low, expect reversion up: {zscore:.3f} <= {-self.threshold}",
                    {'zscore': zscore, 'threshold': self.threshold}
                )
        
        # Hold or no signal
        if current_side:
            return TradingSignal(
                SignalType.HOLD,
                0.3,
                f"MeanRev V1: Holding position, Z-score: {zscore:.3f}",
                {'zscore': zscore}
            )
        else:
            return TradingSignal(
                SignalType.NO_SIGNAL,
                0.0,
                f"MeanRev V1: No signal, Z-score: {zscore:.3f}",
                {'zscore': zscore}
            )

class MeanReversionV2Strategy(BaseStrategy):
    """
    Mean Reversion Strategy Version 2:
    - If Z-score > threshold → Short
    - If Z-score < -threshold → Long
    - Exit the opposite position when crossing threshold again:
      - If long and Z-score > threshold → exit long
      - If short and Z-score < -threshold → exit short
    """
    
    def __init__(self, threshold: float = 1.3, min_signal_strength: float = 0.5):
        self.threshold = threshold
        self.min_signal_strength = min_signal_strength
        
    def get_name(self) -> str:
        return "MeanReversionV2"
    
    def get_required_features(self) -> List[str]:
        return ['statistical_zscore']
    
    def generate_signal(self, features: Dict[str, Any], 
                       current_position: Dict[str, Any] = None) -> TradingSignal:
        """Generate signal based on Mean Reversion V2 logic"""
        
        # Extract z-score from features
        statistical_features = features.get('statistical', {})
        if not statistical_features.get('valid', False):
            return TradingSignal(SignalType.NO_SIGNAL, 0.0, "No valid statistical features")
        
        zscore = statistical_features.get('zscore', 0.0)
        
        # Get current position info
        current_side = None
        if current_position:
            current_side = current_position.get('side', None)
        
        # Exit conditions - opposite threshold crossing
        if current_side == 'long' and zscore >= self.threshold:
            strength = min(1.0, (zscore - self.threshold) / self.threshold + 0.5)
            return TradingSignal(
                SignalType.CLOSE_LONG,
                strength,
                f"MeanRev V2: Exit long, Z-score crossed upper threshold: {zscore:.3f} >= {self.threshold}",
                {'zscore': zscore, 'threshold': self.threshold}
            )
        
        if current_side == 'short' and zscore <= -self.threshold:
            strength = min(1.0, (abs(zscore) - self.threshold) / self.threshold + 0.5)
            return TradingSignal(
                SignalType.CLOSE_SHORT,
                strength,
                f"MeanRev V2: Exit short, Z-score crossed lower threshold: {zscore:.3f} <= {-self.threshold}",
                {'zscore': zscore, 'threshold': self.threshold}
            )
        
        # Entry conditions
        if zscore >= self.threshold and current_side != 'short':
            strength = min(1.0, (zscore - self.threshold) / self.threshold + 0.5)
            if strength >= self.min_signal_strength:
                return TradingSignal(
                    SignalType.SHORT,
                    strength,
                    f"MeanRev V2: Z-score high, expect reversion down: {zscore:.3f} >= {self.threshold}",
                    {'zscore': zscore, 'threshold': self.threshold}
                )
        
        elif zscore <= -self.threshold and current_side != 'long':
            strength = min(1.0, (abs(zscore) - self.threshold) / self.threshold + 0.5)
            if strength >= self.min_signal_strength:
                return TradingSignal(
                    SignalType.LONG,
                    strength,
                    f"MeanRev V2: Z-score low, expect reversion up: {zscore:.3f} <= {-self.threshold}",
                    {'zscore': zscore, 'threshold': self.threshold}
                )
        
        # Hold or no signal
        if current_side:
            return TradingSignal(
                SignalType.HOLD,
                0.3,
                f"MeanRev V2: Holding position, Z-score: {zscore:.3f}",
                {'zscore': zscore}
            )
        else:
            return TradingSignal(
                SignalType.NO_SIGNAL,
                0.0,
                f"MeanRev V2: No signal, Z-score: {zscore:.3f}",
                {'zscore': zscore}
            )

class TrendFollowingV1Strategy(BaseStrategy):
    """
    Trend Following Strategy Version 1:
    - If Z-score > threshold → Open Long
    - If Z-score < -threshold → Close Long and Open Short
    - Reverse logic applies for Short positions
    """
    
    def __init__(self, threshold: float = 1.3, min_signal_strength: float = 0.5):
        self.threshold = threshold
        self.min_signal_strength = min_signal_strength
        
    def get_name(self) -> str:
        return "TrendFollowingV1"
    
    def get_required_features(self) -> List[str]:
        return ['statistical_zscore']
    
    def generate_signal(self, features: Dict[str, Any], 
                       current_position: Dict[str, Any] = None) -> TradingSignal:
        """Generate signal based on Trend Following V1 logic"""
        
        # Extract z-score from features
        statistical_features = features.get('statistical', {})
        if not statistical_features.get('valid', False):
            return TradingSignal(SignalType.NO_SIGNAL, 0.0, "No valid statistical features")
        
        zscore = statistical_features.get('zscore', 0.0)
        
        # Get current position info
        current_side = None
        if current_position:
            current_side = current_position.get('side', None)
        
        # Trend following logic
        if zscore >= self.threshold:
            # Positive trend - go long
            strength = min(1.0, (zscore - self.threshold) / self.threshold + 0.5)
            
            if current_side == 'short':
                # Close short and signal will trigger new long
                return TradingSignal(
                    SignalType.CLOSE_SHORT,
                    strength,
                    f"TrendFollow V1: Close short, trend turned positive: {zscore:.3f} >= {self.threshold}",
                    {'zscore': zscore, 'threshold': self.threshold, 'next_action': 'long'}
                )
            elif current_side != 'long' and strength >= self.min_signal_strength:
                return TradingSignal(
                    SignalType.LONG,
                    strength,
                    f"TrendFollow V1: Positive trend, go long: {zscore:.3f} >= {self.threshold}",
                    {'zscore': zscore, 'threshold': self.threshold}
                )
        
        elif zscore <= -self.threshold:
            # Negative trend - go short
            strength = min(1.0, (abs(zscore) - self.threshold) / self.threshold + 0.5)
            
            if current_side == 'long':
                # Close long and signal will trigger new short
                return TradingSignal(
                    SignalType.CLOSE_LONG,
                    strength,
                    f"TrendFollow V1: Close long, trend turned negative: {zscore:.3f} <= {-self.threshold}",
                    {'zscore': zscore, 'threshold': self.threshold, 'next_action': 'short'}
                )
            elif current_side != 'short' and strength >= self.min_signal_strength:
                return TradingSignal(
                    SignalType.SHORT,
                    strength,
                    f"TrendFollow V1: Negative trend, go short: {zscore:.3f} <= {-self.threshold}",
                    {'zscore': zscore, 'threshold': self.threshold}
                )
        
        # Hold or no signal
        if current_side:
            return TradingSignal(
                SignalType.HOLD,
                0.3,
                f"TrendFollow V1: Holding position, Z-score: {zscore:.3f}",
                {'zscore': zscore}
            )
        else:
            return TradingSignal(
                SignalType.NO_SIGNAL,
                0.0,
                f"TrendFollow V1: No trend signal, Z-score: {zscore:.3f}",
                {'zscore': zscore}
            )

class TrendFollowingV2Strategy(BaseStrategy):
    """
    Trend Following Strategy Version 2:
    - If Z-score > threshold → Open Long
    - If Z-score drops below 0 → Close Long
    - If Z-score < -threshold → Open Short
    - If Z-score rises above 0 → Close Short
    """
    
    def __init__(self, threshold: float = 1.3, min_signal_strength: float = 0.5):
        self.threshold = threshold
        self.min_signal_strength = min_signal_strength
        
    def get_name(self) -> str:
        return "TrendFollowingV2"
    
    def get_required_features(self) -> List[str]:
        return ['statistical_zscore']
    
    def generate_signal(self, features: Dict[str, Any], 
                       current_position: Dict[str, Any] = None) -> TradingSignal:
        """Generate signal based on Trend Following V2 logic"""
        
        # Extract z-score from features
        statistical_features = features.get('statistical', {})
        if not statistical_features.get('valid', False):
            return TradingSignal(SignalType.NO_SIGNAL, 0.0, "No valid statistical features")
        
        zscore = statistical_features.get('zscore', 0.0)
        
        # Get current position info
        current_side = None
        if current_position:
            current_side = current_position.get('side', None)
        
        # Exit conditions first
        if current_side == 'long' and zscore <= 0:
            strength = min(1.0, abs(zscore) / self.threshold + 0.3)
            return TradingSignal(
                SignalType.CLOSE_LONG,
                strength,
                f"TrendFollow V2: Close long, Z-score dropped below zero: {zscore:.3f} <= 0",
                {'zscore': zscore, 'threshold': self.threshold}
            )
        
        if current_side == 'short' and zscore >= 0:
            strength = min(1.0, abs(zscore) / self.threshold + 0.3)
            return TradingSignal(
                SignalType.CLOSE_SHORT,
                strength,
                f"TrendFollow V2: Close short, Z-score rose above zero: {zscore:.3f} >= 0",
                {'zscore': zscore, 'threshold': self.threshold}
            )
        
        # Entry conditions
        if zscore >= self.threshold and current_side != 'long':
            strength = min(1.0, (zscore - self.threshold) / self.threshold + 0.5)
            if strength >= self.min_signal_strength:
                return TradingSignal(
                    SignalType.LONG,
                    strength,
                    f"TrendFollow V2: Strong positive trend, go long: {zscore:.3f} >= {self.threshold}",
                    {'zscore': zscore, 'threshold': self.threshold}
                )
        
        elif zscore <= -self.threshold and current_side != 'short':
            strength = min(1.0, (abs(zscore) - self.threshold) / self.threshold + 0.5)
            if strength >= self.min_signal_strength:
                return TradingSignal(
                    SignalType.SHORT,
                    strength,
                    f"TrendFollow V2: Strong negative trend, go short: {zscore:.3f} <= {-self.threshold}",
                    {'zscore': zscore, 'threshold': self.threshold}
                )
        
        # Hold or no signal
        if current_side:
            return TradingSignal(
                SignalType.HOLD,
                0.3,
                f"TrendFollow V2: Holding position, Z-score: {zscore:.3f}",
                {'zscore': zscore}
            )
        else:
            return TradingSignal(
                SignalType.NO_SIGNAL,
                0.0,
                f"TrendFollow V2: No trend signal, Z-score: {zscore:.3f}",
                {'zscore': zscore}
            )

class MultiFeatureStrategy(BaseStrategy):
    """
    Advanced strategy that uses multiple features for decision making
    """
    
    def __init__(self, zscore_threshold: float = 1.3, rsi_overbought: float = 70, 
                 rsi_oversold: float = 30, min_signal_strength: float = 0.6):
        self.zscore_threshold = zscore_threshold
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.min_signal_strength = min_signal_strength
        
    def get_name(self) -> str:
        return "MultiFeature"
    
    def get_required_features(self) -> List[str]:
        return ['statistical_zscore', 'technical_rsi', 'technical_sma_cross']
    
    def generate_signal(self, features: Dict[str, Any], 
                       current_position: Dict[str, Any] = None) -> TradingSignal:
        """Generate signal based on multiple features"""
        
        # Extract features
        statistical_features = features.get('statistical', {})
        technical_features = features.get('technical', {})
        
        if not (statistical_features.get('valid', False) and technical_features.get('valid', False)):
            return TradingSignal(SignalType.NO_SIGNAL, 0.0, "Missing required features")
        
        zscore = statistical_features.get('zscore', 0.0)
        rsi = technical_features.get('rsi', 50.0)
        sma_cross = technical_features.get('sma_cross', 0)
        
        # Calculate composite signals
        signals = []
        reasons = []
        
        # Z-score signal (mean reversion)
        if zscore >= self.zscore_threshold:
            signals.append(('SHORT', min(1.0, zscore / self.zscore_threshold)))
            reasons.append(f"Z-score high: {zscore:.2f}")
        elif zscore <= -self.zscore_threshold:
            signals.append(('LONG', min(1.0, abs(zscore) / self.zscore_threshold)))
            reasons.append(f"Z-score low: {zscore:.2f}")
        
        # RSI signal
        if rsi >= self.rsi_overbought:
            signals.append(('SHORT', min(1.0, (rsi - 50) / 50)))
            reasons.append(f"RSI overbought: {rsi:.1f}")
        elif rsi <= self.rsi_oversold:
            signals.append(('LONG', min(1.0, (50 - rsi) / 50)))
            reasons.append(f"RSI oversold: {rsi:.1f}")
        
        # SMA cross signal (trend)
        if sma_cross == 1:
            signals.append(('LONG', 0.7))
            reasons.append("SMA bullish cross")
        elif sma_cross == -1:
            signals.append(('SHORT', 0.7))
            reasons.append("SMA bearish cross")
        
        if not signals:
            return TradingSignal(SignalType.NO_SIGNAL, 0.0, "No confluent signals")
        
        # Aggregate signals
        long_strength = sum(s[1] for s in signals if s[0] == 'LONG')
        short_strength = sum(s[1] for s in signals if s[0] == 'SHORT')
        
        # Determine final signal
        if long_strength > short_strength and long_strength >= self.min_signal_strength:
            return TradingSignal(
                SignalType.LONG,
                min(1.0, long_strength / len(signals)),
                f"MultiFeature LONG: {', '.join(reasons)}",
                {'zscore': zscore, 'rsi': rsi, 'sma_cross': sma_cross}
            )
        elif short_strength > long_strength and short_strength >= self.min_signal_strength:
            return TradingSignal(
                SignalType.SHORT,
                min(1.0, short_strength / len(signals)),
                f"MultiFeature SHORT: {', '.join(reasons)}",
                {'zscore': zscore, 'rsi': rsi, 'sma_cross': sma_cross}
            )
        else:
            return TradingSignal(
                SignalType.NO_SIGNAL,
                0.0,
                f"Weak/conflicting signals: {', '.join(reasons)}",
                {'zscore': zscore, 'rsi': rsi, 'sma_cross': sma_cross}
            )

class StrategySelector:
    """Strategy selection and management system"""
    
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.active_strategy: Optional[str] = None
        self.strategy_weights: Dict[str, float] = {}
        
    def add_strategy(self, name: str, strategy: BaseStrategy, weight: float = 1.0):
        """Add a strategy to the selector"""
        self.strategies[name] = strategy
        self.strategy_weights[name] = weight
        logger.info(f"Added strategy: {name} (weight: {weight})")
        
        # Set as active if it's the first strategy
        if self.active_strategy is None:
            self.active_strategy = name
    
    def set_active_strategy(self, name: str):
        """Set the active strategy"""
        if name in self.strategies:
            self.active_strategy = name
            logger.info(f"Active strategy set to: {name}")
        else:
            raise ValueError(f"Strategy {name} not found")
    
    def get_active_strategy(self) -> Optional[BaseStrategy]:
        """Get the currently active strategy"""
        if self.active_strategy and self.active_strategy in self.strategies:
            return self.strategies[self.active_strategy]
        return None
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names"""
        return list(self.strategies.keys())
    
    def generate_signal(self, features: Dict[str, Any], 
                       current_position: Dict[str, Any] = None) -> TradingSignal:
        """Generate signal from active strategy"""
        active_strategy = self.get_active_strategy()
        if not active_strategy:
            return TradingSignal(SignalType.NO_SIGNAL, 0.0, "No active strategy")
        
        # Check if required features are available
        required_features = active_strategy.get_required_features()
        missing_features = []
        
        for feature in required_features:
            if '_' in feature:
                engineer, feature_name = feature.split('_', 1)
                if engineer not in features or not features[engineer].get('valid', False):
                    missing_features.append(feature)
            else:
                if feature not in features:
                    missing_features.append(feature)
        
        if missing_features:
            return TradingSignal(
                SignalType.NO_SIGNAL, 
                0.0, 
                f"Missing required features: {', '.join(missing_features)}"
            )
        
        return active_strategy.generate_signal(features, current_position)
    
    def generate_all_signals(self, features: Dict[str, Any], 
                           current_position: Dict[str, Any] = None) -> Dict[str, TradingSignal]:
        """Generate signals from all strategies"""
        signals = {}
        
        for name, strategy in self.strategies.items():
            try:
                signal = strategy.generate_signal(features, current_position)
                signals[name] = signal
                logger.debug(f"Strategy {name}: {signal}")
            except Exception as e:
                logger.error(f"Error generating signal from strategy {name}: {e}")
                signals[name] = TradingSignal(SignalType.NO_SIGNAL, 0.0, f"Error: {str(e)}")
        
        return signals
    
    def get_strategy_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all strategies"""
        info = {}
        for name, strategy in self.strategies.items():
            info = {
                'name': strategy.get_name(),
                'required_features': strategy.get_required_features(),
                'weight': self.strategy_weights.get(name, 1.0),
                'active': name == self.active_strategy
            }
        return info

# Convenience function to create a strategy selector with all strategies
def create_strategy_selector(threshold: float = 1.3) -> StrategySelector:
    """Create a strategy selector with all available strategies"""
    selector = StrategySelector()
    
    # Add all strategy variants
    selector.add_strategy("mean_reversion_v1", MeanReversionV1Strategy(threshold=threshold))
    selector.add_strategy("mean_reversion_v2", MeanReversionV2Strategy(threshold=threshold))
    selector.add_strategy("trend_following_v1", TrendFollowingV1Strategy(threshold=threshold))
    selector.add_strategy("trend_following_v2", TrendFollowingV2Strategy(threshold=threshold))
    selector.add_strategy("multi_feature", MultiFeatureStrategy(zscore_threshold=threshold))
    
    # Set default active strategy
    selector.set_active_strategy("mean_reversion_v1")
    
    return selector