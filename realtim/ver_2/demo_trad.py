"""
Enhanced Trading System Demo
Showcases feature engineering, multiple strategies, and strategy switching capabilities.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import our enhanced system components
from feature_engineering import create_default_feature_engineering_manager
from strategy_v2 import create_strategy_selector, SignalType

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_market_data(scenario: str = "normal") -> pd.DataFrame:
    """Create sample market data for different scenarios"""
    
    # Base parameters
    base_price = 50000
    base_volume = 1000000
    base_oi = 1500000
    periods = 50
    
    # Create time index
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=periods)
    timestamps = pd.date_range(start=start_time, end=end_time, periods=periods)
    
    if scenario == "normal":
        # Normal market with small fluctuations
        price_changes = np.random.normal(0, 0.01, periods)
        prices = base_price * np.cumprod(1 + price_changes)
        volumes = base_volume * (1 + np.random.normal(0, 0.2, periods))
        open_interest = base_oi * (1 + np.random.normal(0, 0.1, periods))
        
    elif scenario == "trending_up":
        # Strong uptrend with increasing open interest
        trend = np.linspace(0, 0.15, periods)  # 15% uptrend
        noise = np.random.normal(0, 0.02, periods)
        price_changes = trend + noise
        prices = base_price * np.cumprod(1 + price_changes)
        volumes = base_volume * (1 + np.random.normal(0.3, 0.3, periods))  # Higher volume
        open_interest = base_oi * np.linspace(1, 1.4, periods)  # Increasing OI
        
    elif scenario == "trending_down":
        # Strong downtrend with decreasing open interest
        trend = np.linspace(0, -0.12, periods)  # 12% downtrend
        noise = np.random.normal(0, 0.02, periods)
        price_changes = trend + noise
        prices = base_price * np.cumprod(1 + price_changes)
        volumes = base_volume * (1 + np.random.normal(0.2, 0.25, periods))
        open_interest = base_oi * np.linspace(1, 0.7, periods)  # Decreasing OI
        
    elif scenario == "volatile":
        # High volatility market
        price_changes = np.random.normal(0, 0.04, periods)  # 4x normal volatility
        prices = base_price * np.cumprod(1 + price_changes)
        volumes = base_volume * (1 + np.random.normal(0, 0.5, periods))
        open_interest = base_oi * (1 + np.random.normal(0, 0.3, periods))
        
    elif scenario == "oi_spike":
        # Normal market with sudden open interest spike
        price_changes = np.random.normal(0, 0.01, periods)
        prices = base_price * np.cumprod(1 + price_changes)
        volumes = base_volume * (1 + np.random.normal(0, 0.2, periods))
        open_interest = base_oi * (1 + np.random.normal(0, 0.1, periods))
        # Add spike in last 5 periods
        open_interest[-5:] *= 1.8
        
    elif scenario == "oi_drop":
        # Normal market with sudden open interest drop
        price_changes = np.random.normal(0, 0.01, periods)
        prices = base_price * np.cumprod(1 + price_changes)
        volumes = base_volume * (1 + np.random.normal(0, 0.2, periods))
        open_interest = base_oi * (1 + np.random.normal(0, 0.1, periods))
        # Add drop in last 5 periods
        open_interest[-5:] *= 0.4
        
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    # Ensure positive values
    volumes = np.abs(volumes)
    open_interest = np.abs(open_interest)
    
    # Create OHLC data from prices
    highs = prices * (1 + np.abs(np.random.normal(0, 0.005, periods)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.005, periods)))
    opens = np.roll(prices, 1)
    opens[0] = prices[0]
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes,
        'open_interest': open_interest
    })
    
    data.set_index('timestamp', inplace=True)
    return data

async def demo_feature_engineering():
    """Demonstrate feature engineering capabilities"""
    logger.info("🔬 Feature Engineering Demo")
    logger.info("=" * 50)
    
    # Create feature engineering manager
    feature_manager = create_default_feature_engineering_manager()
    
    # Show available features
    available_features = feature_manager.get_available_features()
    logger.info("📊 Available Feature Engineers:")
    for engineer_name, features in available_features.items():
        logger.info(f"   {engineer_name}: {len(features)} features")
        logger.info(f"      {', '.join(features[:5])}{'...' if len(features) > 5 else ''}")
    
    # Test different market scenarios
    scenarios = ["normal", "trending_up", "trending_down", "volatile", "oi_spike", "oi_drop"]
    
    for scenario in scenarios:
        logger.info(f"\n📈 Scenario: {scenario.replace('_', ' ').title()}")
        logger.info("-" * 30)
        
        # Generate sample data
        data = create_sample_market_data(scenario)
        
        # Calculate features
        features = feature_manager.calculate_all_features(data)
        
        # Display key features
        if features.get('statistical', {}).get('valid', False):
            stats = features['statistical']
            logger.info(f"   Z-Score: {stats['zscore']:.3f}")
            logger.info(f"   Current OI: {stats['zscore_current_value']:.0f}")
            logger.info(f"   Rolling Mean: {stats['zscore_mean']:.0f}")
            logger.info(f"   Percentile Rank: {stats['percentile_rank']:.1f}%")
        
        if features.get('technical', {}).get('valid', False):
            tech = features['technical']
            logger.info(f"   RSI: {tech['rsi']:.1f}")
            logger.info(f"   SMA Cross: {tech['sma_cross']}")
            logger.info(f"   Volatility: {tech['volatility']:.0f}")
        
        await asyncio.sleep(0.1)

async def demo_strategy_comparison():
    """Demonstrate different strategy behaviors"""
    logger.info("\n🎯 Strategy Comparison Demo")
    logger.info("=" * 50)
    
    # Create strategy selector
    strategy_selector = create_strategy_selector(threshold=1.3)
    
    # Get all strategies
    strategies = strategy_selector.get_available_strategies()
    logger.info(f"📋 Available Strategies: {len(strategies)}")
    for name in strategies:
        logger.info(f"   • {name}")
    
    # Test scenarios with different z-scores
    test_scenarios = [
        ("High Z-Score (2.0)", 2.0),
        ("Medium High Z-Score (1.5)", 1.5),
        ("Normal Z-Score (0.5)", 0.5),
        ("Medium Low Z-Score (-1.5)", -1.5),
        ("Very Low Z-Score (-2.2)", -2.2),
    ]
    
    for scenario_name, zscore in test_scenarios:
        logger.info(f"\n📊 {scenario_name}")
        logger.info("-" * 40)
        
        # Create mock features with the test z-score
        mock_features = {
            'statistical': {
                'valid': True,
                'zscore': zscore,
                'zscore_current_value': 1000000 + zscore * 100000,
                'zscore_mean': 1000000,
                'zscore_std': 100000
            },
            'technical': {
                'valid': True,
                'rsi': 50 + zscore * 10,  # RSI correlates with z-score
                'sma_cross': 1 if zscore > 0 else -1,
                'current_price': 50000 + zscore * 1000
            }
        }
        
        # Test each strategy
        for strategy_name in strategies:
            strategy_selector.set_active_strategy(strategy_name)
            signal = strategy_selector.generate_signal(mock_features)
            
            # Format signal display
            signal_emoji = {
                SignalType.LONG: "🟢",
                SignalType.SHORT: "🔴",
                SignalType.HOLD: "🟡",
                SignalType.NO_SIGNAL: "⚪"
            }
            
            emoji = signal_emoji.get(signal.signal_type, "❓")
            logger.info(f"   {strategy_name:20} → {emoji} {signal.signal_type.value:12} "
                       f"(strength: {signal.strength:.2f})")
        
        await asyncio.sleep(0.1)

async def demo_position_management():
    """Demonstrate position management across strategies"""
    logger.info("\n💼 Position Management Demo")
    logger.info("=" * 50)
    
    strategy_selector = create_strategy_selector(threshold=1.3)
    
    # Simulate position states
    position_scenarios = [
        ("No Position", None),
        ("Long Position", {'side': 'long', 'size': 0.1, 'entry_price': 50000}),
        ("Short Position", {'side': 'short', 'size': 0.1, 'entry_price': 50000})
    ]
    
    # Test z-score values
    zscore_tests = [2.0, 0.5, -2.0]
    
    for pos_name, position in position_scenarios:
        logger.info(f"\n📍 Current Position: {pos_name}")
        
        for zscore in zscore_tests:
            logger.info(f"\n   Z-Score: {zscore:.1f}")
            logger.info("   " + "-" * 30)
            
            mock_features = {
                'statistical': {
                    'valid': True,
                    'zscore': zscore,
                    'zscore_current_value': 1000000 + zscore * 100000,
                }
            }
            
            # Test Mean Reversion strategies
            for strategy_name in ["mean_reversion_v1", "mean_reversion_v2"]:
                strategy_selector.set_active_strategy(strategy_name)
                signal = strategy_selector.generate_signal(mock_features, position)
                
                logger.info(f"   {strategy_name:18} → {signal.signal_type.value:12} "
                           f"({signal.strength:.2f}) - {signal.reason[:50]}...")
        
        await asyncio.sleep(0.1)

async def demo_multi_feature_strategy():
    """Demonstrate the multi-feature strategy"""
    logger.info("\n🧠 Multi-Feature Strategy Demo")
    logger.info("=" * 50)
    
    strategy_selector = create_strategy_selector()
    strategy_selector.set_active_strategy("multi_feature")
    
    # Test different feature combinations
    feature_combinations = [
        {
            "name": "All Bullish",
            "features": {
                'statistical': {'valid': True, 'zscore': -1.8},  # Low (bullish for mean reversion)
                'technical': {'valid': True, 'rsi': 25, 'sma_cross': 1}  # Oversold + bullish cross
            }
        },
        {
            "name": "All Bearish",
            "features": {
                'statistical': {'valid': True, 'zscore': 1.9},  # High (bearish for mean reversion)
                'technical': {'valid': True, 'rsi': 80, 'sma_cross': -1}  # Overbought + bearish cross
            }
        },
        {
            "name": "Conflicting Signals",
            "features": {
                'statistical': {'valid': True, 'zscore': 1.5},  # High (bearish)
                'technical': {'valid': True, 'rsi': 30, 'sma_cross': 1}  # Oversold + bullish cross
            }
        },
        {
            "name": "Weak Signals",
            "features": {
                'statistical': {'valid': True, 'zscore': 0.8},  # Neutral
                'technical': {'valid': True, 'rsi': 55, 'sma_cross': 0}  # Neutral
            }
        }
    ]
    
    for combo in feature_combinations:
        logger.info(f"\n📊 {combo['name']}")
        logger.info("-" * 30)
        
        features = combo['features']
        stats = features['statistical']
        tech = features['technical']
        
        logger.info(f"   Z-Score: {stats['zscore']:.2f}")
        logger.info(f"   RSI: {tech['rsi']:.1f}")
        logger.info(f"   SMA Cross: {tech['sma_cross']}")
        
        signal = strategy_selector.generate_signal(features)
        
        signal_emoji = {
            SignalType.LONG: "🟢 LONG",
            SignalType.SHORT: "🔴 SHORT",
            SignalType.NO_SIGNAL: "⚪ NO SIGNAL"
        }
        
        emoji = signal_emoji.get(signal.signal_type, "❓")
        logger.info(f"   Result: {emoji} (strength: {signal.strength:.2f})")
        logger.info(f"   Reason: {signal.reason}")
        
        await asyncio.sleep(0.1)

async def demo_real_time_simulation():
    """Simulate real-time trading with strategy switching"""
    logger.info("\n⏰ Real-Time Trading Simulation")
    logger.info("=" * 50)
    
    feature_manager = create_default_feature_engineering_manager()
    strategy_selector = create_strategy_selector()
    
    # Simulate 10 time periods
    logger.info("🔄 Simulating 10 trading cycles with strategy switching...")
    
    strategies_to_test = ["mean_reversion_v1", "trend_following_v1", "multi_feature"]
    current_strategy_idx = 0
    
    for cycle in range(1, 11):
        logger.info(f"\n⏱️  Cycle #{cycle}")
        logger.info("-" * 20)
        
        # Switch strategy every 3 cycles
        if cycle % 3 == 1:
            current_strategy = strategies_to_test[current_strategy_idx % len(strategies_to_test)]
            strategy_selector.set_active_strategy(current_strategy)
            logger.info(f"📋 Strategy: {current_strategy}")
            current_strategy_idx += 1
        
        # Generate random market scenario
        scenarios = ["normal", "trending_up", "volatile", "oi_spike"]
        scenario = np.random.choice(scenarios)
        
        # Create market data
        data = create_sample_market_data(scenario)
        
        # Calculate features
        features = feature_manager.calculate_all_features(data)
        
        # Generate signal
        signal = strategy_selector.generate_signal(features)
        
        # Display results
        if features.get('statistical', {}).get('valid', False):
            zscore = features['statistical']['zscore']
            logger.info(f"📊 Market: {scenario}, Z-Score: {zscore:.2f}")
        
        signal_emoji = {
            SignalType.LONG: "🟢",
            SignalType.SHORT: "🔴",
            SignalType.HOLD: "🟡",
            SignalType.NO_SIGNAL: "⚪"
        }
        
        emoji = signal_emoji.get(signal.signal_type, "❓")
        logger.info(f"🎯 Signal: {emoji} {signal.signal_type.value} (strength: {signal.strength:.2f})")
        
        await asyncio.sleep(0.5)  # Simulate time delay

async def main():
    """Run all demos"""
    logger.info("🚀 Enhanced Trading System Demo Suite")
    logger.info("=" * 60)
    
    try:
        # Run all demo components
        await demo_feature_engineering()
        await demo_strategy_comparison()
        await demo_position_management()
        await demo_multi_feature_strategy()
        await demo_real_time_simulation()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ All demos completed successfully!")
        logger.info("\n📋 Summary of Capabilities Demonstrated:")
        logger.info("   • Feature Engineering: Statistical, Technical, Volume features")
        logger.info("   • Strategy Types: Mean Reversion V1/V2, Trend Following V1/V2, Multi-Feature")
        logger.info("   • Strategy Switching: Dynamic strategy selection")
        logger.info("   • Position Management: Different behaviors based on current positions")
        logger.info("   • Real-time Simulation: Complete trading cycle simulation")
        logger.info("\n🎯 Your enhanced trading system is ready for live trading!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())