#!/usr/bin/env python3
"""
Sophisticated Feature Engineering Example Usage
Demonstrates advanced financial analytics capabilities
"""

import pandas as pd
import numpy as np
from feature_engineering_template import *

def run_sophisticated_example():
    """Example of using sophisticated feature engineering capabilities"""
    
    print("🚀 SOPHISTICATED FEATURE ENGINEERING EXAMPLE")
    print("=" * 60)
    
    # 1. Generate sample financial data for demonstration
    print("\n1. 📊 Generating sample financial data...")
    sample_data = generate_sample_data()
    print(f"Generated {len(sample_data)} data points")
    
    # 2. Apply individual sophisticated feature functions
    print("\n2. 🧠 Testing individual sophisticated features...")
    
    # Test Fourier features
    print("  🌊 Testing Fourier analysis...")
    fourier_feats = fourier_features(sample_data['price'], window=50)
    print(f"     Generated {len(fourier_feats)} Fourier features")
    
    # Test Wavelet features
    print("  📊 Testing Wavelet analysis...")
    wavelet_feats = wavelet_features(sample_data['price'], levels=3)
    print(f"     Generated {len(wavelet_feats)} Wavelet features")
    
    # Test Sentiment features
    print("  💭 Testing Sentiment analysis...")
    sentiment_feats = sentiment_features(sample_data)
    print(f"     Generated {len(sentiment_feats)} Sentiment features")
    
    # Test Regime detection
    print("  🎯 Testing Regime detection...")
    regime_feats = regime_detection_features(sample_data, price_col='price')
    print(f"     Generated {len(regime_feats)} Regime features")
    
    # Test Portfolio optimization
    print("  📈 Testing Portfolio optimization...")
    feature_cols = ['volume', 'volatility']
    returns_cols = ['future_return_2']
    portfolio_feats = portfolio_optimization_features(sample_data, feature_cols, returns_cols, window=50)
    print(f"     Generated {len(portfolio_feats)} Portfolio features")
    
    # 3. Run comprehensive pipeline
    print("\n3. 🎯 Running comprehensive sophisticated pipeline...")
    
    # Add future returns to sample data
    sample_data['future_return_2'] = sample_data['price'].pct_change(2).shift(-2)
    sample_data['future_return_8'] = sample_data['price'].pct_change(8).shift(-8)
    sample_data['future_return_24'] = sample_data['price'].pct_change(24).shift(-24)
    
    # Generate all sophisticated features
    all_features, metadata = generate_all_features(sample_data)
    print(f"     Generated {len(all_features)} total features")
    
    # Calculate ICs
    sample_data_with_features = sample_data.copy()
    for name, series in all_features.items():
        sample_data_with_features[name] = series
    
    sample_data_with_features, updated_metadata = calculate_ics_for_all_features(
        sample_data_with_features, all_features, metadata
    )
    
    # Perform regime-aware selection
    regime_adaptive_feats, regime_metadata = adaptive_feature_selection_with_regimes(
        sample_data_with_features, updated_metadata, regime_threshold=0.01
    )
    
    print(f"     Selected {len(regime_adaptive_feats)} regime-adaptive features")
    
    # 4. Display results
    print("\n4. 📊 Results Summary")
    display_results(updated_metadata, regime_metadata)
    
    # 5. Save example outputs
    print("\n5. 💾 Saving example outputs...")
    save_example_outputs(sample_data_with_features, updated_metadata, regime_metadata)
    
    print("\n✅ Sophisticated feature engineering example completed!")
    print("🎯 Check the generated files for detailed results.")

def generate_sample_data(n_points=1000):
    """Generate sample financial time series data"""
    np.random.seed(42)  # For reproducible results
    
    # Create datetime index
    dates = pd.date_range(start='2020-01-01', periods=n_points, freq='H')
    
    # Generate synthetic price with trends and volatility
    returns = np.random.normal(0, 0.02, n_points)
    
    # Add some cyclical patterns
    time_trend = np.sin(np.arange(n_points) * 2 * np.pi / 24) * 0.01  # Daily cycle
    weekly_trend = np.sin(np.arange(n_points) * 2 * np.pi / (24*7)) * 0.005  # Weekly cycle
    
    returns += time_trend + weekly_trend
    
    # Generate price from returns
    price = 100 * np.exp(np.cumsum(returns))
    
    # Generate volume with correlation to volatility
    volatility = pd.Series(returns).rolling(24).std().fillna(0.01)
    volume = 1000000 * (1 + volatility * 5 + np.random.normal(0, 0.1, n_points))
    volume = np.abs(volume)
    
    # Generate some proxy sentiment indicators
    premium_index = np.random.normal(0, 1, n_points) + returns * 50
    flow_indicator = np.random.normal(0, 100000, n_points) + returns * 1000000
    funding_rate = np.random.normal(0, 0.001, n_points) + returns * 0.1
    
    # Create DataFrame
    data = pd.DataFrame({
        'datetime': dates,
        'price': price,
        'volume': volume,
        'volatility': volatility,
        'premium_index': premium_index,
        'flow_indicator': flow_indicator,
        'funding_rate': funding_rate
    })
    
    return data

def display_results(metadata, regime_metadata):
    """Display analysis results"""
    
    # Feature type breakdown
    feature_types = {}
    for meta in metadata.values():
        ftype = meta.get('feature_type', 'basic')
        feature_types[ftype] = feature_types.get(ftype, 0) + 1
    
    print("   📊 Feature Type Distribution:")
    for ftype, count in sorted(feature_types.items()):
        print(f"      {ftype.capitalize()}: {count}")
    
    # Top features by IC
    sorted_features = sorted(metadata.items(), key=lambda x: abs(x[1]['max_ic']), reverse=True)
    
    print("\n   🏆 Top 5 Features by IC:")
    for i, (name, meta) in enumerate(sorted_features[:5], 1):
        print(f"      {i}. {name}")
        print(f"         IC: {meta['max_ic']:.6f}")
        print(f"         Type: {meta.get('feature_type', 'basic')}")
    
    # Regime-adaptive results
    if regime_metadata:
        print(f"\n   🎯 Regime-Adaptive Selection:")
        print(f"      Total regime-adaptive features: {len(regime_metadata)}")
        
        sorted_regime = sorted(regime_metadata.items(), 
                             key=lambda x: x[1]['weighted_ic'], reverse=True)
        
        print("      Top 3 Regime-Adaptive Features:")
        for i, (name, meta) in enumerate(sorted_regime[:3], 1):
            print(f"         {i}. {name}")
            print(f"            Weighted IC: {meta['weighted_ic']:.6f}")
            print(f"            Regimes: {', '.join(meta['significant_regimes'])}")

def save_example_outputs(data, metadata, regime_metadata):
    """Save example outputs to files"""
    
    # Save sample data with features
    data.to_csv('example_sophisticated_features.csv', index=False)
    
    # Save feature metadata
    import json
    
    # Convert metadata to JSON-serializable format
    json_metadata = {}
    for name, meta in metadata.items():
        json_metadata[name] = {
            'base_column': meta['base_column'],
            'transformation': meta['transformation'],
            'feature_type': meta.get('feature_type', 'basic'),
            'ic_values': meta['ic_values'],
            'max_ic': meta['max_ic'],
            'best_shift': meta['best_shift']
        }
    
    with open('example_feature_metadata.json', 'w') as f:
        json.dump(json_metadata, f, indent=2)
    
    # Save regime metadata if available
    if regime_metadata:
        json_regime_metadata = {}
        for name, meta in regime_metadata.items():
            json_regime_metadata[name] = {
                'base_column': meta['base_column'],
                'transformation': meta['transformation'],
                'feature_type': meta['feature_type'],
                'weighted_ic': meta['weighted_ic'],
                'significant_regimes': meta['significant_regimes'],
                'original_max_ic': meta['original_max_ic']
            }
        
        with open('example_regime_metadata.json', 'w') as f:
            json.dump(json_regime_metadata, f, indent=2)
    
    print("   ✅ Saved:")
    print("      - example_sophisticated_features.csv")
    print("      - example_feature_metadata.json")
    if regime_metadata:
        print("      - example_regime_metadata.json")

def run_mini_genetic_programming_demo():
    """Demonstrate genetic programming with small example"""
    print("\n🧬 MINI GENETIC PROGRAMMING DEMO")
    print("=" * 40)
    
    # Create simple test data
    test_data = generate_sample_data(n_points=200)
    
    # Add future returns
    test_data['future_return_2'] = test_data['price'].pct_change(2).shift(-2)
    test_data['future_return_8'] = test_data['price'].pct_change(8).shift(-8)
    
    # Create some basic features for GP
    test_features = {
        'volume_ma': test_data['volume'].rolling(12).mean(),
        'volatility_zscore': (test_data['volatility'] - test_data['volatility'].rolling(24).mean()) / test_data['volatility'].rolling(24).std()
    }
    
    # Run mini GP evolution
    print("Running 3 generations of genetic programming...")
    
    gp_evolution = EnhancedGeneticFeatureEvolution(
        short_midterm_features=test_features,
        mid_longterm_features=test_features,
        df=test_data,
        generations=3,
        population_size=10,
        mutation_rate=0.3,
        ic_threshold=0.01
    )
    
    generation_data = gp_evolution.evolve_with_tracking()
    
    print(f"✅ GP Demo completed with {len(generation_data)} generations")
    print("   Check generation_X_elite_* files for results")

if __name__ == "__main__":
    # Run the main sophisticated example
    run_sophisticated_example()
    
    # Optionally run GP demo (uncomment to enable)
    # run_mini_genetic_programming_demo()
    
    print("\n🎊 All examples completed successfully!")
    print("📚 Check the README for detailed documentation.")



