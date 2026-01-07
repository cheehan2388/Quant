import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
import random
from scipy.stats import pearsonr, spearmanr
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

def calculate_ic(feature_series, target_series):
    """Calculate Information Coefficient (Pearson correlation)"""
    # Remove NaN values
    valid_mask = ~(pd.isna(feature_series) | pd.isna(target_series))
    if valid_mask.sum() < 10:  # Need at least 10 valid observations
        return 0.0
    
    feature_clean = feature_series[valid_mask]
    target_clean = target_series[valid_mask]
    
    # Calculate correlation
    correlation = feature_clean.corr(target_clean,method = "spearman")
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
                #df[col] = np.log1p(df[col])
                transformed_series = transform_func(df[col])
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
            'ic_values': meta['ic_values'],
            'max_ic': meta['max_ic'],
            'best_shift': meta['best_shift'],
            'abs_max_ic': abs(meta['max_ic'])
        }
    
    return report

# =============================================================================
# GENETIC PROGRAMMING FUNCTIONS
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

class GeneticFeatureEvolution:
    """Genetic Programming for Feature Evolution"""
    
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
        self.generation_stats = []
        
    def create_individual(self):
        """Create a single genetic programming individual"""
        # Choose 2 features from pools (can be from same or different pools)
        all_features = {**self.short_midterm_features, **self.mid_longterm_features}
        feature_names = list(all_features.keys())
        
        if len(feature_names) < 2:
            return None
            
        feature1_name = random.choice(feature_names)
        feature2_name = random.choice(feature_names)
        
        # Choose 2 operators
        operator1 = random.choice(self.operators)
        operator2 = random.choice(self.operators)
        
        individual = {
            'feature1': feature1_name,
            'feature2': feature2_name,
            'operator1': operator1,
            'operator2': operator2,
            'fitness': 0.0,
            'feature_data': None
        }
        
        return individual
    
    def initialize_population(self):
        """Initialize the genetic programming population"""
        print(f"Initializing population of {self.population_size} individuals...")
        self.population = []
        
        for _ in range(self.population_size):
            individual = self.create_individual()
            if individual is not None:
                self.population.append(individual)
        
        print(f"Created {len(self.population)} individuals")
    
    def evaluate_individual(self, individual):
        """Evaluate fitness (IC) of an individual"""
        try:
            all_features = {**self.short_midterm_features, **self.mid_longterm_features}
            
            # Get feature data
            feature1_data = all_features[individual['feature1']].values
            feature2_data = all_features[individual['feature2']].values
            
            # Apply first operator
            intermediate = GeneticOperators.apply_operator(
                feature1_data, feature2_data, individual['operator1']
            )
            
            # Apply second operator (use feature1 again for simplicity)
            final_feature = GeneticOperators.apply_operator(
                intermediate, feature1_data, individual['operator2']
            )
            
            # Store the evolved feature data
            individual['feature_data'] = pd.Series(final_feature, index=self.df.index)
            
            # Calculate IC against all future returns
            ic_values = []
            for shift in [-2, -8, -24]:
                future_return_col = f'future_return_{abs(shift)}'
                if future_return_col in self.df.columns:
                    ic = calculate_ic(individual['feature_data'], self.df[future_return_col])
                    ic_values.append(abs(ic))
            
            # Use maximum absolute IC as fitness
            fitness = max(ic_values) if ic_values else 0.0
            individual['fitness'] = fitness
            
            return fitness
            
        except Exception as e:
            print(f"Error evaluating individual: {e}")
            individual['fitness'] = 0.0
            return 0.0
    
    def evaluate_population(self):
        """Evaluate fitness for entire population"""
        self.fitness_scores = []
        valid_individuals = []
        
        for individual in self.population:
            fitness = self.evaluate_individual(individual)
            if fitness > self.ic_threshold:  # Only keep individuals with IC > threshold
                self.fitness_scores.append(fitness)
                valid_individuals.append(individual)
        
        # Update population to only include valid individuals
        self.population = valid_individuals
        print(f"Population after IC filtering: {len(self.population)} individuals (IC > {self.ic_threshold})")
        
        return len(self.population) > 0
    
    def selection(self, tournament_size=3):
        """Tournament selection"""
        if len(self.population) < 2:
            return self.population.copy()
        
        selected = []
        for _ in range(len(self.population)):
            tournament = random.sample(self.population, min(tournament_size, len(self.population)))
            winner = max(tournament, key=lambda x: x['fitness'])
            selected.append(winner.copy())
        
        return selected
    
    def crossover(self, parent1, parent2):
        """Create offspring by combining parents"""
        child = parent1.copy()
        
        # Randomly inherit features and operators from parents
        if random.random() < 0.5:
            child['feature1'] = parent2['feature1']
        if random.random() < 0.5:
            child['feature2'] = parent2['feature2']
        if random.random() < 0.5:
            child['operator1'] = parent2['operator1']
        if random.random() < 0.5:
            child['operator2'] = parent2['operator2']
        
        child['fitness'] = 0.0
        child['feature_data'] = None
        
        return child
    
    def mutate(self, individual):
        """Apply mutation to an individual"""
        if random.random() < self.mutation_rate:
            # Choose what to mutate
            mutation_type = random.choice(['feature1', 'feature2', 'operator1', 'operator2', 'transform'])
            
            if mutation_type == 'feature1' or mutation_type == 'feature2':
                # Mutate feature selection
                all_features = {**self.short_midterm_features, **self.mid_longterm_features}
                feature_names = list(all_features.keys())
                individual[mutation_type] = random.choice(feature_names)
                
            elif mutation_type == 'operator1' or mutation_type == 'operator2':
                # Mutate operator
                individual[mutation_type] = random.choice(self.operators)
                
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
    
    def evolve(self):
        """Run the genetic programming evolution"""
        print(f"\n🧬 Starting Genetic Programming Evolution for {self.generations} generations")
        print("="*70)
        
        # Initialize population
        self.initialize_population()
        
        if len(self.population) == 0:
            print("❌ Failed to create initial population")
            return []
        
        best_individuals = []
        
        for generation in range(self.generations):
            print(f"\n🔬 Generation {generation + 1}/{self.generations}")
            
            # Evaluate population
            has_valid = self.evaluate_population()
            
            if not has_valid:
                print(f"⚠️  No individuals with IC > {self.ic_threshold} in generation {generation + 1}")
                # Create new random population
                self.initialize_population()
                continue
            
            # Track statistics
            if self.fitness_scores:
                avg_fitness = np.mean(self.fitness_scores)
                max_fitness = np.max(self.fitness_scores)
                min_fitness = np.min(self.fitness_scores)
                
                stats = {
                    'generation': generation + 1,
                    'population_size': len(self.population),
                    'avg_fitness': avg_fitness,
                    'max_fitness': max_fitness,
                    'min_fitness': min_fitness
                }
                self.generation_stats.append(stats)
                
                print(f"📊 Pop: {len(self.population)}, Avg IC: {avg_fitness:.6f}, Max IC: {max_fitness:.6f}")
                
                # Keep track of best individual from this generation
                best_individual = max(self.population, key=lambda x: x['fitness'])
                best_individuals.append({
                    'generation': generation + 1,
                    'individual': best_individual.copy(),
                    'fitness': best_individual['fitness']
                })
            
            # Selection
            selected = self.selection()
            
            # Create next generation
            next_generation = []
            
            # Perform 5 crossover operations as requested
            for _ in range(5):
                if len(selected) >= 2:
                    parent1 = random.choice(selected)
                    parent2 = random.choice(selected)
                    
                    child1 = self.crossover(parent1, parent2)
                    child2 = self.crossover(parent2, parent1)
                    
                    # Apply mutation
                    self.mutate(child1)
                    self.mutate(child2)
                    
                    next_generation.extend([child1, child2])
            
            # Add some of the best individuals (elitism)
            elite_size = min(10, len(selected))
            elite = sorted(selected, key=lambda x: x['fitness'], reverse=True)[:elite_size]
            next_generation.extend(elite)
            
            # Fill remaining slots with new random individuals
            while len(next_generation) < self.population_size:
                new_individual = self.create_individual()
                if new_individual is not None:
                    next_generation.append(new_individual)
            
            self.population = next_generation[:self.population_size]
        
        print(f"\n✅ Evolution completed!")
        return best_individuals
    
    def get_evolved_features(self, best_individuals, top_n=20):
        """Extract top evolved features from evolution results"""
        if not best_individuals:
            return {}, {}
        
        # Sort by fitness and take top N
        sorted_individuals = sorted(best_individuals, key=lambda x: x['fitness'], reverse=True)
        top_individuals = sorted_individuals[:top_n]
        
        evolved_features = {}
        evolved_metadata = {}
        
        for i, item in enumerate(top_individuals):
            individual = item['individual']
            feature_name = f"evolved_gp_{i+1}"
            
            if individual['feature_data'] is not None:
                evolved_features[feature_name] = individual['feature_data']
                
                evolved_metadata[feature_name] = {
                    'generation': item['generation'],
                    'fitness': item['fitness'],
                    'feature1': individual['feature1'],
                    'feature2': individual['feature2'],
                    'operator1': individual['operator1'],
                    'operator2': individual['operator2'],
                    'expression': f"({individual['feature1']} {individual['operator1']} {individual['feature2']}) {individual['operator2']} {individual['feature1']}"
                }
        
        return evolved_features, evolved_metadata

def run_genetic_programming(df, short_midterm_features, mid_longterm_features, 
                           generations=12, population_size=50, mutation_rate=0.3, ic_threshold=0.05):
    """Run genetic programming evolution on feature groups"""
    
    if len(short_midterm_features) == 0 and len(mid_longterm_features) == 0:
        print("⚠️  No features available for genetic programming")
        return {}, {}, []
    
    print(f"\n🧬 GENETIC PROGRAMMING FEATURE EVOLUTION")
    print(f"Short-midterm pool: {len(short_midterm_features)} features")
    print(f"Mid-longterm pool: {len(mid_longterm_features)} features")
    print(f"Parameters: {generations} generations, {population_size} population, {mutation_rate:.1%} mutation rate")
    print(f"IC threshold: {ic_threshold}")
    
    # Initialize genetic programming
    gp_evolution = GeneticFeatureEvolution(
        short_midterm_features=short_midterm_features,
        mid_longterm_features=mid_longterm_features,
        df=df,
        generations=generations,
        population_size=population_size,
        mutation_rate=mutation_rate,
        ic_threshold=ic_threshold
    )
    
    # Run evolution
    best_individuals = gp_evolution.evolve()
    
    # Extract evolved features
    evolved_features, evolved_metadata = gp_evolution.get_evolved_features(best_individuals)
    
    # Generate evolution report
    evolution_report = {
        'summary': {
            'total_generations': generations,
            'population_size': population_size,
            'mutation_rate': mutation_rate,
            'ic_threshold': ic_threshold,
            'short_midterm_pool_size': len(short_midterm_features),
            'mid_longterm_pool_size': len(mid_longterm_features),
            'evolved_features_count': len(evolved_features),
            'best_fitness': max([item['fitness'] for item in best_individuals]) if best_individuals else 0,
            'avg_fitness_final': np.mean([item['fitness'] for item in best_individuals[-10:]]) if len(best_individuals) >= 10 else 0
        },
        'generation_stats': gp_evolution.generation_stats,
        'evolved_features': evolved_metadata,
        'operators_used': gp_evolution.operators
    }
    
    # Save evolution report
    with open('genetic_programming_report.json', 'w') as f:
        json.dump(evolution_report, f, indent=2)
    print(f"Evolution report saved to genetic_programming_report.json")
    
    # Save evolved features to CSV
    if evolved_features:
        evolved_df = pd.DataFrame({'datetime': df['datetime']})
        for feature_name, feature_series in evolved_features.items():
            evolved_df[feature_name] = feature_series
        
        evolved_df.to_csv('genetic_programming_features.csv', index=False)
        print(f"Evolved features saved to genetic_programming_features.csv")
    
    return evolved_features, evolved_metadata, best_individuals

def save_transformed_features(df, output_path='transformed_features_V_S_2.csv'):
    """Save all transformed features to CSV"""
    # Select columns to save (exclude temporary future return columns if needed)
    df.to_csv(output_path, index=False)
    print(f"Transformed features saved to {output_path}")

def main():
    """Main pipeline execution"""
    print("=== Feature Engineering Pipeline ===")
    
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
    
    # Step 8: Genetic Programming Evolution
    print("\n8. Genetic Programming Evolution...")
    gp_evolved_features, gp_evolved_metadata, gp_best_individuals = run_genetic_programming(
        df=df,
        short_midterm_features=short_midterm_features,
        mid_longterm_features=mid_longterm_features,
        generations=12,
        population_size=50,
        mutation_rate=0.3,
        ic_threshold=0.05
    )
    
    # Add evolved features to dataframe
    for feature_name, feature_series in gp_evolved_features.items():
        df[feature_name] = feature_series
    
    # Step 9: Save all results
    print("\n9. Saving all results...")
    save_transformed_features(df)
    
    # Display comprehensive summary
    print("\n" + "="*60)
    print("COMPREHENSIVE FEATURE ANALYSIS SUMMARY")
    print("="*60)
    
    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"Total features generated: {len(feature_metadata)}")
    print(f"Elite features (IC > 0.05): {len(elite_features)}")
    print(f"Short-midterm features (IC > 0.01): {len(short_midterm_features)}")
    print(f"Mid-longterm features (IC > 0.01): {len(mid_longterm_features)}")
    print(f"🧬 GP Evolved features: {len(gp_evolved_features)}")
    
    # Top overall features
    print(f"\n🏆 TOP 10 FEATURES BY ABSOLUTE IC:")
    top_features = sorted(
        feature_metadata.items(),
        key=lambda x: abs(x[1]['max_ic']),
        reverse=True
    )[:10]
    
    for i, (feature_name, metadata) in enumerate(top_features, 1):
        print(f"{i:2d}. {feature_name}")
        print(f"    Base Column: {metadata['base_column']}")
        print(f"    Transformation: {metadata['transformation']}")
        print(f"    Max IC: {metadata['max_ic']:.6f}")
        print(f"    Best Shift: {metadata['best_shift']}")
        print()
    
    # Elite features summary
    if len(elite_features) > 0:
        print(f"💎 TOP 5 ELITE FEATURES:")
        elite_top = sorted(
            elite_metadata.items(),
            key=lambda x: abs(x[1]['max_ic']),
            reverse=True
        )[:5]
        
        for i, (feature_name, metadata) in enumerate(elite_top, 1):
            print(f"{i}. {feature_name} (IC: {metadata['max_ic']:.6f})")
    
    # Group statistics
    print(f"\n📈 GROUP ANALYSIS:")
    
    if len(short_midterm_features) > 0:
        sm_avg_ic = np.mean([abs(meta['max_ic']) for meta in short_midterm_metadata.values()])
        print(f"Short-Midterm Group: {len(short_midterm_features)} features, Avg |IC|: {sm_avg_ic:.6f}")
        
        # Show shift distribution for short-midterm
        shift_counts = {'shift_-2': 0, 'shift_-8': 0}
        for meta in short_midterm_metadata.values():
            if abs(meta['ic_values'].get('shift_-2', 0)) > 0.01:
                shift_counts['shift_-2'] += 1
            if abs(meta['ic_values'].get('shift_-8', 0)) > 0.01:
                shift_counts['shift_-8'] += 1
        print(f"  - Features with shift(-2) IC > 0.01: {shift_counts['shift_-2']}")
        print(f"  - Features with shift(-8) IC > 0.01: {shift_counts['shift_-8']}")
    
    if len(mid_longterm_features) > 0:
        ml_avg_ic = np.mean([abs(meta['max_ic']) for meta in mid_longterm_metadata.values()])
        print(f"Mid-Longterm Group: {len(mid_longterm_features)} features, Avg |IC|: {ml_avg_ic:.6f}")
        
        # Show shift distribution for mid-longterm
        shift_counts = {'shift_-8': 0, 'shift_-24': 0}
        for meta in mid_longterm_metadata.values():
            if abs(meta['ic_values'].get('shift_-8', 0)) > 0.01:
                shift_counts['shift_-8'] += 1
            if abs(meta['ic_values'].get('shift_-24', 0)) > 0.01:
                shift_counts['shift_-24'] += 1
        print(f"  - Features with shift(-8) IC > 0.01: {shift_counts['shift_-8']}")
        print(f"  - Features with shift(-24) IC > 0.01: {shift_counts['shift_-24']}")
    
    # Files generated
    print(f"\n📁 FILES GENERATED:")
    print("Main outputs:")
    print("  - feature_engineering_report_N_S.json: Complete feature analysis")
    print("  - transformed_features_V_S.csv: All transformed features")
    
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
    
    if len(gp_evolved_features) > 0:
        print("Genetic Programming:")
        print("  - genetic_programming_features.csv: GP evolved features data")
        print("  - genetic_programming_report.json: GP evolution analysis")
    
    # Show top GP evolved features if available
    if len(gp_evolved_features) > 0:
        print(f"\n🧬 TOP 3 GP EVOLVED FEATURES:")
        gp_top = sorted(
            gp_evolved_metadata.items(),
            key=lambda x: x[1]['fitness'],
            reverse=True
        )[:3]
        
        for i, (feature_name, metadata) in enumerate(gp_top, 1):
            print(f"{i}. {feature_name}")
            print(f"   Expression: {metadata['expression']}")
            print(f"   Fitness (IC): {metadata['fitness']:.6f}")
            print(f"   Generation: {metadata['generation']}")
    
    print("\n✅ Pipeline completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()