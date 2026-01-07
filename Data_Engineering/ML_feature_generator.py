import os
import random
import operator
import math
import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools, gp
from sklearn.model_selection import train_test_split
import xgboost as xgb
import scipy.stats  # For rank

# Step 1: Load Data (assumes output from auto_feature_gen.py or similar)
def load_data(file_path='merged_by.csv'):
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df = df.ffill().bfill()  # Impute missing
    return df

# Step 2: Feature Importance Selection with XGBoost
def select_important_features(df, target='forward_return', top_n=100, importance_threshold=0.01):
    feature_cols = [col for col in df.columns if col not in ['datetime', target]]
    X = df[feature_cols].fillna(0)
    y = df[target].fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    importances = dict(zip(feature_cols, model.feature_importances_))
    selected = [col for col, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True) if imp > importance_threshold][:top_n]
    print(f"Selected {len(selected)} important features.")
    return selected, importances

# Step 3: Genetic Programming for Feature Evolution
# Define primitive set (operators and functions for evolution)
def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def protected_log(x):
    return np.log1p(abs(x))

def diff(x):
    return np.diff(x, prepend=0)  # Time-series diff

# Add more primitives inspired by 101 alphas
def protected_sqrt(x):
    return np.sqrt(np.abs(x))

def ts_delay(x):
    return np.roll(x, 1)  # Simple delay by 1

def ts_delta(x):
    return np.diff(x, prepend=x[0])

pset = gp.PrimitiveSet("MAIN", 2)  # Arity 2 for two base features
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protected_div, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(np.abs, 1)
pset.addPrimitive(protected_log, 1)
pset.addPrimitive(protected_sqrt, 1)
pset.addPrimitive(ts_delay, 1)
pset.addPrimitive(ts_delta, 1)
pset.addPrimitive(np.maximum, 2)
pset.addPrimitive(np.minimum, 2)
pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1))

# DEAP Setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Remove select_important_features, add compute_feature_scores_and_probs
def compute_feature_scores_and_probs(df, close_dir='../Data/close'):
    feature_cols = [col for col in df.columns if col not in ['datetime']]
    scores = {}
    for col in feature_cols:
        scores[col] = compute_sum_ic(df, col, close_dir)
    
    # Sort by scores descending
    sorted_cols = sorted(scores, key=scores.get, reverse=True)
    ranked_scores = [scores[col] for col in sorted_cols]
    
    # Softmax on scores
    max_score = np.max(ranked_scores)
    exp_scores = np.exp(ranked_scores - max_score)
    probs = exp_scores / exp_scores.sum()
    
    prob_dict = dict(zip(sorted_cols, probs))
    return prob_dict, sorted_cols, scores

def compute_sum_ic(df, feature_col, close_dir):
    ic_sum = 0.0
    close_files = [os.path.join(close_dir, f) for f in os.listdir(close_dir) if f.endswith('.csv')]
    for close_path in close_files:
        close_df = pd.read_csv(close_path)
        close_df['datetime'] = pd.to_datetime(close_df['datetime'])
        merged = pd.merge(df[['datetime', feature_col]], close_df[['datetime', 'Close']], on='datetime', how='inner')
        merged = merged.sort_values('datetime')
        merged['forward_return'] = merged['Close'].pct_change().shift(-1)
        merged.dropna(inplace=True)
        if not merged.empty:
            ic = merged[feature_col].corr(merged['forward_return'])
            ic_sum += abs(ic)  # Sum absolute IC for fitness
    return ic_sum

# Modify evaluate_individual for arity 2 and probabilistic selection
def evaluate_individual(individual, df, prob_dict, close_dir):
    func = toolbox.compile(expr=individual)
    base_features = list(prob_dict.keys())
    probs = list(prob_dict.values())
    if sum(probs) == 0:
        return 0,
    col1, col2 = np.random.choice(base_features, size=2, p=probs / sum(probs), replace=True)
    try:
        new_feature = pd.Series(func(df[col1].values, df[col2].values), index=df.index)
    except Exception as e:
        return 0,
    ic = compute_sum_ic(df.assign(new_feat=new_feature), 'new_feat', close_dir)
    return ic,

toolbox.register("evaluate", evaluate_individual, df=None, prob_dict=None, close_dir=None)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Change to use rank selection: after evaluation, assign fitness based on rank
# We'll modify evolve_features for that

def evolve_features(df, prob_dict, generations=10, pop_size=100, elite_size=5):
    toolbox.unregister("evaluate")
    toolbox.register("evaluate", evaluate_individual, df=df, prob_dict=prob_dict, close_dir='../Data/close')
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(10)
    
    for gen in range(generations):
        # Evaluate
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        # For rank selection: sort by fitness, assign new fitness as rank
        sorted_pop = sorted(pop, key=lambda ind: ind.fitness.values[0], reverse=True)
        for rank, ind in enumerate(sorted_pop):
            ind.fitness.values = (len(sorted_pop) - rank,)  # Higher rank, higher fitness
        
        # Select next generation using roulette on ranked fitness
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Elitism: add top elite from previous pop
        elites = tools.selBest(pop, elite_size)
        offspring.extend(elites)
        
        pop[:] = offspring
        
        hof.update(pop)
    
    evolved = [str(ind) for ind in hof]
    print("Top evolved feature expressions:", evolved)
    return evolved

# Step 4: Generate and Evaluate Evolved Features
def generate_evolved_df(df, evolved_expressions, base_features):
    new_df = df.copy()
    for i, expr_str in enumerate(evolved_expressions):
        func = toolbox.compile(expr=gp.PrimitiveTree.from_string(expr_str, pset))
        col1 = random.choice(base_features)
        col2 = random.choice(base_features)
        new_df[f'evolved_{i}'] = pd.Series(func(new_df[col1].values, new_df[col2].values), index=new_df.index)
    return new_df

# Modify calculate_ic to use sum_ic
def calculate_sum_ic_scores(df, feature_cols, close_dir='../Data/close'):
    ic_results = {}
    for col in feature_cols:
        ic_results[col] = compute_sum_ic(df, col, close_dir)
    return ic_results

# Main Pipeline
if __name__ == "__main__":
    df = load_data()  # Step 1
    
    # Compute probs
    prob_dict, important_features, scores = compute_feature_scores_and_probs(df)
    
    # Evolve new features
    evolved_expressions = evolve_features(df, prob_dict)  # Step 3
    evolved_df = generate_evolved_df(df, evolved_expressions, important_features)  # Step 4
    
    # Evaluate evolved features with summed IC
    evolved_cols = [col for col in evolved_df.columns if 'evolved_' in col]
    ic_results = calculate_sum_ic_scores(evolved_df, evolved_cols)
    
    # Select top evolved by IC
    top_evolved = sorted(ic_results, key=ic_results.get, reverse=True)[:20]
    print("Top Evolved Alphas by summed IC:", top_evolved)
    
    # Save
    evolved_df[top_evolved + ['datetime']].to_csv('evolved_features.csv', index=False)  # No forward_return
    print("Evolved features saved to 'evolved_features.csv'") 