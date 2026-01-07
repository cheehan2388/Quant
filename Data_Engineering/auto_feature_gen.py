import os
import numpy as np
import pandas as pd
from scipy.stats import skew
import featuretools as ft
from featuretools.primitives import make_agg_primitive, make_trans_primitive
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances

# Custom primitives if needed (e.g., z-score, but Featuretools has many built-in)
def zscore(x):
    return (x - x.mean()) / x.std()

ZScore = make_agg_primitive(
    function=zscore,
    input_types=[ft.variable_types.Numeric],
    return_type=ft.variable_types.Numeric
)

# Step 1: Load and Merge Data
def load_and_merge_data(data_dir='../Data/All_data/', target_col='close'):  # Adjust dir as needed
    dfs = []
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(data_dir, file))
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                dfs.append(df)
    
    # Merge on datetime (outer join to keep all)
    merged_df = pd.concat(dfs, axis=1, join='outer')
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]  # Remove duplicate columns
    merged_df = merged_df.sort_values('datetime').reset_index(drop=True)
    merged_df = merged_df.ffill().bfill()  # Simple imputation
    
    # Compute forward returns (target)
    merged_df['forward_return'] = merged_df[target_col].pct_change().shift(-1)
    merged_df.dropna(subset=['forward_return'], inplace=True)
    
    return merged_df

# Step 2: Preprocessing (Log transform skewed features)
def preprocess_data(df, skew_threshold=1.5):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if abs(skew(df[col].dropna())) > skew_threshold and df[col].min() >= 0:
            df[col] = np.log1p(df[col])
            print(f"Applied log1p to {col} due to skewness.")
    return df

# Step 3: Define Groups (based on column name patterns)
def define_groups(df):
    groups = {
        'exchange_flows': [col for col in df.columns if any(k in col.lower() for k in ['inflow', 'outflow', 'netflow', 'reserve', 'whale_ratio'])],
        'utxo_distributions': [col for col in df.columns if any(k in col.lower() for k in ['utxo', 'range_', 'spent_output', 'realized_'])],
        'network_activity': [col for col in df.columns if any(k in col.lower() for k in ['addresses_count', 'transactions_count', 'fees', 'block'])],
        'indicators_ratios': [col for col in df.columns if any(k in col.lower() for k in ['supply_ratio', 'stock_to_flow', 'cdd', 'shutdown'])],
        'price_ohlcv': [col for col in df.columns if any(k in col.lower() for k in ['close', 'open', 'high', 'low', 'volume'])] + ['forward_return'],
        'stablecoin': [col for col in df.columns if 'stablecoin' in col.lower()]
    }
    # Filter out empty groups
    groups = {k: v for k, v in groups.items() if v}
    return groups

# Step 4: Generate Features using Featuretools (per group)
def generate_features(df, groups, max_depth=2):
    all_features = pd.DataFrame(index=df.index)
    all_features['datetime'] = df['datetime']
    
    for group_name, cols in groups.items():
        if group_name == 'price_ohlcv': continue  # Skip target group
        
        print(f"Generating features for group: {group_name}")
        es = ft.EntitySet(id=group_name)
        es = es.add_dataframe(
            dataframe_name=group_name,
            dataframe=df[['datetime'] + cols],
            index=None,
            time_index='datetime'
        )
        
        # DFS with selected primitives
        feature_matrix, feature_defs = ft.dfs(
            entityset=es,
            target_dataframe_name=group_name,
            agg_primitives=['mean', 'std', 'percentile', 'trend'],  # Univariate aggs
            trans_primitives=['add_numeric', 'subtract_numeric', 'divide_numeric', 'diff'],  # Bivariate
            max_depth=max_depth,
            verbose=True
        )
        
        # Add to all features
        feature_matrix = feature_matrix.reset_index(drop=True)
        all_features = pd.concat([all_features, feature_matrix.drop(columns=['datetime'], errors='ignore')], axis=1)
    
    # Merge back with OHLCV and forward_return
    all_features = pd.concat([all_features, df[groups.get('price_ohlcv', [])].reset_index(drop=True)], axis=1)
    all_features = all_features.loc[:, ~all_features.columns.duplicated()]
    
    return all_features

# Step 5: Evaluate Features (IC and XGBoost Importance)
def calculate_ic(df, feature_cols, target='forward_return'):
    ic_results = {}
    for col in feature_cols:
        if col != target and pd.api.types.is_numeric_dtype(df[col]):
            ic = df[col].corr(df[target])
            ic_results[col] = ic
    return ic_results

def xgboost_importance(df, feature_cols, target='forward_return'):
    X = df[feature_cols].fillna(0)
    y = df[target].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    
    importance = model.feature_importances_
    return dict(zip(feature_cols, importance))

# Step 6: Select Top Alphas (High IC, Low Correlation)
def select_alphas(df, ic_results, importance_results, ic_threshold=0.03, corr_threshold=0.5, top_n=50):
    # Rank by abs(IC) + importance
    ranked = []
    for col in ic_results:
        score = abs(ic_results[col]) + importance_results.get(col, 0)
        ranked.append((col, score))
    
    ranked.sort(key=lambda x: x[1], reverse=True)
    candidates = [x[0] for x in ranked if abs(ic_results[x[0]]) > ic_threshold][:top_n * 2]  # Buffer
    
    # Filter low correlation
    selected = []
    corr_matrix = df[candidates].corr().abs()
    for col in candidates:
        if all(corr_matrix.loc[col, s] < corr_threshold for s in selected):
            selected.append(col)
        if len(selected) >= top_n:
            break
    
    return selected

# Main Pipeline
if __name__ == "__main__":
    df = load_and_merge_data()  # Step 1
    df = preprocess_data(df)    # Step 2
    groups = define_groups(df)  # Step 3
    
    # Generate features (this may take time; limit groups if needed)
    feature_df = generate_features(df, groups)  # Step 4
    
    # Evaluate
    feature_cols = [col for col in feature_df.columns if col not in ['datetime', 'forward_return']]
    ic_results = calculate_ic(feature_df, feature_cols)              # IC
    importance_results = xgboost_importance(feature_df, feature_cols)  # XGBoost
    
    # Select
    top_alphas = select_alphas(feature_df, ic_results, importance_results)
    print("Top Selected Alphas:", top_alphas)
    
    # Save output
    feature_df[top_alphas + ['datetime', 'forward_return']].to_csv('generated_features.csv', index=False)
    print("Generated features saved to 'generated_features.csv'") 