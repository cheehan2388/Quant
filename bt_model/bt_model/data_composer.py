from typing import List, Dict
import pandas as pd
from itertools import permutations, combinations


def get_operation_symbol(operation: str) -> str:
    op_symbols = {
        'multiply': '×',
        'divide': '÷',
        'add': '+',
        'subtract': '-'
    }
    return op_symbols.get(operation, operation)


def apply_operation(df: pd.DataFrame, col1: str, col2: str, operation: str) -> pd.Series:
    """Apply the specified operation to two columns."""
    if operation == 'multiply':
        return df[col1] * df[col2]
    elif operation == 'divide':
        return df[col1] / df[col2]
    elif operation == 'add':
        return df[col1] + df[col2]
    elif operation == 'subtract':
        return df[col1] - df[col2]
    else:
        raise ValueError(f"Unsupported operation: {operation}")


def create_composition_name(cols: List[str], operation: str = None) -> str:
    """Create a descriptive name for the composition."""
    if operation is None:
        return cols[0]  # For single column case
    op_symbol = get_operation_symbol(operation)
    return f"({cols[0]} {op_symbol} {cols[1]})"


def advanced_multi_compose_data(df: pd.DataFrame, cols: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Create compositions based on input columns:
    - For single column: Returns DataFrame with just that column
    - For multiple columns: Creates all possible arithmetic combinations
    
    Returns a dictionary of DataFrames, each with different compositions.
    """
    composed_dfs = {}
    
    # Handle single column case
    if len(cols) == 1:
        df_copy = df.copy()
        composition_name = create_composition_name(cols)
        composed_dfs[composition_name] = df_copy
        return composed_dfs

    # For multiple columns, create arithmetic combinations
    # Operations where order doesn't matter (add, multiply)
    order_independent_ops = ['add', 'multiply']
    col_pairs = list(combinations(cols, 2))

    # Operations where order matters (subtract, divide)
    order_dependent_ops = ['subtract', 'divide']
    col_perms = list(permutations(cols, 2))

    # Process order-independent operations
    for operation in order_independent_ops:
        for col1, col2 in col_pairs:
            df_copy = df.copy()
            composition_name = create_composition_name([col1, col2], operation)
            df_copy[composition_name] = apply_operation(df_copy, col1, col2, operation)
            composed_dfs[composition_name] = df_copy

    # Process order-dependent operations
    for operation in order_dependent_ops:
        for col1, col2 in col_perms:
            df_copy = df.copy()
            composition_name = create_composition_name([col1, col2], operation)
            df_copy[composition_name] = apply_operation(df_copy, col1, col2, operation)
            composed_dfs[composition_name] = df_copy

    return composed_dfs


def print_composition_results(composition_name: str, results_df: pd.DataFrame):
    """
    Print the backtest results for a specific composition with proper formatting.
    """
    separator = "=" * 100
    print(f"\n{separator}")
    print(f"Results for Composition: {composition_name}".center(100))
    print(separator)
    print("\nTop 10 Results:")
    print(results_df.head(10))
    print("\nBottom 10 Results:")
    print(results_df.tail(10))
    print(separator)