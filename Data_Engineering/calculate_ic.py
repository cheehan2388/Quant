import pandas as pd
import os
invisible =["low","open","high",'start_time_x','end_time_x',"start_time_y","end_time_y","close"]
def calculate_ic(factor_path, close_path):
    """
    Calculates the Information Coefficient (IC) between factors and forward returns.
    """
    try:
        # Load the datasets
        factor_df = pd.read_csv(factor_path)
        close_df = pd.read_csv(close_path)

        # Ensure 'datetime' column exists and is in datetime format
        if 'datetime' not in factor_df.columns or 'datetime' not in close_df.columns:
            print(f"Skipping because 'datetime' column is missing in {factor_path} or {close_path}")
            return
        
        factor_df['datetime'] = pd.to_datetime(factor_df['datetime'])
        close_df['datetime'] = pd.to_datetime(close_df['datetime'])

        # Merge the two dataframes on 'datetime'
        # This aligns the data based on the factor's datetime index
        merged_df = pd.merge(factor_df, close_df, on='datetime', how='inner')
        merged_df.dropna()
        # Sort by datetime to ensure correct return calculation
        merged_df = merged_df.sort_values(by='datetime').reset_index(drop=True)

        # Calculate forward returns for the 'close' price
        merged_df['forward_return'] = merged_df['Close'].pct_change().shift(-1)

        # Drop rows with NaN values that result from merging and shifting
        merged_df.dropna(subset=['forward_return'], inplace=True)

        # Isolate factor columns (assuming all other columns are factors)
        factor_columns = [col for col in factor_df.columns if col not in invisible]

        # Calculate IC for each factor
        ic_results = {}
        for factor_col in factor_columns:
            if factor_col in merged_df.columns and pd.api.types.is_numeric_dtype(merged_df[factor_col]):
                ic = merged_df[factor_col].corr(merged_df['forward_return'])
                ic_results[factor_col] = ic

        return ic_results

    except Exception as e:
        print(f"An error occurred while processing {factor_path} and {close_path}: {e}")
        return None

def main():
    """
    Main function to run the IC calculation for all files.
    """
    factor_dir = './../Data/factor'
    close_dir = './../Data/close'

    # Get all factor and close price files
    factor_files = [os.path.join(factor_dir, f) for f in os.listdir(factor_dir) if f.endswith('.csv')]
    close_files = [os.path.join(close_dir, f) for f in os.listdir(close_dir) if f.endswith('.csv')]

    # Process each factor against each close price file
    for factor_file in factor_files:
        print(f"Processing Factor File: {os.path.basename(factor_file)}")
        print("-" * 50)
        for close_file in close_files:
            ic_values = calculate_ic(factor_file, close_file)
            if ic_values:
                print(f"  Close Price File: {os.path.basename(close_file)}")
                for factor, ic in ic_values.items():
                    print(f"    - IC for '{factor}': {ic:.4f}")
        print("\n")

if __name__ == "__main__":
    main()
