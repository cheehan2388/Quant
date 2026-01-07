# Feature Engineering Template

A comprehensive feature engineering pipeline that applies multiple transformation techniques to generate and evaluate features based on Information Coefficient (IC) analysis.

## Overview

This template implements a systematic approach to feature engineering by:

1. **Loading and preprocessing data**
2. **Applying 9 different transformation tools** to all numeric columns
3. **Calculating Information Coefficients** against future returns with multiple time shifts
4. **Generating comprehensive reports** in JSON format
5. **Saving all transformed features** to CSV

## Transformation Tools

The template applies the following 9 transformation tools to each numeric column:

### Rolling Window Means
- `rolling_mean_12`: 12-period rolling mean
- `rolling_mean_24`: 24-period rolling mean  
- `rolling_mean_52`: 52-period rolling mean

### Rolling Min-Max Scaling
- `rolling_minmax_12`: Min-max scaling with 12-period rolling window
- `rolling_minmax_24`: Min-max scaling with 24-period rolling window
- `rolling_minmax_52`: Min-max scaling with 52-period rolling window

### Rolling Z-Score
- `rolling_zscore_12`: Z-score normalization with 12-period rolling window
- `rolling_zscore_24`: Z-score normalization with 24-period rolling window
- `rolling_zscore_52`: Z-score normalization with 52-period rolling window

## IC Evaluation

For each transformed feature, the template:

1. **Calculates IC** with future returns at shifts: -2, -8, -24 periods
2. **Selects the maximum absolute IC** as the representative score
3. **Records the best shift window** for each feature

## Files

### Main Files
- `feature_engineering_template.py`: Main template implementation
- `example_usage.py`: Complete usage example with sample data

### Required Data Structure
- `merged_by.csv`: Input data file with datetime column and numeric features
- `../Data/close/`: Directory containing CSV files with close prices for future return calculation

## Usage

### Basic Usage

```python
from feature_engineering_template import main

# Run the complete pipeline
main()
```

### Custom Usage

```python
from feature_engineering_template import *

# Load your data
df = load_data('your_data.csv')

# Calculate future returns
df, future_returns = calculate_future_returns(df)

# Generate transformed features
transformed_features, feature_metadata = generate_all_features(df)

# Calculate IC values
df, feature_metadata = calculate_ics_for_all_features(df, transformed_features, feature_metadata)

# Generate report
report = generate_report(feature_metadata)

# Save results
save_transformed_features(df)
```

### Running the Example

```bash
cd Data_Engineering
python example_usage.py
```

## Input Data Requirements

### Main Data File (`merged_by.csv`)
- Must have a `datetime` column
- Should contain numeric features for transformation
- Missing values will be forward-filled then backward-filled

### Close Price Data (`../Data/close/*.csv`)
- Each file should have `datetime` and `Close` columns
- Used for calculating future returns
- If directory doesn't exist, dummy returns will be generated

## Output Files

### JSON Report (`feature_engineering_report.json`)
Contains comprehensive information about each transformed feature:

```json
{
  "summary": {
    "total_features": 45,
    "transformation_types": [...],
    "shift_windows": [-2, -8, -24],
    "top_10_features": [...]
  },
  "features": {
    "feature_name": {
      "base_column": "original_column",
      "transformation": "rolling_mean_12",
      "ic_values": {
        "shift_-2": 0.123,
        "shift_-8": 0.089,
        "shift_-24": 0.156
      },
      "max_ic": 0.156,
      "best_shift": "shift_-24",
      "abs_max_ic": 0.156
    }
  }
}
```

### CSV File (`transformed_features.csv`)
Contains all original and transformed features ready for further analysis or modeling.

## Key Features

### Robust Error Handling
- Handles missing data automatically
- Graceful degradation when close price data is unavailable
- Continues processing even if individual transformations fail

### Comprehensive Reporting
- Detailed IC analysis for each feature
- Summary statistics and rankings
- Transformation breakdown analysis

### Scalable Design
- Processes any number of input features
- Easy to add new transformation tools
- Efficient memory usage with streaming calculations

### Flexible Configuration
- Customizable shift windows for future returns
- Adjustable rolling window sizes
- Configurable output paths

## Performance Considerations

- **Memory Usage**: Large datasets may require chunked processing
- **Computation Time**: Processing time scales with (features × transformations × shift_windows)
- **Storage**: Output files can be large with many features

## Example Output

After running the pipeline, you'll see output like:

```
=== TOP 10 FEATURES BY ABSOLUTE IC ===
 1. feature_1_rolling_mean_24
    Base Column: feature_1
    Transformation: rolling_mean_24
    Max IC: 0.234567
    Best Shift: shift_-8

 2. feature_3_rolling_zscore_12
    Base Column: feature_3
    Transformation: rolling_zscore_12
    Max IC: -0.198765
    Best Shift: shift_-2
```

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Ensure `merged_by.csv` exists in the current directory
2. **Empty IC values**: Check that future return calculation is working correctly
3. **Memory errors**: Consider reducing the number of features or using chunked processing

### Tips

- Start with a small subset of features for testing
- Monitor memory usage with large datasets
- Check data quality before running transformations
- Verify datetime columns are properly formatted

## Extending the Template

### Adding New Transformations

```python
def apply_custom_transform(series, window):
    """Your custom transformation logic"""
    return transformed_series

# Add to transformations dictionary in generate_all_features()
transformations['custom_transform'] = lambda x: apply_custom_transform(x, 10)
```

### Modifying Shift Windows

```python
# Change shift windows in main() function
shifts = [-1, -5, -10, -20]  # Your custom shifts
```

## Dependencies

- pandas
- numpy  
- scikit-learn
- Standard library modules: os, json, warnings