# Export Function for Load Balancing (CCO) Module

This package provides export functionality for Load Balancing metrics tables, specifically designed to work with the output from `rl_predictor.py`.

## Overview

The export function takes a DataFrame containing recommended cell tilt configurations (metrics) and exports it to various formats including CSV, JSON, and Excel. This is designed for Coverage and Capacity Optimization (CCO) scenarios.

## Structure

```
export_func/
├── __init__.py          # Package initialization
├── export_metrics.py    # Main export functionality
├── test_export.py       # Test script and examples
└── README.md           # This documentation
```

## Usage

### Basic Usage

```python
import pandas as pd
from export_func import export_metrics

# Create or load your metrics DataFrame
# Expected columns: ['cell_id', 'predicted_cell_el_deg']
metrics_df = pd.DataFrame({
    'cell_id': ['cell_001', 'cell_002', 'cell_003'],
    'predicted_cell_el_deg': [8.0, 12.0, 5.0]
})

# Export to CSV
result = export_metrics(metrics_df, "./lb_metrics", format="csv")
print(f"Success: {result['success']}")
print(f"Files: {result['files_created']}")
```

### Integration with rl_predictor.py

```python
from rl_predictor import run_rl_prediction
from export_func import export_metrics

# Run RL prediction (this would normally return a DataFrame)
# For demonstration, we'll assume it returns the metrics DataFrame
metrics_df = run_rl_prediction(
    model_load_path="./cco_agent.zip",
    topology_path="./topology.csv", 
    target_tick=12
)

# Export the results
export_result = export_metrics(metrics_df, "./exported_metrics", format="all")
```

## Function Reference

### `export_metrics(metrics, output_path, format="csv", include_metadata=True)`

**Parameters:**
- `metrics` (pd.DataFrame): DataFrame with Load Balancing metrics
  - Required columns: `['cell_id', 'predicted_cell_el_deg']`
- `output_path` (str): Base path for output files (timestamp will be appended)
- `format` (str): Export format - 'csv', 'json', 'excel', or 'all' (default: 'csv')
- `include_metadata` (bool): Include metadata in JSON exports (default: True)

**Returns:**
- `Dict[str, Any]`: Result dictionary with success status and file paths
  ```python
  {
      'success': True,
      'files_created': ['path/to/file.csv'],
      'errors': []
  }
  ```

### `validate_metrics_format(metrics)`

**Parameters:**
- `metrics` (pd.DataFrame): DataFrame to validate

**Returns:**
- `Dict[str, Any]`: Validation result with success status and any errors/warnings

## Data Format

The export function expects a DataFrame with the following structure:

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `cell_id` | str | Unique cell identifier | 'cell_001', 'cell_002' |
| `predicted_cell_el_deg` | float | Predicted electrical tilt angle | 0.0, 5.0, 10.0, 15.0, 20.0 |

**Note:** Tilt angles should be in the range 0-20 degrees as per the `TILT_SET_PREDICTOR` configuration.

## Export Formats

### CSV Format
- Standard comma-separated values
- No metadata included
- Human-readable and spreadsheet-compatible

### JSON Format
- Structured JSON with optional metadata
- Includes export timestamp, module info, and data description
- Machine-readable and API-friendly
- Special metadata for Load Balancing:
  - `module`: "Load Balancing (CCO)"
  - `optimization_type`: "Coverage and Capacity Optimization (CCO)"
  - `tilt_range`: "0-20 degrees"

### Excel Format
- Microsoft Excel (.xlsx) format
- Single sheet named 'LB_Metrics'
- Spreadsheet-compatible with formatting

## Testing

Run the test script to verify functionality:

```bash
cd apps/load_balancing
python export_func/test_export.py
```

The test script will:
1. Create sample metrics data
2. Test all export formats
3. Validate data structure and tilt ranges
4. Test error handling scenarios
5. Demonstrate integration with rl_predictor.py

## Dependencies

- `pandas`: DataFrame manipulation
- `openpyxl`: Excel file support (for Excel exports)
- `json`: JSON serialization (built-in)
- `datetime`: Timestamp generation (built-in)
- `logging`: Logging functionality (built-in)

## Error Handling

The export function includes comprehensive error handling:
- Validates input DataFrame structure
- Checks for required columns
- Validates tilt angle ranges (0-20 degrees)
- Detects duplicate cell IDs
- Handles file I/O errors
- Provides detailed error messages
- Returns success/failure status with error details

## Validation Features

The validation function checks for:
- **Required columns**: `cell_id`, `predicted_cell_el_deg`
- **Data types**: Ensures tilt values are numeric
- **Tilt range**: Validates 0-20 degree range
- **Cell ID integrity**: Checks for nulls and duplicates
- **Data completeness**: Ensures no missing critical data

## Integration Notes

This export function is designed to work seamlessly with the Load Balancing module's `rl_predictor.py`. The metrics DataFrame structure matches exactly what `run_rl_prediction()` generates through the `map_action_to_config_df()` function, making integration straightforward.

## Differences from Energy Savings Export

- **Data structure**: Only tilt angles (no ON/OFF states)
- **Validation**: Focuses on tilt range validation (0-20 degrees)
- **Metadata**: Includes CCO-specific information
- **Use case**: Coverage and capacity optimization vs energy consumption optimization
