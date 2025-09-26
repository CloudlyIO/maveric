# Export Function for Energy Savings (ES) Module

This package provides export functionality for Energy Savings metrics tables, specifically designed to work with the output from `rl_predictor.py`.

## Overview

The export function takes a DataFrame containing recommended cell configurations (metrics) and exports it to various formats including CSV, JSON, and Excel.

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
# Expected columns: ['cell_id', 'predicted_state', 'predicted_cell_el_deg']
metrics_df = pd.DataFrame({
    'cell_id': ['cell_001', 'cell_002', 'cell_003'],
    'predicted_state': ['ON', 'OFF', 'ON'],
    'predicted_cell_el_deg': [8.0, 'N/A', 12.0]
})

# Export to CSV
result = export_metrics(metrics_df, "./es_metrics", format="csv")
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
    model_load_path="./energy_saver_agent.zip",
    topology_path="./topology.csv", 
    target_tick=12
)

# Export the results
export_result = export_metrics(metrics_df, "./exported_metrics", format="all")
```

## Function Reference

### `export_metrics(metrics, output_path, format="csv", include_metadata=True)`

**Parameters:**
- `metrics` (pd.DataFrame): DataFrame with ES metrics
  - Required columns: `['cell_id', 'predicted_state', 'predicted_cell_el_deg']`
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
| `predicted_state` | str | Predicted cell state | 'ON', 'OFF' |
| `predicted_cell_el_deg` | float/str | Predicted electrical tilt | 8.0, 12.0, 'N/A' |

## Export Formats

### CSV Format
- Standard comma-separated values
- No metadata included
- Human-readable and spreadsheet-compatible

### JSON Format
- Structured JSON with optional metadata
- Includes export timestamp, module info, and data description
- Machine-readable and API-friendly

### Excel Format
- Microsoft Excel (.xlsx) format
- Single sheet named 'ES_Metrics'
- Spreadsheet-compatible with formatting

## Testing

Run the test script to verify functionality:

```bash
cd apps/energy_savings
python export_func/test_export.py
```

The test script will:
1. Create sample metrics data
2. Test all export formats
3. Validate data structure
4. Demonstrate integration with rl_predictor.py

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
- Handles file I/O errors
- Provides detailed error messages
- Returns success/failure status with error details

## Integration Notes

This export function is designed to work seamlessly with the Energy Savings module's `rl_predictor.py`. The metrics DataFrame structure matches exactly what `run_rl_prediction()` generates, making integration straightforward.
