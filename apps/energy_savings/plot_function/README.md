# Plot Function Package

A Python package for generating D3-compatible JSON data for energy savings visualization. This package provides the `Plot` function that combines Reinforcement Learning (RL) model inference with Bayesian Digital Twin (BDT) model predictions to generate optimal cell configurations and UE assignments.

## Overview

The `Plot` function is designed to:
- Load and run inference on the Energy Savings RL model
- Generate optimal electrical tilt configurations for all cell IDs
- Use BDT models to predict RF signal strength and perform cell attachment
- Return structured JSON data optimized for D3.js visualization

## Features

- **RL Model Integration**: Uses Stable Baselines3 PPO model for optimal cell configuration
- **BDT Model Integration**: Leverages Bayesian Digital Twin for RF prediction and cell attachment
- **D3-Compatible Output**: Generates JSON data structure perfect for web visualization
- **Automatic Path Management**: Handles Python path setup and model file locations
- **Error Handling**: Graceful handling of missing model files and invalid inputs

## Installation

### Prerequisites

1. **Python Dependencies**:
   ```bash
   # From project root
   pip install -r radp/requirements.txt
   pip install -r apps/requirements.txt
   pip install stable-baselines3 scipy shapely
   ```

2. **Model Files** (required for full functionality):
   - `energy_saver_agent.zip` - RL model file
   - `bdt_model_map.pickle` - BDT model file
   - Place these in the `apps/energy_savings/` directory

## Usage

### Basic Usage

```python
from plot_function import Plot
import pandas as pd
import numpy as np

# Create input data
ue_data = pd.DataFrame({
    'loc_x': np.random.uniform(40.7, 40.8, 10),  # Longitude
    'loc_y': np.random.uniform(-74.05, -73.95, 10),  # Latitude
    'mock_ue_id': [f'ue_{i}' for i in range(10)]  # UE identifiers
})

# Generate plot data
plot_data = Plot(tick=12, input_dataset=ue_data)
print(f"Generated {len(plot_data['groups'])} groups")
```

### Function Signature

```python
def Plot(
    tick: int, 
    input_dataset: pd.DataFrame, 
    topology_path: str = None, 
    config_path: str = None
) -> Dict[str, Any]:
    """
    Generate D3-compatible JSON data for energy savings visualization.
    
    Args:
        tick (int): Time tick for RL model inference (0-23)
        input_dataset (pd.DataFrame): UE data with columns ['loc_x', 'loc_y', 'mock_ue_id']
        topology_path (str, optional): Path to topology data file
        config_path (str, optional): Path to configuration data file
        
    Returns:
        Dict[str, Any]: JSON data with 'groups' array containing plot points
        
    Raises:
        FileNotFoundError: If model files are missing
        ValueError: If input data format is invalid
    """
```

## Input Data Format

The `input_dataset` DataFrame must contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| `loc_x` | float | Longitude coordinates |
| `loc_y` | float | Latitude coordinates |
| `mock_ue_id` | string | Unique UE identifiers |

## Output Format

The function returns a dictionary with the following structure:

```json
{
  "groups": [
    {
      "title": "cells",
      "data": [
        {"x": 40.7, "y": -74.0},
        {"x": 40.71, "y": -74.01}
      ]
    },
    {
      "title": "cell_0",
      "data": [
        {"x": 40.705, "y": -74.02},
        {"x": 40.715, "y": -74.03}
      ]
    },
    {
      "title": "cell_1",
      "data": [
        {"x": 40.708, "y": -74.025}
      ]
    }
  ]
}
```

### Output Structure

- **`groups`**: Array of plot groups
- **`title`**: Group identifier ("cells" for cell towers, "cell_X" for UE assignments)
- **`data`**: Array of plot points with `x` and `y` coordinates
- **`x`**: Longitude coordinate
- **`y`**: Latitude coordinate

## How It Works

1. **RL Model Inference**: 
   - Loads the PPO model from `energy_saver_agent.zip`
   - Runs inference for the specified tick to get optimal cell configurations
   - Maps action indices to electrical tilt angles or "OFF" states

2. **BDT Model Prediction**:
   - Loads the BDT model from `bdt_model_map.pickle`
   - Predicts RF signal strength for each UE-cell pair
   - Performs cell attachment logic to assign UEs to optimal cells

3. **Data Generation**:
   - Creates cell tower location points
   - Groups UEs by their assigned cells
   - Returns structured JSON data for visualization

## Package Structure

```
apps/energy_savings/plot_function/
├── __init__.py              # Package initialization
├── plot_function.py         # Main Plot function
├── README.md               # This documentation
└── HOW_TO_TEST.md          # Testing guide
```

## Dependencies

### Core Dependencies
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `stable-baselines3` - RL model loading
- `scipy` - Scientific computing
- `shapely` - Geometric operations

### Internal Dependencies
- `radp.digital_twin.rf.bayesian_digital_twin` - BDT model
- `radp.digital_twin.mobility.perform_attachment` - Cell attachment logic

## Error Handling

### Expected Errors (Normal Operation)
- `FileNotFoundError`: When model files are missing (expected in test environments)
- `ValueError`: When input data format is invalid

### Error Recovery
- The function provides clear error messages for debugging
- Missing model files are handled gracefully with informative messages
- Input validation ensures data format compatibility

## Testing

See `HOW_TO_TEST.md` for comprehensive testing instructions and examples.

## Integration

### With Energy Savings App
This package is designed to integrate with the existing energy savings application:

```python
# In main_app.py or similar
from plot_function import Plot

# Use with existing data pipeline
plot_data = Plot(tick=current_tick, input_dataset=ue_data)
# Send to frontend for D3.js visualization
```

### With D3.js Frontend
The output format is optimized for D3.js visualization:

```javascript
// Frontend usage
fetch('/api/plot-data')
  .then(response => response.json())
  .then(data => {
    // data.groups contains the plot data
    // Use with D3.js for visualization
  });
```

## Model Files

### Required Files
- `energy_saver_agent.zip`: RL model trained with PPO algorithm
- `bdt_model_map.pickle`: BDT model for RF prediction

### File Locations
Model files should be placed in the `apps/energy_savings/` directory:
```
apps/energy_savings/
├── energy_saver_agent.zip    # RL model
├── bdt_model_map.pickle      # BDT model
└── plot_function/            # This package
```

## Performance Considerations

- **Model Loading**: Models are loaded once per function call (consider caching for production)
- **Memory Usage**: BDT model can be memory-intensive for large datasets
- **Processing Time**: RL inference is fast, BDT prediction scales with UE count

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're in the `apps/energy_savings` directory
2. **Model Files Missing**: Place model files in the correct directory
3. **Data Format Errors**: Verify DataFrame has required columns
4. **Python Path Issues**: The function handles path setup automatically

### Debug Mode

Enable debug logging by setting environment variable:
```bash
export PLOT_FUNCTION_DEBUG=1
```

## Contributing

When modifying this package:

1. Update tests in `HOW_TO_TEST.md`
2. Maintain backward compatibility
3. Update this README for any API changes
4. Test with both small and large datasets

## License

This package is part of the Maveric project. See the main project LICENSE for details.

## Support

For issues and questions:
1. Check `HOW_TO_TEST.md` for testing guidance
2. Verify model files are in the correct location
3. Ensure all dependencies are installed
4. Check the main project documentation for broader context
