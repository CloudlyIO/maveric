# How to Test the Plot Function

Simple guide to test the `plot_function` package.

## Quick Test

1. **Navigate to the directory**:
   ```bash
   cd apps/energy_savings
   ```

2. **Run Python and test**:
   ```bash
   python3
   ```

3. **Copy and paste this code**:
   ```python
   from plot_function import Plot
   import pandas as pd
   import numpy as np

   # Create sample data
   ue_data = pd.DataFrame({
       'loc_x': np.random.uniform(40.7, 40.8, 10),
       'loc_y': np.random.uniform(-74.05, -73.95, 10),
       'mock_ue_id': [f'ue_{i}' for i in range(10)]
   })

   # Test the function
   plot_data = Plot(tick=12, input_dataset=ue_data)
   print(f"Generated {len(plot_data['groups'])} groups")
   ```

## Expected Results

### Success (with model files)
- Function returns JSON data with 'groups' array
- Each group has 'title' and 'data' fields
- Each data point has 'x' and 'y' coordinates

### Expected Error (without model files)
- `FileNotFoundError: RL Model file not found`
- `FileNotFoundError: BDT Model file not found`
- This is normal - confirms the function works correctly

## Test Data Format

DataFrame needs these columns:
- `loc_x`: Longitude coordinates
- `loc_y`: Latitude coordinates  
- `mock_ue_id`: Unique UE identifiers

## Troubleshooting

1. **Import Error**: Make sure you're in the `apps/energy_savings` directory
2. **Model Files Missing**: This is expected - function will work once model files are available
3. **Data Format Error**: Ensure your DataFrame has the required columns

## End

The function is ready to use. Once you have the model files in place, it will generate the complete visualization data.
