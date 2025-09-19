# Plot Generation Module

This module provides utilities for generating network topology, traffic data, and creating visualizations.

## Overview

The `plot_gen` module contains:

- `plot_gen()`: Main utility function that orchestrates the entire process
- `TrafficDataVisualizer`: Class for creating detailed visualizations

## Usage

### Basic Usage

```python
from maveric_platform_modules.plot_gen import plot_gen

# Generate plots with default parameters
result = plot_gen()

# Generate plots with custom parameters
result = plot_gen(
    output_dir="./my_plots",
    num_sites=10,
    cells_per_site=3,
    days=3,
    num_ues=500,
    plot_max_ticks=10
)
```

### Advanced Usage

```python
from maveric_platform_modules.plot_gen import plot_gen, TrafficDataVisualizer

# Custom spatial and time parameters
spatial_params = {
    "types": ["urban", "suburban", "rural"],
    "proportions": [0.5, 0.3, 0.2]
}

time_params = {
    "total_ticks": 24,
    "time_weights": {
        "urban": [1,1,1,1,1,1,2,3,4,5,6,6,5,5,6,7,8,8,7,6,5,4,3,2],
        "suburban": [1,1,1,1,1,1,1,2,3,4,4,4,4,4,4,5,5,5,4,3,2,2,1,1],
        "rural": [1,1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,2,2,2,1,1]
    }
}

result = plot_gen(
    output_dir="./custom_plots",
    spatial_params=spatial_params,
    time_params=time_params,
    num_sites=15,
    cells_per_site=2,
    days=2,
    num_ues=1000
)
```

## Classes

### TrafficDataVisualizer

The `TrafficDataVisualizer` class provides the core visualization functionality:

```python
from maveric_platform_modules.plot_gen import TrafficDataVisualizer

visualizer = TrafficDataVisualizer()

# Generate a single plot
visualizer.generate_tick_visualization(
    tick_to_plot=0,
    ue_data_for_tick=ue_dataframe,
    topology_df=topology_dataframe,
    spatial_layout=spatial_layout,
    spatial_params_for_colors=spatial_params,
    plot_output_path_template="./plot_tick_{tick}.png"
)
```

**Key Methods:**
- `generate_tick_visualization()`: Creates comprehensive plots with spatial areas, cell towers, Voronoi diagrams, and UE distributions

## Function Parameters

### plot_gen()

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | "./plots" | Directory to save plots |
| `organize_by_day` | bool | False | Whether to organize plots into day subdirectories |
| `num_sites` | int | 5 | Number of cell sites to generate |
| `cells_per_site` | int | 3 | Number of cells per site |
| `lat_range` | Tuple[float, float] | (40.7, 40.8) | Latitude range for topology |
| `lon_range` | Tuple[float, float] | (-74.05, -73.95) | Longitude range for topology |
| `default_cell_tilt` | float | 12.0 | Default cell tilt angle |
| `days` | int | 2 | Number of days to simulate |
| `spatial_params` | Dict[str, Any] | None | Spatial parameters (uses defaults if None) |
| `time_params` | Dict[str, Any] | None | Time parameters (uses defaults if None) |
| `num_ues` | int | 300 | Number of UEs per tick |
| `rng_seed` | Optional[int] | 0 | Random seed for reproducibility |
| `plot_max_ticks` | int | 0 | Maximum ticks to plot per day (0 = all) |
| `generate_dummy_training` | bool | True | Whether to generate dummy training data |
| `num_training_samples` | int | 6000 | Number of training samples |
| `training_ticks` | int | 24 | Number of training ticks |

## Return Value

The `plot_gen()` function returns a dictionary containing:

- `topology_df`: Generated topology DataFrame
- `config_df`: Generated config DataFrame  
- `dummy_training_df`: Generated dummy training DataFrame
- `ue_data_per_day`: List of UE DataFrames per day
- `spatial_layout`: Generated spatial layout
- `spatial_params`: Used spatial parameters
- `time_params`: Used time parameters
- `plots_generated`: List of generated plot file paths

## Output Structure

### Default Behavior (`organize_by_day=False`)
All plots are saved directly to the output directory:

```
output_dir/
├── ue_distribution_day1_tick_0.png
├── ue_distribution_day1_tick_1.png
├── ue_distribution_day2_tick_0.png
└── ...
```

### Organized by Day (`organize_by_day=True`)
Plots are organized into day-specific subdirectories:

```
output_dir/
├── day_1/
│   ├── ue_distribution_tick_0.png
│   ├── ue_distribution_tick_1.png
│   └── ...
├── day_2/
│   ├── ue_distribution_tick_0.png
│   └── ...
└── ...
```

## Configuration Files

The module includes configuration files for customizing plot appearance and behavior:

### `plot_params.json`
Contains default plot settings, colors, and plot types:
- **default_plot_settings**: Figure size, DPI, grid, legend position, font sizes
- **colors**: Color schemes for spatial types, UEs, and cell towers
- **plot_types**: Toggle which plot elements to display

### `visualization_params.json`
Contains visualization-specific settings:
- **default_visualization_settings**: Toggle display of various plot elements
- **marker_sizes**: Size settings for UEs, cell towers, and spatial areas
- **line_widths**: Line width settings for Voronoi edges and borders
- **text_settings**: Toggle display of labels and text elements

## Dependencies

- matplotlib
- pandas
- numpy
- scipy (for Voronoi diagrams)
- shapely (for spatial operations)

## Examples

See the `__main__` section in `plot_gen.py` for a complete working example.

### Quick Start Example

```python
from maveric_platform_modules.plot_gen import plot_gen

# Generate plots with default settings
result = plot_gen()

# Check what was generated
print(f"Generated {len(result['plots_generated'])} plots")
print(f"Plots saved to: {result['plots_generated']}")
```

### Custom Configuration Example

```python
# Use custom spatial and time parameters
spatial_params = {
    "types": ["dense", "suburban", "rural"],
    "proportions": [0.4, 0.4, 0.2]
}

time_params = {
    "total_ticks": 24,
    "time_weights": {
        "dense": [1,1,1,1,1,1,2,3,4,5,6,6,5,5,6,7,8,8,7,6,5,4,3,2],
        "suburban": [1,1,1,1,1,1,1,2,3,4,4,4,4,4,4,5,5,5,4,3,2,2,1,1],
        "rural": [1,1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,2,2,2,1,1]
    }
}

result = plot_gen(
    output_dir="./custom_plots",
    organize_by_day=True,  # Organize plots into day subdirectories
    spatial_params=spatial_params,
    time_params=time_params,
    num_sites=8,
    cells_per_site=2,
    days=3,
    num_ues=500,
    plot_max_ticks=5
)
```
