Maveric In‑Memory Generators

Purpose
- Self‑contained utilities for topology, UE traffic, and plotting.
- In‑memory only: returns pandas DataFrames and matplotlib Figures.

Folder
```
maveric_partform_module/
├─ __init__.py                  # re‑exports public functions
├─ core.py                      # local topology/traffic core 
├─ topology_gen.py              # topology + config generator (DFs)
├─ spatial_traffic_load_gen.py  # UE traffic per‑day (DFs)
├─ plot_gen.py                  # figures from generated data
├─ utils.py                     # convenience wrappers
├─ spatial_params.json          # default spatial mix
├─ time_params.json             # default time profile
├─ check_in_memory.py           # CLI sanity checker
└─ quick_check_in_memory.ipynb  # optional notebook for checking
```

Install
- Python 3.10+
- pip install: `pandas shapely scipy matplotlib`

Quick Start
```python
from maveric_partform_module import topology_gen, spatial_traffic_load_gen, plot_gen

topo_df, cfg_df, _ = topology_gen(num_sites=5, cells_per_site=3)
days = spatial_traffic_load_gen(topology_df=topo_df, days=2, num_ues_per_tick=300)
figs = plot_gen(days=1, num_sites=5, cells_per_site=3, num_ues_per_tick=300, plot_max_ticks=5)
```

API
- topology_gen.topology_gen
  - Inputs: num_sites, cells_per_site, lat_range, lon_range, default_cell_tilt, [generate_dummy_training, ue_data_for_training, num_training_samples, possible_tilts, assumed_optimal_tilt, tilt_penalty_factor]
  - Returns: [topology_df, config_df, dummy_training_df]
- spatial_traffic_load_gen.spatial_traffic_load_gen
  - Inputs: topology_df, days, num_ues_per_tick, [spatial_params dict|path, time_params dict|path]
  - Returns: list of per‑day DataFrames with columns ['ue_id','lon','lat','tick','space_type','day']
- plot_gen.plot_gen
  - Inputs: num_sites, cells_per_site, lat_range, lon_range, default_cell_tilt, days, num_ues_per_tick, [spatial_params, time_params, plot_max_ticks]
  - Returns: dict[(day,tick)] -> matplotlib Figure

CLI
- Module: `python -m maveric_partform_module.check_in_memory --num-sites 3 --cells-per-site 3 --days 1 --num-ues 50 --plot-max-ticks 2 --save-plot /tmp/sample.png`
- Script: `python maveric_partform_module/check_in_memory.py --num-sites 3 --cells-per-site 3 --days 1 --num-ues 50 --plot-max-ticks 2 --save-plot /tmp/sample.png`

Notes
- Defaults use local JSONs; pass dicts/paths to override.
- No file writes; save figures yourself via `fig.savefig(...)` if desired.
