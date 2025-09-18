Maveric Platform Modules

Purpose
- Modular, side‑effect‑free generators for data synthesis. No file writes, no plotting. Everything returns Pandas DataFrames.

Structure
```
maveric_platform_modules/
├─ README.md
├─ topology_gen/
│  ├─ README.md
│  ├─ __init__.py
│  └─ topology_gen.py
└─ spatial_traffic_load_gen/
   ├─ README.md
   ├─ __init__.py
   ├─ spatial_traffic_load_gen.py
   ├─ spatial_params.json
   └─ time_params.json
```

Install
```
pip install pandas numpy shapely scipy
```

Quick Start
```
from maveric_platform_modules.topology_gen.topology_gen import topology_gen
from maveric_platform_modules.spatial_traffic_load_gen import (
    spatial_traffic_load_gen,
    load_default_spatial_params,
    load_default_time_params,
)

# 1) Topology + Dummy Training
topology_df, config_df, dummy_train_df = topology_gen(
    num_sites=5,
    cells_per_site=3,
    num_training_samples=6000,
)

# 2) UE Data Per Day
sp, tp = load_default_spatial_params(), load_default_time_params()
ue_days = spatial_traffic_load_gen(
    topology_df=topology_df,
    days=2,
    spatial_params=sp,
    time_params=tp,
    num_ues=300,
)

# Verify (prints shapes; no files written)
print('topology/config/dummy:', topology_df.shape, config_df.shape, dummy_train_df.shape)
print('day1/day2:', ue_days[0].shape, ue_days[1].shape)
```

Notes
- Dependencies: pandas, numpy, scipy (Voronoi), shapely.
- All functions return DataFrames in-memory; nothing is written unless you call `.to_csv(...)` yourself.
