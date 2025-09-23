RADP Datagen (Topology + Spatial Traffic)

Purpose
- Modular, side‑effect‑free generators. No file writes, no plotting. Everything returns Pandas DataFrames.

Structure
```
maveric_platform_modules/radp/util/datagen/
├─ README.md  # this file
├─ topology/
│  ├─ README.md
│  ├─ __init__.py            # exports topology_gen
│  ├─ topo.py                # implementation
│  └─ lib/                   # optional helpers
└─ spatial_traffic/
   ├─ README.md
   ├─ __init__.py            # exports spatial_traffic_load_gen + loaders
   ├─ spatial_traffic.py     # implementation
   └─ lib/
      ├─ spatial_params.json
      └─ time_params.json
```

Install
```
pip install pandas numpy shapely scipy
```

Quick Start (New Paths)
```
from maveric_platform_modules.radp.util.datagen.topology import topology_gen
from maveric_platform_modules.radp.util.datagen.spatial_traffic import (
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

print('topology/config/dummy:', topology_df.shape, config_df.shape, dummy_train_df.shape)
print('day1/day2:', ue_days[0].shape, ue_days[1].shape)
```

Notes
- Dependencies: pandas, numpy, scipy (Voronoi), shapely.
- All functions return DataFrames in-memory; nothing is written unless you call `.to_csv(...)` yourself.
