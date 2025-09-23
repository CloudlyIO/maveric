Spatial Traffic Generator

Overview
- Generates per-day UE datasets distributed across Voronoi cells.
- Uses spatial and time parameter JSONs (defaults provided).

Function
- `spatial_traffic_load_gen(*, topology_df, days=2, spatial_params=None, time_params=None, num_ues=300, rng_seed=0) -> list[pd.DataFrame]`

Returns
- List of DataFrames per day with columns: `ue_id, lon, lat, tick, space_type, day`

Quick Start
```
from maveric_platform_modules.radp.util.datagen.spatial_traffic import (
    spatial_traffic_load_gen, load_default_spatial_params, load_default_time_params
)

sp = load_default_spatial_params()
tp = load_default_time_params()
days = spatial_traffic_load_gen(
    topology_df=topology_df,
    days=2,
    spatial_params=sp,
    time_params=tp,
    num_ues=300,
)
```

Notes
- UE positions are sampled within clipped Voronoi polygons of tower locations.
- Time-of-day weights drive proportional UE counts per tick by area type.
