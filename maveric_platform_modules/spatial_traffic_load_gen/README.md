Spatial Traffic Load Generator

Purpose
- Produce per‑day UE datasets using Voronoi spatial cells and time‑of‑day weights — all in memory.

Install
```
pip install pandas numpy shapely scipy
```

Usage
```
from maveric_platform_modules.spatial_traffic_load_gen import (
    spatial_traffic_load_gen,
    load_default_spatial_params,
    load_default_time_params,
)

sp = load_default_spatial_params()
tp = load_default_time_params()

ue_days = spatial_traffic_load_gen(
    topology_df=topology_df,   # DataFrame from topology_gen
    days=2,
    spatial_params=sp,
    time_params=tp,
    num_ues=300,
    rng_seed=1,
)

# Verify (list of per-day DataFrames; no files written)
print('days:', len(ue_days), 'day1/day2:', ue_days[0].shape, ue_days[1].shape)
```

API
- Function: `spatial_traffic_load_gen(...) -> list[pandas.DataFrame]`
- Inputs
  - `topology_df`: must include `[cell_lat, cell_lon, cell_id]`
  - `days`: number of days to generate
  - `spatial_params`: `{types: [...], proportions: [...]}`
  - `time_params`: `{total_ticks: int, time_weights: {type: [weights...]}}`
  - `num_ues`: users per tick
  - `rng_seed`: seed for reproducibility (optional)
- Returns: `[ue_data_day_1, ue_data_day_2, ...]`
  - Each DF has `[ue_id, lon, lat, tick, space_type, day]` where `day` is 1‑based

Defaults
- If omitted, defaults are loaded from `spatial_params.json` and `time_params.json` in this folder (aligned with the legacy artifact).

Notes
- Purely in-memory; no file writes. No plotting. Decoupled from topology generation.
```
# Optional: persist
ue_days[0].to_csv("ue_day_1.csv", index=False)
```
