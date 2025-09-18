Topology Generator

Purpose
- Produce synthetic topology and initial config, plus a dummy UE training dataset — all in memory.

Install
```
pip install pandas numpy shapely scipy
```

Usage
```
from maveric_platform_modules.topology_gen.topology_gen import topology_gen

topology_df, config_df, dummy_train_df = topology_gen(
    num_sites=5,
    cells_per_site=3,
    lat_range=(40.7, 40.8),
    lon_range=(-74.05, -73.95),
    default_cell_tilt=12.0,
    generate_dummy_training=True,
    num_training_samples=6000,
    training_ticks=24,
    rng_seed=0,
)

# Verify (3 DataFrames returned; no files written)
print('shapes:', topology_df.shape, config_df.shape, dummy_train_df.shape)
```

API
- Function: `topology_gen(...) -> list[pandas.DataFrame]`
- Inputs
  - `num_sites`, `cells_per_site`: Size of the synthetic network
  - `lat_range`, `lon_range`: Placement bounds for sites
  - `default_cell_tilt`: Default tilt for initial config
  - `generate_dummy_training`, `num_training_samples`, `training_ticks`, `rng_seed`: Dummy training generation controls
- Returns: `[topology_df, config_df, dummy_ue_training_df]`
  - `topology_df`: `[ecgi, site_id, cell_name, enodeb_id, cell_az_deg, tac, cell_lat, cell_lon, cell_id, cell_carrier_freq_mhz]`
  - `config_df`: `[cell_id, cell_el_deg, cell_carrier_freq_mhz]`
  - `dummy_ue_training_df`: `[cell_id, avg_rsrp, lon, lat, cell_el_deg]`

Notes
- Purely in-memory; no file writes. No plotting.
```
# Optional: persist to CSVs
topology_df.to_csv("topology.csv", index=False)
config_df.to_csv("config.csv", index=False)
dummy_train_df.to_csv("dummy_ue_training_data.csv", index=False)
```
