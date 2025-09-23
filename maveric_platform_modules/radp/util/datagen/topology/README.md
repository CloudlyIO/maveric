Topology Generator

Overview
- Creates synthetic topology and config DataFrames (optionally dummy UE training).
- Pure in-memory generation; no files written.

Function
- `topology_gen(*, num_sites=5, cells_per_site=3, lat_range=(40.7, 40.8), lon_range=(-74.05, -73.95), default_cell_tilt=12.0, generate_dummy_training=True, num_training_samples=6000, training_ticks=24, rng_seed=0) -> list[pd.DataFrame]`

Returns
- `[topology_df, config_df, dummy_ue_training_df]`
  - `topology_df`: `ecgi, site_id, cell_name, enodeb_id, cell_az_deg, tac, cell_lat, cell_lon, cell_id, cell_carrier_freq_mhz`
  - `config_df`: `cell_id, cell_el_deg, cell_carrier_freq_mhz`
  - `dummy_ue_training_df`: `cell_id, avg_rsrp, lon, lat, cell_el_deg`

Quick Start
```
from maveric_platform_modules.radp.util.datagen.topology import topology_gen

topology_df, config_df, dummy_train_df = topology_gen(
    num_sites=5,
    cells_per_site=3,
    generate_dummy_training=False,
)
```

Notes
- Dummy training is optional and sampled within the topology extent.
