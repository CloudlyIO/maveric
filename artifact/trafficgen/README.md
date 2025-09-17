Trafficgen: Parameters and Examples

Outputs default to `artifact/generated_data/trafficgen`.

APIs
- `topology_gen(...)`
  - `num_sites`: number of eNB/gNB sites (default 5)
  - `cells_per_site`: sectors per site (default 3)
  - `lat_range`: tuple `(min_lat, max_lat)` (default `(40.7, 40.8)`)
  - `lon_range`: tuple `(min_lon, max_lon)` (default `(-74.05, -73.95)`)
  - `default_cell_tilt`: electrical tilt degrees (default `12.0`)
  - `output_dir`: where to write outputs (default `artifact/generated_data/trafficgen`)
  - `topology_filename`: filename for topology CSV (default `topology.csv`)
  - `config_filename`: filename for config CSV (default `config.csv`)

- `spatial_traffic_load_gen(...)`
  - `days`: number of simulated days (default `2`)
  - `output_dir`: base output directory (default `artifact/generated_data/trafficgen`)
  - `topology_csv`: input topology CSV name in `output_dir` (default `topology.csv`)
  - `config_csv`: input config CSV name in `output_dir` (default `config.csv`)
  - `spatial_params`: path or dict for spatial params JSON (default `artifact/trafficgen/spatial_params.json`)
  - `time_params`: path or dict for time params JSON (default `artifact/trafficgen/time_params.json`)
  - `num_ues`: UEs per tick (default `300`)
  - `ue_data_dir`: subfolder for per-tick UE CSVs (default `ue_data_per_tick`)
  - `plot`: generate plots (default `False`)
  - `plot_dir`: subfolder for plots (default `plots`)
  - `plot_max_ticks`: max ticks to plot per day (default `5`)
  - `generate_dummy_training`: write combined UE training CSV (default `False`)
  - `dummy_training_csv`: filename for training CSV (default `dummy_ue_training_data.csv`)
  - `num_training_samples`: rows for training CSV if enabled (default `6000`)

Examples
- Generate topology and config
  - `python -c "from artifact import topology_gen; topology_gen(num_sites=3, cells_per_site=3)"`
  - Writes `topology.csv`, `config.csv` under `artifact/generated_data/trafficgen`.

- Generate 1 day of spatial UE data (no plots)
  - `python -c "from artifact import spatial_traffic_load_gen; spatial_traffic_load_gen(days=1, num_ues=200)"`
  - Writes per-tick CSVs under `artifact/generated_data/trafficgen/Day_0/ue_data_per_tick`.

- Generate with training CSV and plots
  - `python -c "from artifact import spatial_traffic_load_gen; spatial_traffic_load_gen(days=1, num_ues=200, generate_dummy_training=True, plot=True)"`
  - Writes training CSV at `artifact/generated_data/trafficgen/dummy_ue_training_data.csv` and plots under `Day_0/plots`.

Param JSONs
- Defaults live at:
  - `artifact/trafficgen/spatial_params.json`
  - `artifact/trafficgen/time_params.json`
- You can pass a dict instead of a file; it will be saved alongside outputs.

