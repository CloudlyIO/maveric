MRO: Parameters and Examples

Outputs default to `artifact/generated_data/mro`.

API
- `mro_gen(...)`
  - `params`: path to params JSON (default `artifact/mro/params_example.json`)
  - `output_dir`: where to write outputs (default `artifact/generated_data/mro`)
  - `filename`: output CSV filename (default `ue_tracks.csv`)

Examples
- Generate tracks with default params
  - `python -c "from artifact import mro_gen; mro_gen()"`
  - Writes `artifact/generated_data/mro/ue_tracks.csv`.

- Generate with a custom params file
  - `python -c "from artifact import mro_gen; mro_gen(params='artifact/mro/params_example.json', output_dir='artifact/generated_data/mro', filename='tracks.csv')"`

Params JSON (overview)
- File: `artifact/mro/params_example.json`
- Key fields under `ue_tracks_generation.params`:
  - `gauss_markov_params.rng_seed`: random seed (int)
  - `gauss_markov_params.lon_x_dims`, `lon_y_dims`: grid dims (int)
  - `gauss_markov_params.alpha`, `variance`: Gauss–Markov process params (float)
  - `num_ticks`, `num_batches`: simulation length and batching (int)
  - `lat_lon_boundaries.{min_lat,max_lat,min_lon,max_lon}`: bounds (float)
  - `simulation_time_interval_seconds`: tick interval (float)
  - `ue_class_distribution`: counts, velocities, variances per class

