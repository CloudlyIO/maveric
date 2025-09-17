Artifacts: Minimal Run Guide

Run these from the repo root. Defaults write under `artifact/generated_data`.

Setup
- Create venv and install deps:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r artifact/requirements.txt`

Generate: Topology + Config
- Command:
  - `python -c "from artifact import topology_gen; r=topology_gen(); print(r['topology_path'], r['config_path'])"`
- Writes:
  - `artifact/generated_data/trafficgen/topology.csv`
  - `artifact/generated_data/trafficgen/config.csv`

Generate: Spatial UE Traffic (per-tick files)
- Command:
  - `python -c "from artifact import spatial_traffic_load_gen; spatial_traffic_load_gen(days=1, num_ues=100, generate_dummy_training=True)"`
- Writes:
  - `artifact/generated_data/trafficgen/Day_0/ue_data_per_tick/ue_data_tick_*.csv`
  - Optional training: `artifact/generated_data/trafficgen/dummy_ue_training_data.csv`
  - Optional plots (add `plot=True`): `artifact/generated_data/trafficgen/Day_0/plots/*.png`

Generate: MRO UE Tracks
- Command:
  - `python -c "from artifact import mro_gen; mro_gen()"`
- Writes:
  - `artifact/generated_data/mro/ue_tracks.csv` (uses `artifact/mro/params_example.json`)

Notes
- Use Python 3.10+.
- If you run from a different working directory, pass absolute paths via `output_dir`/`params`.

Included Sample Data (committed)
```
artifact/generated_data
├── trafficgen
│   ├── topology.csv
│   ├── config.csv
│   ├── Day_0
│   │   ├── ue_data_per_tick
│   │   │   ├── ue_data_tick_0.csv
│   │   │   ├── ue_data_tick_1.csv
│   │   │   └── …
│   │   └── plots
│   │       ├── ue_distribution_tick_0.png
│   │       ├── ue_distribution_tick_11.png
│   │       └── …
│   └── dummy_ue_training_data.csv
└── mro
    └── ue_tracks.csv
```
Regenerating will overwrite files and may change content if randomness is involved. Defaults use fixed seeds where applicable to keep outputs stable.
