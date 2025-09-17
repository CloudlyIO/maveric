# Copied from maveric/artifact/mro/datagen.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from .ue_tracks_params import UETracksGenerationParams
from .ue_tracks import UETracksGenerator


def get_ue_data(params: Dict) -> pd.DataFrame:
    ue_tracks_params = UETracksGenerationParams(params)

    ue_tracks_generation = pd.DataFrame()
    for batch_df in UETracksGenerator.generate_as_lon_lat_points(
        rng_seed=ue_tracks_params.rng_seed,
        lon_x_dims=ue_tracks_params.lon_x_dims,
        lon_y_dims=ue_tracks_params.lon_y_dims,
        num_ticks=ue_tracks_params.num_ticks,
        num_batches=ue_tracks_params.num_batches,
        num_UEs=ue_tracks_params.num_UEs,
        alpha=ue_tracks_params.alpha,
        variance=ue_tracks_params.variance,
        min_lat=ue_tracks_params.min_lat,
        max_lat=ue_tracks_params.max_lat,
        min_lon=ue_tracks_params.min_lon,
        max_lon=ue_tracks_params.max_lon,
        mobility_class_distribution=ue_tracks_params.mobility_class_distribution,
        mobility_class_velocities=ue_tracks_params.mobility_class_velocities,
        mobility_class_velocity_variances=ue_tracks_params.mobility_class_velocity_variances,
    ):
        ue_tracks_generation = pd.concat([ue_tracks_generation, batch_df], ignore_index=True)

    return ue_tracks_generation


def save_ue_data_from_params_json(params_json: str = "params.json", out: str = "ue_tracks.csv") -> pd.DataFrame:
    params_path = Path(params_json)
    if not params_path.exists():
        params_path = Path(__file__).with_name("params_example.json")

    with open(params_path, "r") as f:
        params = json.load(f)

    df = get_ue_data(params)
    out_path = Path(out)
    df.to_csv(out_path, index=False)
    return df

