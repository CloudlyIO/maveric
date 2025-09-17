#!/usr/bin/env python3
import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import numpy as np

from .config_gen import ScenarioConfigurationGenerator
from .traffic_demand_simulation import TrafficDemandModel
from .plot_gen import TrafficDataVisualizer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_traffic_data(
    *,
    days: int = 2,
    generate_config_flag: bool = False,
    num_sites: int = 5,
    cells_per_site: int = 3,
    lat_range: Tuple[float, float] = (40.7, 40.8),
    lon_range: Tuple[float, float] = (-74.05, -73.95),
    default_cell_tilt: float = 12.0,
    output_dir: str = "./generated_data",
    topology_csv: str = "topology.csv",
    config_csv: str = "config.csv",
    dummy_training_csv: str = "dummy_ue_training_data.csv",
    spatial_params_json: str = "spatial_params.json",
    time_params_json: str = "time_params.json",
    ue_data_dir: str = "ue_data_per_tick",
    plot_dir: str = "plots",
    num_ues: int = 300,
    generate_dummy_training_flag: bool = False,
    num_training_samples: int = 6000,
    generate_plots_flag: bool = False,
    plot_max_ticks: int = 5,
) -> Dict[str, Any]:
    base_dir = Path(__file__).resolve().parent
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    topology_csv_path = output_dir_path / topology_csv
    config_csv_path = output_dir_path / config_csv
    dummy_training_csv_path = output_dir_path / dummy_training_csv

    spatial_params_path = Path(spatial_params_json)
    if not spatial_params_path.exists():
        spatial_params_path = base_dir / "spatial_params.json"
    time_params_path = Path(time_params_json)
    if not time_params_path.exists():
        time_params_path = base_dir / "time_params.json"
    with open(spatial_params_path, "r") as f:
        spatial_params = json.load(f)
    with open(time_params_path, "r") as f:
        time_params = json.load(f)

    config_generator = ScenarioConfigurationGenerator()
    traffic_model = TrafficDemandModel()
    visualizer = TrafficDataVisualizer()

    if generate_config_flag or not topology_csv_path.exists():
        default_cfg_params = {"cell_el_deg": default_cell_tilt}
        topology_df, _ = config_generator.generate_topology_and_config_files(
            num_sites=num_sites,
            cells_per_site=cells_per_site,
            lat_range=tuple(lat_range),
            lon_range=tuple(lon_range),
            default_config_params=default_cfg_params,
            output_topology_path=str(topology_csv_path),
            output_config_path=str(config_csv_path),
        )
    else:
        topology_df = pd.read_csv(topology_csv_path)

    if topology_df is None or topology_df.empty:
        logger.error("Topology data is not available.")
        return {"topology_df": pd.DataFrame(), "spatial_layout": [], "per_day_files": {}, "output_dir": output_dir_path}

    spatial_layout = traffic_model.generate_spatial_layout(topology_df=topology_df, spatial_params=spatial_params)
    if not spatial_layout:
        logger.error("Failed to generate the spatial layout. Cannot proceed.")
        return {"topology_df": topology_df, "spatial_layout": [], "per_day_files": {}, "output_dir": output_dir_path}

    per_day_files: Dict[int, Dict[int, Path]] = {}
    all_ue_dfs_for_training = []
    for day in range(days):
        day_output_dir = output_dir_path / f"Day_{day}"
        day_ue_data_dir = day_output_dir / Path(ue_data_dir).name
        day_plot_dir = day_output_dir / Path(plot_dir).name
        day_ue_data_dir.mkdir(parents=True, exist_ok=True)
        day_plot_dir.mkdir(parents=True, exist_ok=True)

        ue_data_per_tick_dict = traffic_model.distribute_ues_over_time(
            spatial_layout=spatial_layout, time_params=time_params, num_ues_per_tick=num_ues
        )
        per_day_files[day] = {}
        for tick, ue_df in ue_data_per_tick_dict.items():
            if ue_df.empty:
                continue
            ue_df_with_day = ue_df.copy(); ue_df_with_day['day'] = day
            all_ue_dfs_for_training.append(ue_df_with_day)
            out_csv = day_ue_data_dir / f"ue_data_tick_{tick}.csv"
            ue_df.to_csv(out_csv, index=False)
            per_day_files[day][tick] = out_csv

        if generate_plots_flag:
            ticks_to_plot = sorted(ue_data_per_tick_dict.keys())
            if plot_max_ticks > 0 and len(ticks_to_plot) > plot_max_ticks:
                sel = np.linspace(0, len(ticks_to_plot) - 1, plot_max_ticks, dtype=int)
                ticks_to_plot = [ticks_to_plot[i] for i in sel]
            for tick in ticks_to_plot:
                ue_df_plot = ue_data_per_tick_dict.get(tick)
                if ue_df_plot is not None and not ue_df_plot.empty:
                    plot_path_template = str(day_plot_dir / "ue_distribution_tick_{tick}.png")
                    visualizer.generate_tick_visualization(
                        tick_to_plot=tick,
                        ue_data_for_tick=ue_df_plot,
                        topology_df=topology_df,
                        spatial_layout=spatial_layout,
                        spatial_params_for_colors=spatial_params,
                        plot_output_path_template=plot_path_template,
                    )

    training_path: Path | None = None
    if generate_dummy_training_flag and all_ue_dfs_for_training:
        combined_ue_data = pd.concat(all_ue_dfs_for_training, ignore_index=True)
        if not combined_ue_data.empty:
            config_generator.generate_dummy_training_data(
                topology_df=topology_df,
                ue_data_all_ticks=combined_ue_data,
                num_training_samples=num_training_samples,
                output_training_data_path=str(dummy_training_csv_path),
            )
            training_path = dummy_training_csv_path

    return {
        "topology_df": topology_df,
        "spatial_layout": spatial_layout,
        "per_day_files": per_day_files,
        "dummy_training_csv": training_path,
        "output_dir": output_dir_path,
    }

