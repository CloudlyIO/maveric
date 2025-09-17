from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .generate_traffic_data import generate_traffic_data


def spatial_traffic_load_gen(
    *,
    days: int = 2,
    output_dir: str | Path = "artifact/generated_data/trafficgen",
    topology_csv: str = "topology.csv",
    config_csv: str = "config.csv",
    spatial_params: Dict[str, Any] | str | Path = "artifact/trafficgen/spatial_params.json",
    time_params: Dict[str, Any] | str | Path = "artifact/trafficgen/time_params.json",
    num_ues: int = 300,
    ue_data_dir: str = "ue_data_per_tick",
    plot: bool = False,
    plot_dir: str = "plots",
    plot_max_ticks: int = 5,
    generate_dummy_training: bool = False,
    dummy_training_csv: str = "dummy_ue_training_data.csv",
    num_training_samples: int = 6000,
):
    def _json_path(obj: Dict[str, Any] | str | Path, *, tmp_name: str) -> str:
        if isinstance(obj, (str, Path)):
            return str(obj)
        out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / tmp_name
        with open(p, "w") as f:
            json.dump(obj, f)
        return str(p)

    return generate_traffic_data(
        days=days,
        generate_config_flag=False,
        num_sites=5,
        cells_per_site=3,
        lat_range=(40.7, 40.8),
        lon_range=(-74.05, -73.95),
        default_cell_tilt=12.0,
        output_dir=str(output_dir),
        topology_csv=topology_csv,
        config_csv=config_csv,
        dummy_training_csv=dummy_training_csv,
        spatial_params_json=_json_path(spatial_params, tmp_name="spatial_params.json"),
        time_params_json=_json_path(time_params, tmp_name="time_params.json"),
        ue_data_dir=ue_data_dir,
        plot_dir=plot_dir,
        num_ues=num_ues,
        generate_dummy_training_flag=generate_dummy_training,
        num_training_samples=num_training_samples,
        generate_plots_flag=plot,
        plot_max_ticks=plot_max_ticks,
    )
