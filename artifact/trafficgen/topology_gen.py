from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from .config_gen import ScenarioConfigurationGenerator


def topology_gen(
    *,
    num_sites: int = 5,
    cells_per_site: int = 3,
    lat_range: Tuple[float, float] = (40.7, 40.8),
    lon_range: Tuple[float, float] = (-74.05, -73.95),
    default_cell_tilt: float = 12.0,
    output_dir: str | Path = "artifact/generated_data/trafficgen",
    topology_filename: str = "topology.csv",
    config_filename: str = "config.csv",
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    topology_path = out_dir / topology_filename
    config_path = out_dir / config_filename

    cfg = ScenarioConfigurationGenerator()
    topology_df, config_df = cfg.generate_topology_and_config_files(
        num_sites=num_sites,
        cells_per_site=cells_per_site,
        lat_range=lat_range,
        lon_range=lon_range,
        default_config_params={"cell_el_deg": default_cell_tilt},
        output_topology_path=str(topology_path),
        output_config_path=str(config_path),
    )

    return {
        "topology_df": topology_df,
        "config_df": config_df,
        "topology_path": topology_path,
        "config_path": config_path,
    }
