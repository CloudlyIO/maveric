from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def topology_gen(**kwargs) -> Dict[str, Any]:
    from .trafficgen.topology_gen import topology_gen as _topology_gen
    return _topology_gen(**kwargs)


def spatial_traffic_load_gen(**kwargs) -> Dict[str, Any]:
    from .trafficgen.spatial_traffic_load_gen import spatial_traffic_load_gen as _spatial_gen
    return _spatial_gen(**kwargs)


def mro_gen(
    *,
    params: str | Path = "artifact/mro/params_example.json",
    output_dir: str | Path = "artifact/generated_data/mro",
    filename: str = "ue_tracks.csv",
) -> pd.DataFrame:
    from .mro.generate_mro_data import generate_mro_data
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    df = generate_mro_data(params=str(params), out=str(out_path))
    return df
