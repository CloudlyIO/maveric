from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .core import TrafficDemandModel


def _load_params(obj: Dict[str, Any] | str | Path | None, local_filename: str) -> Dict[str, Any]:
    """Load params from dict, explicit path, or the local module JSON file.

    - If `obj` is a dict, return it.
    - If `obj` is a str/Path, read JSON from that path. Raises if not found.
    - If `obj` is None, load from `<module_dir>/<local_filename>`. Raises if not found.
    """
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (str, Path)):
        p = Path(obj)
        if not p.exists():
            raise FileNotFoundError(f"Params file not found: {p}")
        with open(p, "r") as f:
            return json.load(f)
    # None: use local file
    local_p = Path(__file__).resolve().parent / local_filename
    if not local_p.exists():
        raise FileNotFoundError(f"Default params file missing: {local_p}")
    with open(local_p, "r") as f:
        return json.load(f)


def spatial_traffic_load_gen(
    *,
    topology_df: pd.DataFrame,
    days: int = 2,
    spatial_params: Dict[str, Any] | str | Path | None = None,
    time_params: Dict[str, Any] | str | Path | None = None,
    num_ues_per_tick: int = 300,
) -> List[pd.DataFrame]:
    """Generate per-day UE traffic DataFrames without writing to disk.

    Returns a list with one DataFrame per day. Each DataFrame includes columns
    ['ue_id', 'lon', 'lat', 'tick', 'space_type', 'day'].
    """
    if topology_df is None or topology_df.empty:
        return []

    model = TrafficDemandModel()
    spatial_params_dict = _load_params(spatial_params, local_filename="spatial_params.json")
    time_params_dict = _load_params(time_params, local_filename="time_params.json")

    spatial_layout = model.generate_spatial_layout(topology_df=topology_df, spatial_params=spatial_params_dict)
    if not spatial_layout:
        return []

    per_day_results: List[pd.DataFrame] = []
    for day in range(days):
        ue_data_per_tick = model.distribute_ues_over_time(
            spatial_layout=spatial_layout, time_params=time_params_dict, num_ues_per_tick=num_ues_per_tick
        )
        # Combine ticks for the day, annotate day number
        if not ue_data_per_tick:
            per_day_results.append(pd.DataFrame(columns=['ue_id', 'lon', 'lat', 'tick', 'space_type', 'day']))
            continue
        frames = []
        for tick, df in sorted(ue_data_per_tick.items()):
            if df is None or df.empty:
                continue
            df2 = df.copy()
            df2['day'] = day
            frames.append(df2)
        day_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['ue_id', 'lon', 'lat', 'tick', 'space_type', 'day'])
        per_day_results.append(day_df)

    return per_day_results
