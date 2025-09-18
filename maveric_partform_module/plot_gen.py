from __future__ import annotations

from typing import Any, Dict, List, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import pandas as pd

from shapely.geometry import Polygon

from .core import TrafficDemandModel
from .core import c as _C

from .topology_gen import topology_gen as _topology_gen
from .spatial_traffic_load_gen import spatial_traffic_load_gen as _spatial_gen
from .spatial_traffic_load_gen import _load_params


def _make_space_colors(spatial_layout: List[Dict[str, Any]], spatial_params: Dict[str, Any]) -> Dict[str, Tuple[float, float, float, float]]:
    space_types_defined = spatial_params.get("types", [])
    if not space_types_defined and spatial_layout:
        space_types_defined = sorted(list(set(cell['type'] for cell in spatial_layout)))
    cmap_spaces = plt.get_cmap('tab10', max(1, len(space_types_defined)))
    return {stype: cmap_spaces(i) for i, stype in enumerate(space_types_defined)}


def plot_gen(
    *,
    # Topology params
    num_sites: int = 5,
    cells_per_site: int = 3,
    lat_range: Tuple[float, float] = (40.7, 40.8),
    lon_range: Tuple[float, float] = (-74.05, -73.95),
    default_cell_tilt: float = 12.0,
    # Traffic params
    days: int = 1,
    spatial_params: Dict[str, Any] | str | Path | None = None,
    time_params: Dict[str, Any] | str | Path | None = None,
    num_ues_per_tick: int = 300,
    # Plot control
    plot_max_ticks: int = 5,
) -> Dict[Tuple[int, int], Figure]:
    """Orchestrate in-memory generation and produce matplotlib Figures.

    Returns a dict mapping (day, tick) -> Figure.
    """
    # 1) Generate topology in-memory
    topo_df, _cfg_df, _dummy = _topology_gen(
        num_sites=num_sites,
        cells_per_site=cells_per_site,
        lat_range=lat_range,
        lon_range=lon_range,
        default_cell_tilt=default_cell_tilt,
        generate_dummy_training=False,
    )

    # 2) Compute spatial layout for plotting
    model = TrafficDemandModel()
    # Ensure dicts for params using spatial_traffic_load_gen's helpers would need import; keep simple here
    # Reuse spatial gen to parse files and produce UE data
    per_day = _spatial_gen(
        topology_df=topo_df,
        days=days,
        spatial_params=spatial_params,
        time_params=time_params,
        num_ues_per_tick=num_ues_per_tick,
    )

    # Need spatial_layout for background polygons; recompute here
    # Convert spatial_params/time_params to dicts via TrafficDemandModel usage path
    # spatial_params may be a path or dict; if None, load module-local JSON
    spatial_params_dict = _load_params(spatial_params, local_filename="spatial_params.json")
    spatial_layout = model.generate_spatial_layout(topo_df, spatial_params_dict)

    space_colors = _make_space_colors(spatial_layout, spatial_params_dict)

    figures: Dict[Tuple[int, int], Figure] = {}
    # 3) Create plots per day limited by plot_max_ticks
    for day_idx, day_df in enumerate(per_day):
        if day_df is None or day_df.empty:
            continue
        ticks = sorted(day_df['tick'].unique().tolist())
        if plot_max_ticks > 0 and len(ticks) > plot_max_ticks:
            step = max(1, len(ticks) // plot_max_ticks)
            ticks = ticks[::step][:plot_max_ticks]

        for tick in ticks:
            df_tick = day_df[day_df['tick'] == tick]
            if df_tick.empty:
                continue

            fig, ax = plt.subplots(figsize=(12, 9))
            # draw background spatial polygons
            for cell in spatial_layout:
                poly = Polygon(cell['bounds'])
                x, y = poly.exterior.xy
                color = space_colors.get(cell['type'], (0.8, 0.8, 0.8, 0.3))
                ax.fill(x, y, alpha=0.2, fc=color, ec='gray', linewidth=0.5)

            # legend for space types
            legends_spaces = [Patch(facecolor=color, edgecolor='gray', label=str(st)) for st, color in space_colors.items()]

            # cell towers
            ax.scatter(topo_df[_C.CELL_LON], topo_df[_C.CELL_LAT], marker='^', s=80, c='black', label='Cell Towers')

            # UEs
            ax.scatter(df_tick[_C.LON], df_tick[_C.LAT], color='blue', alpha=0.6, s=10, zorder=5, label="UEs")

            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
            ax.set_title(f"UE Distribution and Spatial Types (Day {day_idx}, Tick {tick})")
            # Compose legend
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small', title="Legend", frameon=True)
            ax.grid(True, linestyle='--', alpha=0.4)
            plt.subplots_adjust(right=0.75)

            figures[(day_idx, int(tick))] = fig

    return figures
