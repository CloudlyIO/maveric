# Copied from maveric/artifact/trafficgen/plot_gen.py

import os
import logging
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    from shapely.geometry import Polygon
except ImportError:
    print("Error: Shapely library not found. Install: pip install Shapely")
    raise


class c:
    CELL_ID = "cell_id"; CELL_LAT = "cell_lat"; CELL_LON = "cell_lon"; LAT = "lat"; LON = "lon"


logger = logging.getLogger(__name__)


class TrafficDataVisualizer:
    def __init__(self):
        pass

    def generate_tick_visualization(
        self,
        tick_to_plot: int,
        ue_data_for_tick: pd.DataFrame,
        topology_df: pd.DataFrame,
        spatial_layout: List[Dict[str, Any]],
        spatial_params_for_colors: Dict[str, Any],
        plot_output_path_template: str = "./plots/ue_distribution_tick_{tick}.png",
        serving_cell_data_for_tick: Optional[pd.DataFrame] = None,
    ):
        COL_LAT = c.LAT; COL_LON = c.LON; COL_CELL_LAT = c.CELL_LAT; COL_CELL_LON = c.CELL_LON; COL_CELL_ID = c.CELL_ID
        if ue_data_for_tick is None or ue_data_for_tick.empty:
            logger.warning(f"No UE data for tick {tick_to_plot}. Skipping plot.")
            return

        fig, ax = plt.subplots(figsize=(12, 9))

        space_types_defined = spatial_params_for_colors.get("types", [])
        if not space_types_defined and spatial_layout:
            space_types_defined = sorted(list(set(cell['type'] for cell in spatial_layout)))
        cmap_spaces = plt.get_cmap('tab10', max(1, len(space_types_defined)))
        space_colors = {stype: cmap_spaces(i) for i, stype in enumerate(space_types_defined)}

        legends_spaces = []
        for cell in spatial_layout:
            poly = Polygon(cell['bounds'])
            x, y = poly.exterior.xy
            color = space_colors.get(cell['type'], (0.8, 0.8, 0.8, 0.3))
            ax.fill(x, y, alpha=0.2, fc=color, ec='gray', linewidth=0.5)
        for stype, color in space_colors.items():
            legends_spaces.append(Patch(facecolor=color, edgecolor='gray', label=f"{stype}"))

        ax.scatter(topology_df[COL_CELL_LON], topology_df[COL_CELL_LAT], marker='^', s=80, c='black', label='Cell Towers')

        ue_plot_data = ue_data_for_tick[[COL_LON, COL_LAT]].copy()
        ax.scatter(ue_plot_data[COL_LON], ue_plot_data[COL_LAT], color='blue', alpha=0.6, s=10, zorder=5, label="UEs")

        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title(f"UE Distribution and Spatial Types (Tick {tick_to_plot})")
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small', title="Legend", frameon=True)
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.subplots_adjust(right=0.75)

        out_file = plot_output_path_template.format(tick=tick_to_plot)
        os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
        plt.savefig(out_file, bbox_inches='tight')
        plt.close(fig)

