from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Local core implementations (no artifact dependency)
from .core import ScenarioConfigurationGenerator as _Cfg
from .core import c as _C


def _generate_dummy_training_data(
    topology_df: pd.DataFrame,
    ue_data_all_ticks: pd.DataFrame,
    *,
    num_training_samples: int = 12000,
    possible_tilts: Optional[List[float]] = None,
    assumed_optimal_tilt: float = 8.0,
    tilt_penalty_factor: float = 0.5,
) -> pd.DataFrame:
    """In-memory clone of artifact.trafficgen.config_gen.generate_dummy_training_data without writing.

    Returns an empty DataFrame if inputs are insufficient.
    """
    if possible_tilts is None:
        possible_tilts = list(np.arange(0.0, 21.0, 1.0))

    COL_CELL_ID = _C.CELL_ID
    COL_CELL_EL_DEG = _C.CELL_EL_DEG
    COL_LAT = _C.LAT
    COL_LON = _C.LON
    COL_CELL_LAT = _C.CELL_LAT
    COL_CELL_LON = _C.CELL_LON

    required_topo_cols = [COL_CELL_ID, COL_CELL_LAT, COL_CELL_LON]
    if not all(col in topology_df.columns for col in required_topo_cols):
        return pd.DataFrame()

    required_ue_cols = [COL_LAT, COL_LON, 'ue_id', 'tick']
    if ue_data_all_ticks is None or ue_data_all_ticks.empty or not all(col in ue_data_all_ticks.columns for col in required_ue_cols):
        return pd.DataFrame()

    dummy_training_data_list: List[Dict[str, Any]] = []
    assumed_cell_txpwr_val = 25.0
    path_loss_exponent = 3.5

    num_cells = len(topology_df[COL_CELL_ID].unique())
    if num_cells == 0:
        return pd.DataFrame()

    num_unique_ue_tick_pairs_to_sample = max(1, int(round(num_training_samples / num_cells)))
    num_available_ue_tick_pairs = len(ue_data_all_ticks)
    if num_unique_ue_tick_pairs_to_sample > num_available_ue_tick_pairs:
        num_unique_ue_tick_pairs_to_sample = num_available_ue_tick_pairs
    if num_unique_ue_tick_pairs_to_sample == 0:
        return pd.DataFrame()

    sampled_ue_data = ue_data_all_ticks.sample(n=num_unique_ue_tick_pairs_to_sample, replace=False, random_state=42)
    for _, ue_row in sampled_ue_data.iterrows():
        ue_lat_val = ue_row[COL_LAT]
        ue_lon_val = ue_row[COL_LON]
        for _, cell_row in topology_df.iterrows():
            cell_lat_val = cell_row[COL_CELL_LAT]
            cell_lon_val = cell_row[COL_CELL_LON]
            try:
                dist_km = np.sqrt((ue_lat_val - cell_lat_val) ** 2 + (ue_lon_val - cell_lon_val) ** 2) * 111
                dist_m = max(1.0, dist_km * 1000.0)
                effective_start_power = assumed_cell_txpwr_val - 40
                simple_rsrp = effective_start_power - 10 * path_loss_exponent * np.log10(dist_m)
            except Exception:
                simple_rsrp = assumed_cell_txpwr_val - 140

            random_tilt = float(np.random.choice(possible_tilts))
            tilt_deviation = abs(random_tilt - assumed_optimal_tilt)
            rsrp_penalty = tilt_deviation * tilt_penalty_factor
            adjusted_rsrp = simple_rsrp - rsrp_penalty

            dummy_training_data_list.append({
                COL_CELL_ID: cell_row[COL_CELL_ID],
                "avg_rsrp": adjusted_rsrp,
                COL_LON: ue_lon_val,
                COL_LAT: ue_lat_val,
                COL_CELL_EL_DEG: random_tilt,
            })

    return pd.DataFrame(dummy_training_data_list)


def topology_gen(
    *,
    num_sites: int = 5,
    cells_per_site: int = 3,
    lat_range: Tuple[float, float] = (40.7, 40.8),
    lon_range: Tuple[float, float] = (-74.05, -73.95),
    default_cell_tilt: float = 12.0,
    # Dummy training generation is optional and in-memory only
    generate_dummy_training: bool = False,
    ue_data_for_training: Optional[pd.DataFrame] = None,
    num_training_samples: int = 6000,
    possible_tilts: Optional[List[float]] = None,
    assumed_optimal_tilt: float = 8.0,
    tilt_penalty_factor: float = 0.5,
) -> List[pd.DataFrame]:
    """Generate topology and initial config DataFrames without writing to disk.

    Returns [topology_df, config_df, dummy_training_df].
    If generate_dummy_training is False or training inputs are missing, dummy_training_df is empty.
    """
    cfg = _Cfg()
    topology_df = cfg._generate_dummy_topology_df(
        num_sites=num_sites,
        cells_per_site=cells_per_site,
        lat_range=lat_range,
        lon_range=lon_range,
    )
    config_df = cfg._generate_initial_config_df(topology_df, default_config_params={_C.CELL_EL_DEG: default_cell_tilt})

    dummy_training_df = pd.DataFrame()
    if generate_dummy_training:
        dummy_training_df = _generate_dummy_training_data(
            topology_df=topology_df,
            ue_data_all_ticks=ue_data_for_training if ue_data_for_training is not None else pd.DataFrame(),
            num_training_samples=num_training_samples,
            possible_tilts=possible_tilts,
            assumed_optimal_tilt=assumed_optimal_tilt,
            tilt_penalty_factor=tilt_penalty_factor,
        )

    return [topology_df, config_df, dummy_training_df]
