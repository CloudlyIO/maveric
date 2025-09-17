# Copied from maveric/artifact/trafficgen/config_gen.py

import os
import logging
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import numpy as np


class c:
    CELL_ID = "cell_id"; CELL_LAT = "cell_lat"; CELL_LON = "cell_lon"; CELL_AZ_DEG = "cell_az_deg"
    ECGI = "ecgi"; SITE_ID = "site_id"; CELL_NAME = "cell_name"; ENODEB_ID = "enodeb_id"; TAC = "tac"
    CELL_CARRIER_FREQ_MHZ = "cell_carrier_freq_mhz"; LAT = "lat"; LON = "lon"; CELL_EL_DEG = "cell_el_deg"


logger = logging.getLogger(__name__)


class ScenarioConfigurationGenerator:
    def __init__(self):
        pass

    def _generate_dummy_topology_df(
        self,
        num_sites: int,
        cells_per_site: int = 3,
        lat_range: Tuple[float, float] = (40.7, 40.8),
        lon_range: Tuple[float, float] = (-74.05, -73.95),
        start_ecgi: int = 1001,
        start_enodeb_id: int = 1,
        default_tac: int = 1,
        default_freq: int = 2100,
        azimuth_step: int = 120,
    ) -> pd.DataFrame:
        logger.info(f"Generating dummy topology for {num_sites} sites with {cells_per_site} cells each.")
        topology_data = []
        current_ecgi = start_ecgi
        current_enodeb_id = start_enodeb_id

        for i in range(num_sites):
            site_lat = np.random.uniform(lat_range[0], lat_range[1])
            site_lon = np.random.uniform(lon_range[0], lon_range[1])
            site_id_str = f"Site{i+1}"

            for j in range(cells_per_site):
                cell_az = (j * azimuth_step) % 360
                cell_id_str = f"cell_{current_enodeb_id}_{j}"
                cell_name_str = f"{site_id_str}_Cell{j+1}"
                row = {
                    c.ECGI: current_ecgi,
                    c.SITE_ID: site_id_str,
                    c.CELL_NAME: cell_name_str,
                    c.ENODEB_ID: current_enodeb_id,
                    c.CELL_AZ_DEG: cell_az,
                    c.TAC: default_tac,
                    c.CELL_LAT: site_lat,
                    c.CELL_LON: site_lon,
                    c.CELL_ID: cell_id_str,
                    c.CELL_CARRIER_FREQ_MHZ: default_freq,
                }
                topology_data.append(row)
            current_ecgi += cells_per_site
            current_enodeb_id += 1

        df = pd.DataFrame(topology_data)
        column_order = [
            c.ECGI, c.SITE_ID, c.CELL_NAME, c.ENODEB_ID, c.CELL_AZ_DEG, c.TAC,
            c.CELL_LAT, c.CELL_LON, c.CELL_ID, c.CELL_CARRIER_FREQ_MHZ,
        ]
        return df.reindex(columns=column_order)

    def _generate_initial_config_df(self, topology_df: pd.DataFrame, default_config_params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        if default_config_params is None:
            default_config_params = {c.CELL_EL_DEG: 12.0}
        config_rows = []
        default_freq = 2100
        for _, row in topology_df.iterrows():
            config_rows.append({
                c.CELL_ID: row[c.CELL_ID],
                c.CELL_EL_DEG: default_config_params.get(c.CELL_EL_DEG, 12.0),
                c.CELL_CARRIER_FREQ_MHZ: row.get(c.CELL_CARRIER_FREQ_MHZ, default_freq),
            })
        return pd.DataFrame(config_rows)

    def generate_topology_and_config_files(
        self,
        num_sites: int = 5,
        cells_per_site: int = 3,
        lat_range: Tuple[float, float] = (40.7, 40.8),
        lon_range: Tuple[float, float] = (-74.05, -73.95),
        default_config_params: Optional[Dict[str, Any]] = None,
        output_topology_path: str = "topology.csv",
        output_config_path: str = "config.csv",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        topology_df = self._generate_dummy_topology_df(
            num_sites=num_sites,
            cells_per_site=cells_per_site,
            lat_range=lat_range,
            lon_range=lon_range,
        )
        initial_config_df = self._generate_initial_config_df(topology_df, default_config_params)
        os.makedirs(os.path.dirname(output_topology_path) or '.', exist_ok=True)
        os.makedirs(os.path.dirname(output_config_path) or '.', exist_ok=True)
        topology_df.to_csv(output_topology_path, index=False)
        initial_config_df.to_csv(output_config_path, index=False)
        return topology_df, initial_config_df

    def generate_dummy_training_data(
        self,
        topology_df: pd.DataFrame,
        ue_data_all_ticks: pd.DataFrame,
        num_training_samples: int = 12000,
        possible_tilts: Optional[List[float]] = None,
        assumed_optimal_tilt: float = 8.0,
        tilt_penalty_factor: float = 0.5,
        output_training_data_path: str = "dummy_ue_training_data.csv",
    ) -> pd.DataFrame:
        if possible_tilts is None:
            possible_tilts = list(np.arange(0.0, 21.0, 1.0))
        COL_CELL_ID = c.CELL_ID
        COL_CELL_EL_DEG = c.CELL_EL_DEG
        COL_LAT = c.LAT
        COL_LON = c.LON
        COL_CELL_LAT = c.CELL_LAT
        COL_CELL_LON = c.CELL_LON

        required_topo_cols = [COL_CELL_ID, COL_CELL_LAT, COL_CELL_LON]
        if not all(col in topology_df.columns for col in required_topo_cols):
            raise ValueError("Topology DF missing required columns: " + str(required_topo_cols))

        required_ue_cols = [COL_LAT, COL_LON, 'ue_id', 'tick']
        if not all(col in ue_data_all_ticks.columns for col in required_ue_cols):
            raise ValueError("UE data DF missing required columns: " + str(required_ue_cols))

        dummy_training_data_list = []
        assumed_cell_txpwr_val = 25.0
        path_loss_exponent = 3.5

        if ue_data_all_ticks.empty:
            return pd.DataFrame()

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

        final_dummy_training_df = pd.DataFrame(dummy_training_data_list)
        if not final_dummy_training_df.empty:
            os.makedirs(os.path.dirname(output_training_data_path) or '.', exist_ok=True)
            final_dummy_training_df.to_csv(output_training_data_path, index=False)
        return final_dummy_training_df

