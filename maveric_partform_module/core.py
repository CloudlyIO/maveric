from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.spatial import Voronoi
except ImportError:
    Voronoi = None  # type: ignore

try:
    from shapely.geometry import Polygon, box, Point
except ImportError as e:
    raise ImportError("Shapely library not found. Install: pip install Shapely") from e


class c:
    CELL_ID = "cell_id"; CELL_LAT = "cell_lat"; CELL_LON = "cell_lon"; CELL_AZ_DEG = "cell_az_deg"
    ECGI = "ecgi"; SITE_ID = "site_id"; CELL_NAME = "cell_name"; ENODEB_ID = "enodeb_id"; TAC = "tac"
    CELL_CARRIER_FREQ_MHZ = "cell_carrier_freq_mhz"; LAT = "lat"; LON = "lon"; CELL_EL_DEG = "cell_el_deg"


logger = logging.getLogger(__name__)


class ScenarioConfigurationGenerator:
    def __init__(self) -> None:
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
        topology_data: List[Dict[str, Any]] = []
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
        config_rows: List[Dict[str, Any]] = []
        default_freq = 2100
        for _, row in topology_df.iterrows():
            config_rows.append({
                c.CELL_ID: row[c.CELL_ID],
                c.CELL_EL_DEG: default_config_params.get(c.CELL_EL_DEG, 12.0),
                c.CELL_CARRIER_FREQ_MHZ: row.get(c.CELL_CARRIER_FREQ_MHZ, default_freq),
            })
        return pd.DataFrame(config_rows)


class TrafficDemandModel:
    def __init__(self) -> None:
        pass

    def _space_boundary(self, cell_topology_data: pd.DataFrame, buffer_percent: float = 0.3) -> Dict[str, float]:
        if cell_topology_data.empty:
            raise ValueError("Cell topology data cannot be empty for space boundary calculation.")
        min_lat = cell_topology_data[c.CELL_LAT].min()
        max_lat = cell_topology_data[c.CELL_LAT].max()
        min_lon = cell_topology_data[c.CELL_LON].min()
        max_lon = cell_topology_data[c.CELL_LON].max()
        lat_range_val = max_lat - min_lat
        lon_range_val = max_lon - min_lon
        lat_buffer = lat_range_val * buffer_percent if lat_range_val > 1e-9 else 0.1
        lon_buffer = lon_range_val * buffer_percent if lon_range_val > 1e-9 else 0.1
        return {
            "min_lon_buffered": max(min_lon - lon_buffer, -180.0),
            "min_lat_buffered": max(min_lat - lat_buffer, -90.0),
            "max_lon_buffered": min(max_lon + lon_buffer, 180.0),
            "max_lat_buffered": min(max_lat + lat_buffer, 90.0),
        }

    def _assign_space_types_to_polygons(self, polygons: List[Polygon], spatial_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        space_types = spatial_params.get("types", [])
        proportions = spatial_params.get("proportions", [])
        num_polygons = len(polygons)

        if not space_types or len(space_types) != len(proportions) or num_polygons == 0:
            logger.error("Invalid 'types'/'proportions' in spatial_params or no polygons to assign. Returning empty list.")
            return []

        counts = np.round(np.array(proportions) * num_polygons).astype(int)
        diff = num_polygons - counts.sum()
        if diff != 0:
            counts[np.argmax(counts)] += diff

        type_assignments: List[str] = []
        for i, stype in enumerate(space_types):
            type_assignments.extend([stype] * counts[i])

        rng = np.random.default_rng(42)
        rng.shuffle(polygons)

        spatial_cells_with_types: List[Dict[str, Any]] = []
        for i, poly in enumerate(polygons):
            assigned_type = type_assignments[i]
            spatial_cells_with_types.append({
                "bounds": list(poly.exterior.coords), "type": assigned_type, "cell_id": i,
                "area_sq_km": float(poly.area)
            })
        return spatial_cells_with_types

    def generate_spatial_layout(self, topology_df: pd.DataFrame, spatial_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        if topology_df is None or topology_df.empty:
            logger.error("Topology DataFrame is empty or None. Cannot generate spatial layout.")
            return []
        if Voronoi is None:
            logger.error("SciPy Voronoi not available. Cannot compute spatial layout.")
            return []

        points = topology_df[[c.CELL_LON, c.CELL_LAT]].values
        if len(points) < 2:
            logger.error("Need at least 2 cell sites to compute Voronoi layout.")
            return []

        bounds = self._space_boundary(topology_df)
        minx, miny, maxx, maxy = bounds["min_lon_buffered"], bounds["min_lat_buffered"], bounds["max_lon_buffered"], bounds["max_lat_buffered"]
        bbox = box(minx, miny, maxx, maxy)

        padding = np.array([[minx, miny], [minx, maxy], [maxx, miny], [maxx, maxy]])
        aug_points = np.vstack([points, padding])
        vor = Voronoi(aug_points)

        valid_polygons: List[Polygon] = []
        num_original_points = len(points)
        for i in range(num_original_points):
            region_idx = vor.point_region[i]
            region_vertices_indices = vor.regions[region_idx]
            if -1 in region_vertices_indices or not region_vertices_indices:
                continue
            try:
                polygon = Polygon(vor.vertices[region_vertices_indices])
                clipped_poly = polygon.intersection(bbox)
                if not clipped_poly.is_empty and clipped_poly.area > 1e-12:
                    valid_polygons.append(clipped_poly)
            except Exception:
                continue

        if not valid_polygons:
            logger.error("No valid spatial cells created after Voronoi processing.")
            return []

        return self._assign_space_types_to_polygons(valid_polygons, spatial_params)

    def distribute_ues_over_time(self, spatial_layout: List[Dict[str, Any]], time_params: Dict[str, Any], num_ues_per_tick: int) -> Dict[int, pd.DataFrame]:
        if not spatial_layout:
            logger.error("Spatial layout is empty. Cannot distribute UEs.")
            return {}
        total_ticks = time_params.get("total_ticks", 1)
        time_weights_all_types = time_params.get("time_weights", {})

        cells_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for cell_info in spatial_layout:
            cells_by_type.setdefault(cell_info['type'], []).append(cell_info)

        all_ue_data_per_tick: Dict[int, pd.DataFrame] = {}
        rng = np.random.default_rng(0)
        for tick in range(total_ticks):
            tick_ue_data_list: List[Dict[str, Any]] = []
            current_ue_id_in_tick = 0

            tick_type_weights = {stype: time_weights_all_types.get(stype, [0]*total_ticks)[tick] for stype in cells_by_type.keys()}
            total_weight_this_tick = sum(tick_type_weights.values())
            if total_weight_this_tick <= 1e-9:
                all_ue_data_per_tick[tick] = pd.DataFrame(columns=['ue_id', c.LON, c.LAT, 'tick', 'space_type'])
                continue

            ue_counts = {stype: (w / total_weight_this_tick) * num_ues_per_tick for stype, w in tick_type_weights.items()}
            int_ue_counts = {stype: int(count) for stype, count in ue_counts.items()}
            remainder = num_ues_per_tick - sum(int_ue_counts.values())
            if remainder > 0:
                sorted_types = sorted(ue_counts, key=lambda k: ue_counts[k] - int_ue_counts[k], reverse=True)
                for i in range(remainder):
                    int_ue_counts[sorted_types[i % len(sorted_types)]] += 1

            for space_type, num_ues_for_type in int_ue_counts.items():
                available_cells_for_this_type = cells_by_type.get(space_type, [])
                if not available_cells_for_this_type:
                    continue
                for _ in range(num_ues_for_type):
                    chosen_cell_info = available_cells_for_this_type[rng.integers(0, len(available_cells_for_this_type))]
                    polygon_boundary = Polygon(chosen_cell_info['bounds'])
                    min_x, min_y, max_x, max_y = polygon_boundary.bounds
                    for _attempt in range(100):
                        x = rng.uniform(min_x, max_x)
                        y = rng.uniform(min_y, max_y)
                        if polygon_boundary.contains(Point(x, y)):
                            tick_ue_data_list.append({'ue_id': current_ue_id_in_tick, c.LON: x, c.LAT: y, 'tick': tick, 'space_type': space_type})
                            current_ue_id_in_tick += 1
                            break

            all_ue_data_per_tick[tick] = pd.DataFrame(tick_ue_data_list) if tick_ue_data_list else pd.DataFrame(columns=['ue_id', c.LON, c.LAT, 'tick', 'space_type'])
        return all_ue_data_per_tick

    def generate_traffic_demand(self, topology_df: pd.DataFrame, spatial_params: Dict[str, Any], time_params: Dict[str, Any], num_ues_per_tick: int):
        spatial_layout = self.generate_spatial_layout(topology_df, spatial_params)
        if not spatial_layout:
            return {}, []
        ue_data_per_tick = self.distribute_ues_over_time(spatial_layout, time_params, num_ues_per_tick)
        return ue_data_per_tick, spatial_layout

