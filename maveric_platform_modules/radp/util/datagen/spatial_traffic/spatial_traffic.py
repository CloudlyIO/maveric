from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.spatial import Voronoi
except Exception as _e:  # pragma: no cover - import-time guard
    Voronoi = None  # type: ignore

try:
    from shapely.geometry import Polygon, box, Point
except Exception as _e:  # pragma: no cover - import-time guard
    Polygon = None  # type: ignore
    box = None  # type: ignore
    Point = None  # type: ignore

if TYPE_CHECKING:  # only for static type checkers; no runtime import
    from shapely.geometry import Polygon as ShapelyPolygon


class c:
    CELL_ID = "cell_id"; CELL_LAT = "cell_lat"; CELL_LON = "cell_lon"; LAT = "lat"; LON = "lon"


@dataclass
class TrafficDemandModel:
    """Self-contained traffic model that builds a spatial layout and UE distributions.
    In-memory only; no file I/O and no plotting.
    """

    def _space_boundary(self, cell_topology_data: pd.DataFrame, buffer_percent: float = 0.3) -> Dict[str, float]:
        if cell_topology_data.empty:
            raise ValueError("Cell topology data cannot be empty for space boundary calculation.")
        min_lat = float(cell_topology_data[c.CELL_LAT].min())
        max_lat = float(cell_topology_data[c.CELL_LAT].max())
        min_lon = float(cell_topology_data[c.CELL_LON].min())
        max_lon = float(cell_topology_data[c.CELL_LON].max())
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

    def _assign_space_types_to_polygons(self, polygons: List['ShapelyPolygon'], spatial_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        space_types = spatial_params.get("types", [])
        proportions = spatial_params.get("proportions", [])
        num_polygons = len(polygons)

        if not space_types or len(space_types) != len(proportions) or num_polygons == 0:
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
            return []
        if Voronoi is None or Polygon is None or box is None:
            raise ImportError("scipy.spatial.Voronoi and shapely are required for spatial layout generation.")

        points = topology_df[[c.CELL_LON, c.CELL_LAT]].values
        if len(points) < 2:
            return []

        bounds = self._space_boundary(topology_df)
        minx, miny, maxx, maxy = bounds["min_lon_buffered"], bounds["min_lat_buffered"], bounds["max_lon_buffered"], bounds["max_lat_buffered"]
        bbox = box(minx, miny, maxx, maxy)

        padding = np.array([[minx, miny], [minx, maxy], [maxx, miny], [maxx, maxy]])
        aug_points = np.vstack([points, padding])
        vor = Voronoi(aug_points)

        valid_polygons: List['ShapelyPolygon'] = []
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
            return []

        return self._assign_space_types_to_polygons(valid_polygons, spatial_params)

    def distribute_ues_over_time(
        self,
        *,
        spatial_layout: List[Dict[str, Any]],
        time_params: Dict[str, Any],
        num_ues_per_tick: int,
        rng_seed: Optional[int] = 0,
    ) -> Dict[int, pd.DataFrame]:
        if not spatial_layout:
            return {}
        total_ticks = int(time_params.get("total_ticks", 1))
        time_weights_all_types: Dict[str, List[float]] = time_params.get("time_weights", {})

        cells_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for cell_info in spatial_layout:
            cells_by_type.setdefault(cell_info['type'], []).append(cell_info)

        all_ue_data_per_tick: Dict[int, pd.DataFrame] = {}
        rng = np.random.default_rng(rng_seed)
        for tick in range(total_ticks):
            tick_ue_data_list: List[Dict[str, Any]] = []
            current_ue_id_in_tick = 0

            tick_type_weights = {stype: time_weights_all_types.get(stype, [0] * total_ticks)[tick] for stype in cells_by_type.keys()}
            total_weight_this_tick = float(sum(tick_type_weights.values()))
            if total_weight_this_tick <= 1e-9:
                all_ue_data_per_tick[tick] = pd.DataFrame(columns=['ue_id', c.LON, c.LAT, 'tick', 'space_type'])
                continue

            ue_counts = {stype: (w / total_weight_this_tick) * num_ues_per_tick for stype, w in tick_type_weights.items()}
            int_ue_counts = {stype: int(count) for stype, count in ue_counts.items()}
            remainder = num_ues_per_tick - sum(int_ue_counts.values())
            if remainder > 0 and len(int_ue_counts) > 0:
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


def spatial_traffic_load_gen(
    *,
    topology_df: pd.DataFrame,
    days: int = 2,
    spatial_params: Dict[str, Any] = None,
    time_params: Dict[str, Any] = None,
    num_ues: int = 300,
    rng_seed: Optional[int] = 0,
) -> List[pd.DataFrame]:
    """Generate per-day UE datasets, in-memory only.

    Returns a list of DataFrames per day: [ue_data_day_1, ue_data_day_2, ...]. Each
    DataFrame contains columns: [ue_id, lon, lat, tick, space_type, day].
    """
    if spatial_params is None:
        spatial_params = {"types": ["dense", "suburban", "rural"], "proportions": [0.4, 0.4, 0.2]}
    if time_params is None:
        time_params = {
            "total_ticks": 24,
            "time_weights": {
                "dense":   [1,1,1,1,1,1,2,3,4,5,6,6,5,5,6,7,8,8,7,6,5,4,3,2],
                "suburban": [1,1,1,1,1,1,1,2,3,4,4,4,4,4,4,5,5,5,4,3,2,2,1,1],
                "rural":    [1,1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,2,2,2,1,1],
            },
        }

    model = TrafficDemandModel()
    spatial_layout = model.generate_spatial_layout(topology_df=topology_df, spatial_params=spatial_params)
    if not spatial_layout:
        return []

    per_day_dfs: List[pd.DataFrame] = []
    for day_idx in range(days):
        day = day_idx + 1
        ue_data_per_tick = model.distribute_ues_over_time(
            spatial_layout=spatial_layout,
            time_params=time_params,
            num_ues_per_tick=num_ues,
            rng_seed=(None if rng_seed is None else (rng_seed + day_idx)),
        )
        # Flatten ticks into a single per-day DataFrame
        frames = []
        for tick, df in ue_data_per_tick.items():
            if df is not None and not df.empty:
                df2 = df.copy()
                df2['day'] = day
                frames.append(df2)
        per_day_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['ue_id', c.LON, c.LAT, 'tick', 'space_type', 'day'])
        per_day_dfs.append(per_day_df)

    return per_day_dfs


def load_default_spatial_params() -> Dict[str, Any]:
    p = Path(__file__).resolve().parent / "lib" / "spatial_params.json"
    if p.exists():
        import json
        with open(p, "r") as f:
            return json.load(f)
    return {"types": ["dense", "suburban", "rural"], "proportions": [0.4, 0.4, 0.2]}


def load_default_time_params() -> Dict[str, Any]:
    p = Path(__file__).resolve().parent / "lib" / "time_params.json"
    if p.exists():
        import json
        with open(p, "r") as f:
            return json.load(f)
    return {
        "total_ticks": 24,
        "time_weights": {
            "dense":   [1,1,1,1,1,1,2,3,4,5,6,6,5,5,6,7,8,8,7,6,5,4,3,2],
            "suburban": [1,1,1,1,1,1,1,2,3,4,4,4,4,4,4,5,5,5,4,3,2,2,1,1],
            "rural":    [1,1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,2,2,2,1,1],
        },
    }

