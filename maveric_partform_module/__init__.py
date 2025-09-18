"""
Modular traffic generation utilities for Maveric platform (in-memory only).

This package provides decoupled, write-free utilities:
- topology_gen.topology_gen() -> [topology_df, config_df, dummy_training_df]
- spatial_traffic_load_gen.spatial_traffic_load_gen() -> [ue_data_day_1_df, ue_data_day_2_df, ...]
- plot_gen.plot_gen() -> dict of figures keyed by (day, tick)

All functions avoid writing to disk and return pandas DataFrames or matplotlib Figure objects.
"""

from .topology_gen import topology_gen  # noqa: F401
from .spatial_traffic_load_gen import spatial_traffic_load_gen  # noqa: F401
from .plot_gen import plot_gen  # noqa: F401

__all__ = [
    "topology_gen",
    "spatial_traffic_load_gen",
    "plot_gen",
]
