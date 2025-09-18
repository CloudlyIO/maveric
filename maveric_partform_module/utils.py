from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .topology_gen import topology_gen as _topology_gen
from .spatial_traffic_load_gen import spatial_traffic_load_gen as _spatial_gen
from .plot_gen import plot_gen as _plot_gen


def topology_gen(**kwargs) -> list[pd.DataFrame]:
    """Convenience wrapper for maveric_partform_module.topology_gen.topology_gen"""
    return _topology_gen(**kwargs)


def spatial_traffic_load_gen(**kwargs) -> list[pd.DataFrame]:
    """Convenience wrapper for maveric_partform_module.spatial_traffic_load_gen.spatial_traffic_load_gen"""
    return _spatial_gen(**kwargs)


def plot_gen(**kwargs) -> Dict[tuple[int, int], Any]:
    """Convenience wrapper for maveric_partform_module.plot_gen.plot_gen"""
    return _plot_gen(**kwargs)

