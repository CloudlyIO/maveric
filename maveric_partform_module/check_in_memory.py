from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")  # headless

try:
    # When executed as module: python -m maveric_partform_module.check_in_memory
    from .utils import topology_gen, spatial_traffic_load_gen, plot_gen
except Exception:
    # When executed as script: python maveric_partform_module/check_in_memory.py
    import sys
    pkg_root = Path(__file__).resolve().parent.parent
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    from maveric_partform_module.utils import topology_gen, spatial_traffic_load_gen, plot_gen


def run_check(
    *,
    num_sites: int,
    cells_per_site: int,
    days: int,
    num_ues_per_tick: int,
    plot_max_ticks: int,
    save_plot: Path | None,
) -> int:
    # 1) Topology + Config
    topo_df, cfg_df, dummy_df = topology_gen(num_sites=num_sites, cells_per_site=cells_per_site)
    print(f"Topology rows: {len(topo_df)} | Config rows: {len(cfg_df)} | Dummy rows: {len(dummy_df)}")

    # 2) Spatial traffic (uses local JSON defaults)
    per_day = spatial_traffic_load_gen(topology_df=topo_df, days=days, num_ues_per_tick=num_ues_per_tick)
    if not per_day:
        print("No UE data generated.")
        return 1
    print(f"Days generated: {len(per_day)}")

    # Expected rows from local time_params.json
    time_params_path = Path(__file__).resolve().parent / "time_params.json"
    with open(time_params_path, "r") as f:
        time_params = json.load(f)
    expected_per_day = time_params.get("total_ticks", 1) * num_ues_per_tick

    ok_all = True
    for i, day_df in enumerate(per_day):
        rows = len(day_df)
        ok = rows == expected_per_day
        ok_all = ok_all and ok
        print(f"Day {i}: rows={rows} expected={expected_per_day} ok={ok}")

    # 3) Plotting (in-memory)
    figs = plot_gen(
        days=days,
        num_sites=num_sites,
        cells_per_site=cells_per_site,
        num_ues_per_tick=num_ues_per_tick,
        plot_max_ticks=plot_max_ticks,
    )
    print(f"Figures generated: {len(figs)}")

    if save_plot and figs:
        key = sorted(figs.keys())[0]
        out_path = Path(save_plot)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        figs[key].savefig(out_path, bbox_inches="tight")
        print(f"Saved sample figure for (day,tick)={key} to: {out_path}")

    # Cleanup figures
    for fig in figs.values():
        try:
            fig.clf()
        except Exception:
            pass

    return 0 if ok_all else 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick CLI check for in-memory generators")
    p.add_argument("--num-sites", type=int, default=3)
    p.add_argument("--cells-per-site", type=int, default=3)
    p.add_argument("--days", type=int, default=1)
    p.add_argument("--num-ues", type=int, default=50, help="UEs per tick")
    p.add_argument("--plot-max-ticks", type=int, default=2)
    p.add_argument("--save-plot", type=Path, default=None, help="Optional output path to save one sample figure")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    code = run_check(
        num_sites=args.num_sites,
        cells_per_site=args.cells_per_site,
        days=args.days,
        num_ues_per_tick=args.num_ues,
        plot_max_ticks=args.plot_max_ticks,
        save_plot=args.save_plot,
    )
    raise SystemExit(code)
