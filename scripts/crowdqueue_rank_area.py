"""Rank-area correlation in the CrowdQueue 2018 entrance experiments.

The CrowdQueue runs (Adrian et al. 2020, doi:10.34735/ped.2018.1) vary
corridor width (1.2-5.6 m) and motivation (h0 high, h- low, h+ one run)
in front of a 0.5 m gate. This script applies the paper's rank-area
measure (as for the CROMA runs in experimental_rank_area.py) to them:

  rank  crossing order at the flow line y = -0.6 m inside the gate
        passage (flow is in -y),
  area  mean individual Voronoi area (cutoff 1 m), either over all
        tracked frames (as in the paper) or only over the frames before
        the agent crosses.

The walkable area is the archive polygon clipped to the corridor, the
gate and the area behind it, as in jupedsim-dakota-calibration
(crowdqueue/scenario_cq.py): the archive polygon is a 16 m box in which
the space beside the barriers is walkable. Points outside the clipped
area are dropped and counted.

Runs 010, 080 and 200 are marked "trajectories are not corrected" in
the archive; they are analysed but flagged.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import pandas as pd
import pedpy
import shapely
from pedpy import Cutoff, compute_individual_voronoi_polygons
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLOW_LINE_Y = -0.6
UNCORRECTED = {"010", "080", "200"}
RUN_RE = re.compile(r"^(?P<id>\d+)_(?P<priming>[cq])_(?P<width>\d+)_(?P<motivation>h[0+-])$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with the archive *.h5 files.")
    parser.add_argument("--cutoff", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "build" / "crowdqueue_rank_area")
    return parser.parse_args()


def clipped_walkable_area(h5_file: Path, width: float) -> shapely.Polygon:
    geo = shapely.from_wkt(h5py.File(h5_file).attrs["wkt_geometry"])
    half = width / 2 + 0.15  # cut through the 0.25 m barriers, not along their faces
    clipped = geo.intersection(shapely.box(-half, -4.0, half, 12.0))
    parts = [g for g in getattr(clipped, "geoms", [clipped]) if g.geom_type == "Polygon"]
    return max(parts, key=lambda g: g.area)


def inside(data: pd.DataFrame, area: shapely.Polygon) -> pd.Series:
    return pd.Series(shapely.contains_xy(area, data["x"].to_numpy(), data["y"].to_numpy()), index=data.index)


def crossing_frames(data: pd.DataFrame) -> pd.Series:
    crossed = data[data["y"] <= FLOW_LINE_Y]
    return crossed.groupby("id")["frame"].min()


def analyse_run(h5_file: Path, meta: dict, cutoff: float) -> tuple[dict, pd.DataFrame]:
    traj = pedpy.load_trajectory_from_ped_data_archive_hdf5(h5_file)
    return rank_area(traj, clipped_walkable_area(h5_file, meta["width"]), meta, cutoff)


def rank_area(
    traj: pedpy.TrajectoryData, area: shapely.Polygon, meta: dict, cutoff: float
) -> tuple[dict, pd.DataFrame]:
    """Rank-area measures of one run; used for experiment and simulation."""
    data = traj.data
    keep = inside(data, area)
    data = data[keep]
    cells = compute_individual_voronoi_polygons(
        traj_data=pedpy.TrajectoryData(data[["id", "frame", "x", "y"]].copy(), frame_rate=traj.frame_rate),
        walkable_area=pedpy.WalkableArea(area),
        cut_off=Cutoff(radius=cutoff, quad_segments=3),
    )
    cells["area"] = cells["polygon"].apply(lambda poly: poly.area)
    cross = crossing_frames(data)
    cells = cells.join(cross.rename("crossing_frame"), on="id", how="inner")
    agents = pd.DataFrame({
        "crossing_frame": cross,
        "area_all": cells.groupby("id")["area"].mean(),
        "area_before": cells[cells["frame"] < cells["crossing_frame"]].groupby("id")["area"].mean(),
    }).dropna(subset=["area_all"])
    agents["rank"] = agents["crossing_frame"].rank(method="first")
    summary = {
        **meta,
        "n_tracked": int(traj.data["id"].nunique()),
        "n_crossed": len(agents),
        "dropped_points": int((~keep).sum()),
        "dropped_fraction": float((~keep).mean()),
    }
    for col in ("area_all", "area_before"):
        valid = agents[["rank", col]].dropna()
        summary[f"rho_{col}"] = float(spearmanr(valid["rank"], valid[col]).statistic)
        summary[f"n_{col}"] = len(valid)
    agents = agents.reset_index().assign(run=meta["run"])
    return summary, agents


def run_meta(h5_file: Path) -> dict | None:
    match = RUN_RE.match(h5_file.stem)
    if match is None:
        return None
    return {
        "run": h5_file.stem,
        "id": match["id"],
        "priming": "queuing" if match["priming"] == "q" else "crowding",
        "width": int(match["width"]) / 10,
        "motivation": match["motivation"],
        "uncorrected": match["id"] in UNCORRECTED,
    }


def main() -> None:
    args = parse_args()
    files = sorted(args.data_dir.glob("*.h5"))
    if not files:
        raise SystemExit(f"No *.h5 files in {args.data_dir}")
    summaries, details = [], []
    for h5_file in files:
        meta = run_meta(h5_file)
        if meta is None:
            continue
        print(f"{meta['run']}")
        summary, agents = analyse_run(h5_file, meta, args.cutoff)
        summaries.append(summary)
        details.append(agents)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pd.concat(details).to_csv(args.output_dir / "details.csv", index=False)
    cols = ["run", "width", "motivation", "n_tracked", "n_crossed", "dropped_fraction",
            "rho_area_all", "rho_area_before", "uncorrected"]
    print(summary[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
