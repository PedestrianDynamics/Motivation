"""Waiting-phase measures of a CROMA run or its replica, at door opening.

Measures (revision/croma_replica_plan.md, amendment):
  rho_entry_rank   Spearman(entry order, distance rank to the door at
                   opening); 1 means nobody overtook while waiting,
  rho_rank_area    Spearman(rank at opening, mean Voronoi area over the
                   waiting phase), cutoff 1 m,
  depth_med        median distance behind the front-most person at opening,
  frac_front3m     fraction of people within 3 m of the front at opening.
Experiment and simulation use this code with the same walkable area.

  python scripts/croma_waiting.py --output build/croma_waiting.csv
computes the experimental values for all CROMA runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pedpy
import shapely
from pedpy import Cutoff, compute_individual_voronoi_polygons
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from croma_ordering import condition
from experimental_rank_area import load_walkable_area

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOOR = (-0.02, 20.0)
CROMA_FPS = 50.0
FRONT_BAND = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--traj-dir", type=Path, default=PROJECT_ROOT.parent / "trajectories_croma")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_experiment(path: Path) -> tuple[pd.DataFrame, int]:
    """Trajectory (id, frame, x, y) and the frame at which the door opens."""
    df = pd.read_csv(path, sep="\t", names=["id", "frame", "x", "y", "z", "m"], comment="#")
    open_frame = int(df.loc[df["y"] >= DOOR[1], "frame"].min())
    return df[["id", "frame", "x", "y"]], open_frame


def waiting_measures(df: pd.DataFrame, fps: float, open_frame: int,
                     walkable: pedpy.WalkableArea) -> dict:
    waiting = df[df["frame"] <= open_frame]
    waiting = waiting[shapely.contains_xy(walkable.polygon, waiting["x"].to_numpy(), waiting["y"].to_numpy())]
    entry = waiting.groupby("id")["frame"].min()
    at_open = waiting[waiting["frame"] == open_frame].set_index("id")
    distance = np.hypot(at_open["x"] - DOOR[0], at_open["y"] - DOOR[1])
    rank = distance.rank(method="first")
    cells = compute_individual_voronoi_polygons(
        traj_data=pedpy.TrajectoryData(waiting.copy(), frame_rate=fps),
        walkable_area=walkable, cut_off=Cutoff(radius=1.0, quad_segments=3),
    )
    area = cells.assign(area=cells["polygon"].apply(lambda p: p.area)).groupby("id")["area"].mean()
    depth = at_open["y"].max() - at_open["y"]
    return {
        "n_at_open": len(at_open),
        "rho_entry_rank": float(spearmanr(entry.reindex(rank.index), rank).statistic),
        "rho_rank_area": float(spearmanr(rank, area.reindex(rank.index)).statistic),
        "depth_med": float(depth.median()),
        "frac_front3m": float((depth <= FRONT_BAND).mean()),
    }


def main() -> None:
    args = parse_args()
    walkable = load_walkable_area(None)
    rows = []
    for path in sorted(args.traj_dir.glob("*_Combined.txt")):
        df, open_frame = load_experiment(path)
        rows.append({"run": path.name[:5], "condition": condition(path),
                     "door_open_s": (open_frame - df["frame"].min()) / CROMA_FPS,
                     **waiting_measures(df, CROMA_FPS, open_frame, walkable)})
    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
