"""Ordering and rank-area in the CROMA entrance runs.

Per run (door closed while people arrive, then opened):
  rho_entry_cross        Spearman(order of first observation, crossing order);
                         1 means first come, first served, lower values mean
                         people overtook while waiting,
  rho_pos_at_open_cross  Spearman(distance to the door when it opens, crossing
                         order),
  rho_area_all           the paper's rank-area measure: crossing order vs mean
                         Voronoi area over all frames (cutoff 1 m),
  rho_area_wait          the same, with the area averaged only over frames
                         before the door opens.
The door opens at the first frame in which someone reaches y = 20 m, and
crossing means y >= 20 m, as in experimental_rank_area.py. The motivation
condition (nM/hM) is read from the PeTrack project name in the file header.
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
from experimental_rank_area import load_walkable_area

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOOR = (-0.02, 20.0)
FPS = 50.0  # "# framerate: 50 fps" in every CROMA file header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--traj-dir", type=Path, default=PROJECT_ROOT.parent / "trajectories_croma")
    parser.add_argument("--output", type=Path, required=True, help="CSV file to write.")
    return parser.parse_args()


def condition(path: Path) -> str:
    header = path.read_text(encoding="utf-8", errors="replace")[:400]
    return "hM" if "_hM_" in header else "nM"


def spearman_on_common(a: pd.Series, b: pd.Series) -> float:
    common = a.dropna().index.intersection(b.dropna().index)
    return float(spearmanr(a.loc[common], b.loc[common]).statistic)


def analyse(path: Path, walkable: pedpy.WalkableArea) -> dict:
    df = pd.read_csv(path, sep="\t", names=["id", "frame", "x", "y", "z", "m"], comment="#")
    cross = df[df["y"] >= DOOR[1]].groupby("id")["frame"].min()
    t_open = int(cross.min())
    first_seen = df.groupby("id")["frame"].min()
    at_open = df[df["frame"] == t_open].set_index("id")
    distance = np.hypot(at_open["x"] - DOOR[0], at_open["y"] - DOOR[1])

    inside = shapely.contains_xy(walkable.polygon, df["x"].to_numpy(), df["y"].to_numpy())
    traj = pedpy.TrajectoryData(df.loc[inside, ["id", "frame", "x", "y"]].copy(), frame_rate=FPS)
    cells = compute_individual_voronoi_polygons(
        traj_data=traj, walkable_area=walkable, cut_off=Cutoff(radius=1.0, quad_segments=3)
    )
    cells["area"] = cells["polygon"].apply(lambda poly: poly.area)
    rank = cross.rank()
    return {
        "run": path.name[:5],
        "condition": condition(path),
        "n_crossed": len(cross),
        "door_open_s": t_open / FPS,
        "dropped_fraction": float(1 - inside.mean()),
        "rho_entry_cross": spearman_on_common(first_seen, cross),
        "rho_pos_at_open_cross": spearman_on_common(distance, cross),
        "rho_area_all": spearman_on_common(rank, cells.groupby("id")["area"].mean()),
        "rho_area_wait": spearman_on_common(rank, cells[cells["frame"] < t_open].groupby("id")["area"].mean()),
    }


def main() -> None:
    args = parse_args()
    walkable = load_walkable_area(None)
    files = sorted(args.traj_dir.glob("*_Combined.txt"))
    if not files:
        raise SystemExit(f"No *_Combined.txt files in {args.traj_dir}")
    table = pd.DataFrame([analyse(path, walkable) for path in files])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
