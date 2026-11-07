"""Shape of the waiting crowd in the CROMA runs and their simulated twins.

For each snapshot shown in croma_trajectories_4panel.png (experiment, frame
at which the door opens) and sim_testing_trajectories_4panel.png
(simulation, last frame), measure with depth = y_front - y, where y_front
is the position of the front-most person:
  depth_med, depth_p90  median and 90th percentile of depth [m],
  sd_x                  lateral standard deviation of all persons [m],
  sd_x_front3m          lateral standard deviation within 3 m of the front [m],
  frac_front3m          fraction of persons within 3 m of the front.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plot_experimental_trajectories as exp_traj
import plot_sim_testing_trajectories as sim_traj

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONT_BAND = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--traj-dir", type=Path, default=PROJECT_ROOT.parent / "trajectories_croma")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "build" / "testing_runs")
    parser.add_argument("--output", type=Path, required=True, help="CSV file to write.")
    return parser.parse_args()


def shape(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    depth = y.max() - y
    near = depth <= FRONT_BAND
    return {
        "n": len(x),
        "depth_med": float(np.median(depth)),
        "depth_p90": float(np.percentile(depth, 90)),
        "sd_x": float(np.std(x)),
        "sd_x_front3m": float(np.std(x[near])),
        "frac_front3m": float(near.mean()),
    }


def experiment_rows(traj_dir: Path) -> list[dict]:
    rows = []
    for label, stem, group in exp_traj.SCENARIOS:
        data = exp_traj.load_trajectory(traj_dir / f"{stem}.txt")
        frame = exp_traj.door_open_frame(data)
        snap = data[data[:, 1].astype(int) == frame]
        rows.append({"run": label, "condition": group, "source": "experiment",
                     "frame": frame, **shape(snap[:, 2], snap[:, 3])})
    return rows


def simulation_rows(runs_dir: Path) -> list[dict]:
    rows = []
    for label, group in sim_traj.SCENARIOS:
        csv = sim_traj.latest_motivation_csv(runs_dir / f"test_{label}")
        df = pd.read_csv(csv, usecols=["frame", "id", "x", "y"])
        frame = int(df["frame"].max())
        snap = df[df["frame"] == frame]
        rows.append({"run": label, "condition": group, "source": "simulation",
                     "frame": frame, **shape(snap["x"].to_numpy(), snap["y"].to_numpy())})
    return rows


def main() -> None:
    args = parse_args()
    table = pd.DataFrame(experiment_rows(args.traj_dir) + simulation_rows(args.runs_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
