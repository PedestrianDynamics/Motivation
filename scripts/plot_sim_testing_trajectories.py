"""Plot simulated trajectories from the four base_testing_<run> runs as a
1x4 panel figure, in the same axes/style as croma_trajectories_4panel.png
for direct visual comparison with the CROMA experiments.

Reads each run's most recent *_motivation.csv file under --runs-dir/<run>/
and traces (x, y) per agent across all frames.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# (label, motivation_group)
SCENARIOS: List[Tuple[str, str]] = [
    ("1C060", "nM"),
    ("2C020", "nM"),
    ("2C070", "hM"),
    ("2C120", "hM"),
]

TRAJ_COLOR = "steelblue"
MARKER_COLOR = "crimson"
DOOR_Y = 19.0  # simulation door line


def latest_motivation_csv(run_dir: Path) -> Path:
    csvs = sorted(run_dir.glob("*_motivation.csv"))
    if not csvs:
        raise FileNotFoundError(f"No *_motivation.csv in {run_dir}")
    return csvs[-1]


def plot_panel(ax: Axes, df: pd.DataFrame, title: str) -> None:
    for _, sub in df.groupby("id"):
        sub = sub.sort_values("frame")
        ax.plot(sub["x"], sub["y"], color=TRAJ_COLOR, linewidth=0.6, alpha=0.7)

    last_frame = int(df["frame"].max())
    snap = df[df["frame"] == last_frame]
    ax.scatter(
        snap["x"],
        snap["y"],
        marker="o",
        s=18,
        facecolor=MARKER_COLOR,
        edgecolor="black",
        linewidth=0.4,
        zorder=5,
        label=f"final (frame {last_frame})",
    )
    ax.axhline(DOOR_Y, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=7, frameon=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("/tmp"),
        help="Directory containing test_<run>/ subdirs with *_motivation.csv outputs.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT.parent
        / "motivation-for-springer"
        / "figures"
        / "sim_testing_trajectories_4panel.png",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(14, 8), constrained_layout=True)

    for (label, group), ax in zip(SCENARIOS, axes):
        run_dir = args.runs_dir / f"test_{label}"
        csv = latest_motivation_csv(run_dir)
        df = pd.read_csv(
            csv,
            usecols=["frame", "id", "x", "y"],
            dtype={"frame": int, "id": int, "x": float, "y": float},
        )
        plot_panel(ax, df, f"{label} ({group}) — simulation")

    fig.savefig(args.output, dpi=200)
    print(f"Wrote {args.output}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
