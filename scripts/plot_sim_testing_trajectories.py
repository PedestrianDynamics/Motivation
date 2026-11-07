"""Plot simulated trajectories from the four base_testing_<run> runs as a
1x4 panel figure, in the same axes/style as croma_trajectories_4panel.png
for direct visual comparison with the CROMA experiments.

Reads the single *_motivation.csv file under --runs-dir/<run>/ and traces
(x, y) per agent across all frames. Walls and the closed door are drawn
from the walkable area stored in the run's sqlite file.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shapely
from matplotlib.axes import Axes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# (label, motivation_group)
SCENARIOS: list[tuple[str, str]] = [
    ("1C060", "nM"),
    ("2C020", "nM"),
    ("2C070", "hM"),
    ("2C120", "hM"),
]

TRAJ_COLOR = "steelblue"
MARKER_COLOR = "crimson"
WALL_COLOR = "black"
DOOR_COLOR = "dimgray"
# View: the corridor, the funnel entrance and the door; the walkable area
# extends further but no agent goes there.
XLIM = (-7.0, 7.0)
YLIM = (-12.0, 22.5)


def latest_motivation_csv(run_dir: Path) -> Path:
    csvs = sorted(run_dir.glob("*_motivation.csv"))
    if len(csvs) != 1:
        raise RuntimeError(f"Expected exactly one *_motivation.csv in {run_dir}, found {len(csvs)}")
    return csvs[0]


def walkable_area(run_dir: Path) -> shapely.Polygon:
    dbs = sorted(run_dir.glob("*.sqlite"))
    if len(dbs) != 1:
        raise RuntimeError(f"Expected exactly one *.sqlite in {run_dir}, found {len(dbs)}")
    with sqlite3.connect(f"file:{dbs[0]}?mode=ro", uri=True) as con:
        rows = con.execute("SELECT wkt FROM geometry").fetchall()
    if len(rows) != 1:
        raise RuntimeError(f"Expected one stored geometry in {dbs[0]}, found {len(rows)}")
    return shapely.from_wkt(rows[0][0])


def plot_geometry(ax: Axes, area: shapely.Polygon) -> None:
    """Walls as lines; holes (the closed door) as filled obstacles."""
    ax.plot(*area.exterior.xy, color=WALL_COLOR, linewidth=1.0)
    for hole in area.interiors:
        ax.fill(*hole.xy, color=DOOR_COLOR, linewidth=0, label="closed door")


def plot_panel(ax: Axes, df: pd.DataFrame, area: shapely.Polygon, title: str) -> None:
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
    plot_geometry(ax, area)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.legend(loc="lower center", fontsize=7, frameon=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=PROJECT_ROOT / "build" / "testing_runs",
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
        plot_panel(ax, df, walkable_area(run_dir), f"{label} ({group}) — simulation")

    fig.savefig(args.output, dpi=200)
    print(f"Wrote {args.output}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
