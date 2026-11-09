"""Figure for the two-exit demonstration (scripts/two_exit_demo.py).

(a) Positions at one time, coloured by motivation, marker by exit;
    trajectories in grey.
(b) Motivation against distance to the agent's own exit at the same time.

  python scripts/plot_two_exit_demo.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from two_exit_demo import EXIT_X, ROOM, geometry

MARKERS = ("o", "^")
LABELS = ("left exit", "right exit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path,
                        default=PROJECT_ROOT / "build" / "two_exit_demo" / "two_exit_seed101_motivation.csv")
    parser.add_argument("--time", type=float, default=3.0, help="Snapshot time [s].")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT.parent / "motivation-for-springer" / "figures")
    return parser.parse_args()


def draw_room(ax) -> None:
    walkable, exits = geometry()
    ax.plot(*walkable.exterior.xy, color="black", linewidth=0.8)
    for exit_polygon in exits:
        ax.fill(*exit_polygon.exterior.xy, color="lightgrey", linewidth=0)


def main() -> None:
    args = parse_args()
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8})
    df = pd.read_csv(args.input, skipinitialspace=True)
    snap = df[(df["time"] - args.time).abs() < 1e-6]
    top = ROOM.bounds[3]
    snap = snap.assign(distance=((snap["x"] - snap["goal"].map(dict(enumerate(EXIT_X)))) ** 2
                                 + (snap["y"] - top) ** 2) ** 0.5)
    norm = Normalize(vmin=0.0, vmax=float(df["motivation"].max()))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(5.2, 2.4), width_ratios=(1.3, 1.0),
                                     constrained_layout=True)
    for _, grp in df[df["time"] <= args.time].groupby("id"):
        ax_a.plot(grp["x"], grp["y"], color="lightgrey", linewidth=0.3, zorder=1)
    draw_room(ax_a)
    for k, marker in enumerate(MARKERS):
        part = snap[snap["goal"] == k]
        points = ax_a.scatter(part["x"], part["y"], c=part["motivation"], cmap="viridis", norm=norm,
                              marker=marker, s=10, edgecolor="black", linewidth=0.2, zorder=3)
        ax_b.scatter(part["distance"], part["motivation"], c=part["motivation"], cmap="viridis",
                     norm=norm, marker=marker, s=10, edgecolor="black", linewidth=0.2,
                     label=f"{LABELS[k]} ({len(part)})")
    ax_a.set_aspect("equal")
    ax_a.set_xlabel("x [m]")
    ax_a.set_ylabel("y [m]")
    ax_a.set_title(f"(a) t = {args.time:g} s")
    fig.colorbar(points, ax=ax_a, label="motivation", shrink=0.8, pad=0.02)
    ax_b.set_xlabel("distance to own exit [m]")
    ax_b.set_ylabel("motivation")
    ax_b.set_title(f"(b) t = {args.time:g} s")
    legend = ax_b.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.8,
                         edgecolor="lightgrey", fontsize=7)
    for handle in legend.legend_handles:
        handle.set_array(None)
        handle.set_facecolor("grey")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = args.output_dir / f"two_exit_demo.{ext}"
        fig.savefig(path, dpi=300)
        print(f"Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
