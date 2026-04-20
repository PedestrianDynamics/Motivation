"""Plot CROMA experimental trajectories as a 2x2 panel figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TRAJ_DIR = PROJECT_ROOT.parent / "trajectories_croma"

# (label, filename_stem, motivation_group)
SCENARIOS: List[Tuple[str, str, str]] = [
    ("1C060", "1C060_cam6_cam5_frameshift0_Combined", "nM"),
    ("2C020", "2C020_cam6_cam5_frameshift0_Combined", "nM"),
    ("2C070", "2C070_cam6_cam5_frameshift0_Combined", "hM"),
    ("2C120", "2C120_cam6_cam5_frameshift0_Combined", "hM"),
]

TRAJ_COLOR = "steelblue"
MARKER_COLOR = "crimson"
DOOR_Y = 20.0  # CROMA door line (see scripts/experimental_rank_area.py)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traj-dir",
        type=Path,
        default=DEFAULT_TRAJ_DIR,
        help="Directory containing CROMA *_Combined.txt trajectory files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT.parent / "motivation-for-springer" / "figures"
        / "croma_trajectories_4panel.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def load_trajectory(path: Path) -> np.ndarray:
    """Load CROMA trajectory file; returns array with columns [id, frame, x, y]."""
    data = np.loadtxt(path, comments="#", usecols=(0, 1, 2, 3))
    return data


def door_open_frame(data: np.ndarray) -> int:
    """First frame at which any agent has crossed the door line (y >= DOOR_Y)."""
    crossed = data[data[:, 3] >= DOOR_Y]
    if crossed.size == 0:
        return int(data[:, 1].max())
    return int(crossed[:, 1].min())


def plot_panel(ax: Axes, data: np.ndarray, title: str) -> None:
    ids = data[:, 0].astype(int)
    frames = data[:, 1].astype(int)
    xs = data[:, 2]
    ys = data[:, 3]

    for agent_id in np.unique(ids):
        mask = ids == agent_id
        order = np.argsort(frames[mask])
        ax.plot(
            xs[mask][order],
            ys[mask][order],
            color=TRAJ_COLOR,
            linewidth=0.6,
            alpha=0.7,
        )

    t_open = door_open_frame(data)
    snapshot = data[frames == t_open]
    ax.scatter(
        snapshot[:, 2],
        snapshot[:, 3],
        marker="o",
        s=18,
        facecolor=MARKER_COLOR,
        edgecolor="black",
        linewidth=0.4,
        zorder=5,
        label=f"door opens (frame {t_open})",
    )
    ax.axhline(DOOR_Y, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=7, frameon=False)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(14, 8), constrained_layout=True)

    for (label, stem, group), ax in zip(SCENARIOS, axes):
        traj_path = args.traj_dir / f"{stem}.txt"
        if not traj_path.exists():
            raise FileNotFoundError(f"Missing trajectory: {traj_path}")
        data = load_trajectory(traj_path)
        plot_panel(ax, data, f"{label} ({group})")

    fig.savefig(args.output, dpi=200)
    print(f"Wrote {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
