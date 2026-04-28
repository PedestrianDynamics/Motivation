"""Plot simulation trajectories per submodel as 1x5 panel figures.

Produces two variants:
  * <stem>_uniform.png     — all agents in a single color (matches Fig 7 style).
  * <stem>_motivation.png  — colored by per-agent mean motivation (cividis).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODELS: List[str] = ["SE", "V", "P", "PVE", "BASE_MODEL"]
MODEL_TITLES: Dict[str, str] = {
    "SE": r"$SE$",
    "V": r"$V$",
    "P": r"$P$",
    "PVE": r"$PVE$",
    "BASE_MODEL": "Base model",
}

UNIFORM_COLOR = "steelblue"
MARKER_COLOR = "crimson"
MOTIVATION_CMAP = "cividis"

MOTIVATION_CSV_COLUMNS = [
    "frame",
    "id",
    "time",
    "motivation",
    "x",
    "y",
    "value",
    "rank_abs",
    "rank_q",
    "payoff_p",
    "rank_update_flag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-runs-dir",
        type=Path,
        default=PROJECT_ROOT / "files" / "coordination_scenarios"
        / "agents_80_open_100" / "base_runs",
        help="Directory with per-seed *_motivation.csv files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=101,
        help="Seed to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT.parent / "motivation-for-springer" / "figures",
        help="Directory for output PNGs.",
    )
    parser.add_argument(
        "--stem",
        default="simulation_trajectories_5panel",
        help="Base stem for the two output files.",
    )
    return parser.parse_args()


def find_motivation_csv(base_runs_dir: Path, model: str, seed: int) -> Path:
    pattern = f"base_{model}_seed{seed}_*_motivation.csv"
    candidates = sorted(base_runs_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No motivation csv for model={model} seed={seed} in {base_runs_dir}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_motivation_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "frame" in df.columns and "id" in df.columns:
        return df
    return pd.read_csv(path, names=MOTIVATION_CSV_COLUMNS)


def last_positions(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("id")["frame"].idxmax()
    return df.loc[idx, ["id", "x", "y"]].reset_index(drop=True)


def plot_uniform_panel(ax: Axes, df: pd.DataFrame, title: str) -> None:
    for _, grp in df.sort_values("frame").groupby("id"):
        ax.plot(
            grp["x"].to_numpy(),
            grp["y"].to_numpy(),
            color=UNIFORM_COLOR,
            linewidth=0.6,
            alpha=0.7,
        )
    lp = last_positions(df)
    ax.scatter(
        lp["x"],
        lp["y"],
        marker="o",
        s=18,
        facecolor=MARKER_COLOR,
        edgecolor="black",
        linewidth=0.4,
        zorder=5,
    )
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_aspect("equal")


def plot_motivation_panel(
    ax: Axes,
    df: pd.DataFrame,
    title: str,
    norm: Normalize,
) -> None:
    mean_motivation = df.groupby("id")["motivation"].mean()
    cmap = plt.get_cmap(MOTIVATION_CMAP)
    for agent_id, grp in df.sort_values("frame").groupby("id"):
        color = cmap(norm(float(mean_motivation.loc[agent_id])))
        ax.plot(
            grp["x"].to_numpy(),
            grp["y"].to_numpy(),
            color=color,
            linewidth=0.6,
            alpha=0.85,
        )
    lp = last_positions(df)
    lp_colors = [
        cmap(norm(float(mean_motivation.loc[int(i)]))) for i in lp["id"].to_numpy()
    ]
    ax.scatter(
        lp["x"],
        lp["y"],
        marker="o",
        s=18,
        c=lp_colors,
        edgecolor="black",
        linewidth=0.4,
        zorder=5,
    )
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_aspect("equal")


def compute_global_motivation_range(
    frames_by_model: Dict[str, pd.DataFrame],
) -> Tuple[float, float]:
    mins: List[float] = []
    maxs: List[float] = []
    for df in frames_by_model.values():
        means = df.groupby("id")["motivation"].mean()
        if len(means):
            mins.append(float(means.min()))
            maxs.append(float(means.max()))
    if not mins:
        return 0.0, 1.0
    return min(mins), max(maxs)


def make_panels_figure(
    frames_by_model: Dict[str, pd.DataFrame],
    models: List[str],
    panel_plotter: Callable[..., None],
    suptitle: Optional[str] = None,
    cbar_norm: Optional[Normalize] = None,
) -> Figure:
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 7), constrained_layout=True)
    if n == 1:
        axes = [axes]

    first_ax = axes[0]
    for ax, model in zip(axes, models):
        df = frames_by_model[model]
        if cbar_norm is None:
            panel_plotter(ax, df, MODEL_TITLES.get(model, model))
        else:
            panel_plotter(ax, df, MODEL_TITLES.get(model, model), cbar_norm)
        if ax is first_ax:
            ax.set_ylabel("y [m]")

    if suptitle:
        fig.suptitle(suptitle)

    if cbar_norm is not None:
        sm = ScalarMappable(norm=cbar_norm, cmap=MOTIVATION_CMAP)
        sm.set_array([])
        fig.colorbar(
            sm,
            ax=axes,
            label="per-agent mean motivation",
            shrink=0.7,
            pad=0.01,
        )
    return fig


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames_by_model: Dict[str, pd.DataFrame] = {}
    for model in MODELS:
        csv_path = find_motivation_csv(args.base_runs_dir, model, args.seed)
        frames_by_model[model] = load_motivation_csv(csv_path)

    uniform_fig = make_panels_figure(
        frames_by_model,
        MODELS,
        plot_uniform_panel,
    )
    uniform_path = args.output_dir / f"{args.stem}_uniform.png"
    uniform_fig.savefig(uniform_path, dpi=200)
    print(f"Wrote {uniform_path}")
    plt.close(uniform_fig)

    vmin, vmax = compute_global_motivation_range(frames_by_model)
    norm = Normalize(vmin=vmin, vmax=vmax)
    motivation_fig = make_panels_figure(
        frames_by_model,
        MODELS,
        plot_motivation_panel,
        cbar_norm=norm,
    )
    motivation_path = args.output_dir / f"{args.stem}_motivation.png"
    motivation_fig.savefig(motivation_path, dpi=200)
    print(f"Wrote {motivation_path}")
    plt.close(motivation_fig)


if __name__ == "__main__":
    main()
