"""Crowd depth over time before door opening: CROMA runs against the three replicas.

Reads timeseries.csv and advance.csv written by croma_timeseries.py. One panel
per paper run; lines are the median depth behind the front-most person, model
lines the mean over seeds with the seed range shaded. Each panel states the
fraction of persons who stood still for 5 s and then advanced more than 3 m.

  python scripts/plot_croma_timeseries.py --input-dir build/croma_timeseries
writes croma_depth_timeseries.pdf/.png to the paper's figures/ directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input-dir", type=Path, default=PROJECT_ROOT / "build" / "croma_timeseries"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT.parent / "motivation-for-springer" / "figures",
    )
    parser.add_argument("--t-min", type=float, default=-90.0)
    return parser.parse_args()


def main():
    """Plot median crowd depth against time to door opening.

    Saves
    -----
    croma_depth_timeseries.pdf and .png in the output directory.
    """
    args = parse_args()

    # --- Data ---
    series = pd.read_csv(args.input_dir / "timeseries.csv").dropna(subset=["depth_med"])
    series["t"] = series["t_to_open_s"].round()
    series = series[series["t"] >= args.t_min]
    advance = pd.read_csv(args.input_dir / "advance.csv").dropna(subset=["frac_adv3m"])
    adv3 = advance.groupby(["source", "run"])["frac_adv3m"].mean()

    runs = [("1C060", "nM"), ("2C020", "nM"), ("2C070", "hM"), ("2C120", "hM")]
    pal = [
        "#90c1c6",
        "#72a5b4",
        "#58849f",
        "#446485",
        "#324465",
        "#1f253f",
    ]  # cubehelix_6
    styles = {
        "experiment": {"label": "Experiment", "color": "black", "ls": "-", "lw": 2.4},
        "BASE_MODEL": {"label": "Static", "color": "grey", "ls": ":", "lw": 1.8},
        "V": {
            "label": "Static heterogeneity (V)",
            "color": pal[2],
            "ls": "--",
            "lw": 1.8,
        },
        "PVE": {"label": "EVP", "color": "#bd0c0c", "ls": "-", "lw": 1.8},
    }

    # --- Style Setup ---
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": "white"})

    # --- Plot ---
    fig = plt.figure(figsize=(10, 8), dpi=150)
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.08, hspace=0.30, left=0.08, right=0.99, top=0.91, bottom=0.10)
    axes = [plt.subplot(gs[i, j]) for i in range(2) for j in range(2)]
    y_max = series["depth_med"].max() * 1.05

    for i, (ax, (run, cond)) in enumerate(zip(axes, runs)):
        row, col = divmod(i, 2)
        ax.grid(False)
        ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
        ax.patch.set_edgecolor("lightgrey")
        ax.patch.set_linewidth(0.8)

        lines = []
        for source, st in styles.items():
            g = series[(series["source"] == source) & (series["run"] == run)]
            stats = g.groupby("t")["depth_med"].agg(["mean", "min", "max"])
            if source != "experiment":
                ax.fill_between(
                    stats.index,
                    stats["min"],
                    stats["max"],
                    color=st["color"],
                    alpha=0.15,
                    lw=0,
                )
            ax.plot(
                stats.index,
                stats["mean"],
                color=st["color"],
                ls=st["ls"],
                lw=st["lw"],
                label=st["label"],
                zorder=3 if source == "experiment" else 2,
            )
            lines.append(
                f"{st['label'].split(' (')[0]}: {adv3.get((source, run), float('nan')):.0%}"
            )

        ax.text(
            0.97,
            0.97,
            "stood, then advanced > 3 m\n" + "\n".join(lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="dimgrey",
        )
        ax.set_xlim(args.t_min, 0)
        ax.set_ylim(0, y_max)
        ax.set_title(
            r"$\bf{(" + chr(ord("a") + i) + r")}$" + f"  {run} ({cond})",
            loc="left",
            fontsize=11,
            pad=7,
        )
        if col == 0:
            ax.set_ylabel(
                "Median depth behind front [m]", fontsize=9, labelpad=6, color="dimgrey"
            )
        else:
            ax.tick_params(labelleft=False)
        if row == 1:
            ax.set_xlabel(
                "Time to door opening [s]", fontsize=9, labelpad=6, color="dimgrey"
            )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.53, 0.99),
        frameon=True,
        facecolor="white",
        framealpha=0.8,
        edgecolor="lightgrey",
        labelcolor="dimgrey",
        fontsize=9,
    )
    peak = (
        series[series["run"] == "2C120"]
        .groupby(["source", "t"])["depth_med"]
        .mean()
        .groupby("source")
        .max()
    )
    fig.text(
        0.99,
        0.01,
        "Among the models only EVP lets persons stand and then advance; in 2C120 its crowd "
        f"stretches to {peak['PVE']:.0f} m before compacting (experiment {peak['experiment']:.0f} m)",
        ha="right",
        va="bottom",
        fontsize=9,
        color="dimgrey",
        style="italic",
    )
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)

    # --- Save ---
    for ext in ("pdf", "png"):
        fig.savefig(
            args.output_dir / f"croma_depth_timeseries.{ext}",
            dpi=150,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    main()
