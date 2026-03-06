#!/usr/bin/env python3
"""Plot one frame with geometry, agent circles, and rank/payoff labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jupedsim.internal.notebook_utils import read_sqlite_file
from matplotlib.patches import Circle


def _label_text_color(rgba: tuple[float, float, float, float]) -> str:
    r, g, b, _ = rgba
    # Perceived luminance to switch text for contrast.
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.6 else "white"


def _load_motivation_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "frame" in df.columns and "id" in df.columns:
        return df
    return pd.read_csv(
        path,
        names=[
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
        ],
    )


def _validate_requested_columns(df: pd.DataFrame, label: str, color_by: str) -> None:
    needed: set[str] = set()
    if label in {"rank", "both"}:
        needed.add("rank_abs")
    if label in {"payoff", "both"}:
        needed.add("payoff_p")
    if color_by == "rank":
        needed.add("rank_abs")
    if color_by == "payoff":
        needed.add("payoff_p")
    if color_by == "motivation":
        needed.add("motivation")
    if color_by == "value":
        needed.add("value")
    missing = sorted(col for col in needed if col not in df.columns)
    if missing:
        raise ValueError(
            "Motivation CSV is missing required columns for this plot: "
            f"{', '.join(missing)}. Use the CSV produced by the current runtime "
            "(with rank/payoff debug columns)."
        )


def _select_frame(
    frame: int | None, time_s: float | None, frame_rate: float, available: np.ndarray
) -> int:
    if frame is not None:
        target = int(frame)
    elif time_s is not None:
        target = int(round(time_s * frame_rate))
    else:
        raise ValueError("Provide --frame or --time.")
    if target in set(int(v) for v in available):
        return target
    nearest = int(available[np.argmin(np.abs(available - target))])
    return nearest


def _select_motivation_slice(
    motivation_df: pd.DataFrame, sqlite_frame: int, frame_rate: float
) -> tuple[pd.DataFrame, float]:
    """Select motivation rows aligned to sqlite frame, using nearest time fallback."""
    target_time = float(sqlite_frame) / float(frame_rate)
    if "time" not in motivation_df.columns:
        return motivation_df[motivation_df["frame"] == sqlite_frame], target_time

    # Prefer exact frame match when available.
    exact = motivation_df[motivation_df["frame"] == sqlite_frame]
    if not exact.empty:
        return exact, target_time

    # Fallback: nearest recorded motivation time.
    times = motivation_df["time"].astype(float).to_numpy()
    nearest_time = float(times[np.argmin(np.abs(times - target_time))])
    tol = 1e-9
    aligned = motivation_df[
        (motivation_df["time"].astype(float) - nearest_time).abs() <= tol
    ]
    return aligned, nearest_time


def _plot_geometry(ax: plt.Axes, walkable_area: object) -> None:
    polygon = walkable_area.polygon
    x_ext, y_ext = polygon.exterior.xy
    ax.plot(np.asarray(x_ext), np.asarray(y_ext), color="black", lw=1.2)
    for interior in polygon.interiors:
        x_in, y_in = interior.xy
        ax.plot(np.asarray(x_in), np.asarray(y_in), color="black", lw=1.2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", required=True, type=Path)
    parser.add_argument("--motivation-csv", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--time", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--label", choices=["rank", "payoff", "both"], default="rank")
    parser.add_argument(
        "--color-by",
        choices=["rank", "payoff", "motivation", "value", "expectancy"],
        default="rank",
    )
    args = parser.parse_args()

    traj, walkable_area = read_sqlite_file(str(args.sqlite))
    tdf = traj.data
    motivation_df = _load_motivation_csv(args.motivation_csv)
    _validate_requested_columns(motivation_df, args.label, args.color_by)
    config = json.loads(args.json.read_text(encoding="utf8"))

    available_frames = np.sort(tdf["frame"].unique())
    frame = _select_frame(args.frame, args.time, traj.frame_rate, available_frames)

    tf = tdf[tdf["frame"] == frame][["id", "frame", "x", "y"]].copy()
    mf, aligned_time = _select_motivation_slice(motivation_df, frame, traj.frame_rate)
    merge_on_id_and_frame = "frame" in mf.columns and bool((mf["frame"] == frame).all())
    merge_cols = ["id", "frame"] if merge_on_id_and_frame else ["id"]
    df = tf.merge(mf, on=merge_cols, how="left", suffixes=("", "_m"))

    radius = float(config["velocity_init_parameters"].get("radius", 0.15)) * 1.3

    fig, ax = plt.subplots(figsize=(8, 8))
    _plot_geometry(ax, walkable_area)

    color_key: Literal["rank_abs", "payoff_p", "motivation", "value", "expectancy"]
    if args.color_by == "rank":
        color_key = "rank_abs"
    elif args.color_by == "payoff":
        color_key = "payoff_p"
    elif args.color_by == "value":
        color_key = "value"
    elif args.color_by == "expectancy":
        color_key = "expectancy"
    else:
        color_key = "motivation"

    if color_key == "expectancy":
        door = config["motivation_parameters"]["motivation_doors"][0]["vertices"]
        x0 = 0.5 * (door[0][0] + door[1][0])
        y0 = 0.5 * (door[0][1] + door[1][1])
        width = float(config["motivation_parameters"]["width"])
        height = float(config["motivation_parameters"]["height"])
        dist = np.sqrt((df["x"].to_numpy(dtype=float) - x0) ** 2 + (df["y"].to_numpy(dtype=float) - y0) ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            expr = 1.0 / ((dist / width) ** 2 - 1.0)
            expectancy_vals = np.where(dist >= width, 1.0, 1.0 + np.exp(expr) * np.e * height)
        df[color_key] = expectancy_vals

    cvals = df[color_key].to_numpy(dtype=float)
    if np.isnan(cvals).all():
        cvals = np.zeros_like(cvals)
    vmin, vmax = np.nanmin(cvals), np.nanmax(cvals)
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    cmap = plt.get_cmap("viridis")

    for _, row in df.iterrows():
        val = float(row[color_key]) if pd.notna(row[color_key]) else vmin
        color = cmap((val - vmin) / (vmax - vmin))
        circle = Circle(
            (float(row["x"]), float(row["y"])),
            radius=radius,
            fc=color,
            ec="k",
            lw=0.5,
            alpha=0.85,
        )
        ax.add_patch(circle)

        if args.label == "rank":
            label = f"{int(row['rank_abs'])}" if pd.notna(row["rank_abs"]) else "NA"
        elif args.label == "payoff":
            label = (
                f"{float(row['payoff_p']):.2f}" if pd.notna(row["payoff_p"]) else "NA"
            )
        else:
            if pd.notna(row["rank_abs"]) and pd.notna(row["payoff_p"]):
                label = f"{int(row['rank_abs'])}|{float(row['payoff_p']):.2f}"
            else:
                label = "NA"
        ax.text(
            float(row["x"]),
            float(row["y"]),
            label,
            ha="center",
            va="center",
            fontsize=7,
            color=_label_text_color(color),
        )

    ax.set_title(
        f"Frame {frame} (t~{aligned_time:.2f}s) | color={args.color_by} | label={args.label}"
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    _, y_max = ax.get_ylim()
    ax.set_ylim(0.0, y_max)
    for spine in ax.spines.values():
        spine.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=color_key)
    fig.tight_layout()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)

        out_path = args.out.with_name(
            f"{args.out.stem}_{args.frame:05d}{args.out.suffix}"
        )

        fig.savefig(out_path, dpi=300)
        print(f"Saved: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
