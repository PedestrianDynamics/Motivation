"""Compute stable overtaking events for experimental trajectory files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from shapely.geometry import Polygon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from overtaking_analysis import (
    build_summary_rows,
    build_time_summary,
    compute_overtakes_using_time_distance,
    filter_candidate_overtakes,
    plot_heatmaps,
    plot_model_comparison,
    plot_time_comparison,
    tagged_filename,
    write_csv,
)


DEFAULT_EXPERIMENT_DIR = Path("/Users/chraibi/workspace/Writing/Motivation/CroMa videos")
DEFAULT_MEASUREMENT_CONFIG = PROJECT_ROOT / "files" / "base_P.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute stable overtaking events for experimental trajectory files."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Directory containing experimental *_Combined.txt files.",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Explicitly map an experiment label to a trajectory txt file.",
    )
    parser.add_argument(
        "--measurement-config",
        type=Path,
        default=DEFAULT_MEASUREMENT_CONFIG,
        help="JSON config used to load the bottleneck measurement line.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=2.0,
        help="Maximum pair distance [m] for a candidate overtake.",
    )
    parser.add_argument(
        "--stability-window",
        type=int,
        default=5,
        help="Number of frames for the post-overtake stability check.",
    )
    parser.add_argument(
        "--speed-window",
        type=int,
        default=10,
        help="Number of frames used for the speed estimate.",
    )
    parser.add_argument(
        "--rel-speed-threshold",
        type=float,
        default=0.01,
        help="Minimum relative speed difference [m/s] for a stable overtake.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of heatmap bins per axis.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=10.0,
        help="Optional shared maximum for heatmap color scaling.",
    )
    parser.add_argument(
        "--t-min",
        type=float,
        default=10.0,
        help="Minimum time included in the analysis.",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=20.0,
        help="Maximum time included in the analysis.",
    )
    parser.add_argument(
        "--output-dir",
        default="overtaking_results_experimental",
        help="Directory for CSV summaries and figures.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix added to output filenames.",
    )
    return parser.parse_args()


def parse_input_overrides(items: Sequence[str]) -> Dict[str, Path]:
    overrides: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --input value '{item}'. Expected LABEL=PATH.")
        label, raw_path = item.split("=", 1)
        overrides[label.strip()] = Path(raw_path).expanduser()
    return overrides


def discover_experiment_files(experiment_dir: Path) -> Dict[str, Path]:
    discovered: Dict[str, Path] = {}
    for path in sorted(experiment_dir.glob("*_Combined.txt")):
        discovered[path.stem.split("_cam", 1)[0]] = path
    return discovered


def parse_frame_rate(path: Path) -> float:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("# framerate:"):
                return float(line.strip().split()[2])
    raise ValueError(f"Could not determine frame rate from {path}")


def load_measurement_line(config_path: Path):
    import json
    import pedpy

    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return pedpy.MeasurementLine(data["measurement_line"]["vertices"])


def build_experiment_geometry() -> Polygon:
    exterior = [
        (-7.0, -11.0),
        (-3.57, -3.0),
        (-3.64, 19.64),
        (-1.47, 19.57),
        (-1.32, 19.71),
        (-0.82, 19.71),
        (-0.67, 19.57),
        (-0.38, 19.57),
        (-0.38, 21.23),
        (-0.67, 21.23),
        (-0.82, 21.09),
        (-1.32, 21.09),
        (-1.47, 21.23),
        (-1.62, 21.23),
        (-1.62, 21.18),
        (-1.495, 21.18),
        (-1.37, 21.065),
        (-1.37, 19.735),
        (-1.495, 19.62),
        (-3.69, 19.69),
        (-3.62, -3.0),
        (-7.0, -11.0),
        (7.0, -11.0),
        (3.57, -3.0),
        (3.64, 19.64),
        (1.47, 19.57),
        (1.32, 19.71),
        (0.82, 19.71),
        (0.67, 19.57),
        (0.38, 19.57),
        (0.38, 21.23),
        (0.67, 21.23),
        (0.82, 21.09),
        (1.32, 21.09),
        (1.47, 21.23),
        (1.62, 21.23),
        (1.62, 21.18),
        (1.495, 21.18),
        (1.37, 21.065),
        (1.37, 19.735),
        (1.495, 19.62),
        (3.69, 19.69),
        (3.62, -3.0),
        (7.0, -11.0),
        (-7.0, -11.0),
    ]
    return Polygon(exterior)


def load_experiment_dataframe(path: Path) -> Tuple[pd.DataFrame, float]:
    fps = parse_frame_rate(path)
    rows: List[Tuple[int, int, float, float, float, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            rows.append(
                (
                    int(parts[0]),
                    int(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                    float(parts[5]),
                )
            )
    dataframe = pd.DataFrame(rows, columns=["id", "frame", "x", "y", "z", "m"])
    return dataframe, fps


def compute_label_rows(
    label: str,
    path: Path,
    measurement_line,
    geometry: Polygon,
    t_min: float,
    t_max: float,
    distance_threshold: float,
    stability_window: int,
    speed_window: int,
    rel_speed_threshold: float,
):
    import pedpy

    traj_df, fps = load_experiment_dataframe(path)
    traj = pedpy.TrajectoryData(traj_df.copy(), frame_rate=fps)
    time_distance_df = pedpy.compute_time_distance_line(
        traj_data=traj,
        measurement_line=measurement_line,
    )
    candidate_events, _ = compute_overtakes_using_time_distance(
        traj_df=traj_df[["id", "frame", "x", "y"]].copy(),
        time_distance_df=time_distance_df,
        distance_threshold=distance_threshold,
    )
    if candidate_events.empty:
        return [], geometry

    stable_events = filter_candidate_overtakes(
        traj_df=traj_df[["id", "frame", "x", "y"]].copy(),
        time_distance_df=time_distance_df,
        candidate_events=candidate_events,
        frame_rate=fps,
        stability_window=stability_window,
        speed_window=speed_window,
        rel_speed_threshold=rel_speed_threshold,
    )
    if stable_events.empty:
        return [], geometry

    stable_events = stable_events.copy()
    stable_events["time"] = stable_events["frame"] / float(fps)
    stable_events = stable_events[
        (stable_events["time"] >= t_min) & (stable_events["time"] <= t_max)
    ]
    if stable_events.empty:
        return [], geometry

    pair_counts = (
        stable_events.groupby(["idA", "idB"], as_index=False)["is_overtake"]
        .sum()
        .rename(columns={"is_overtake": "num_overtakes"})
    )
    pair_lookup = {
        (int(row["idA"]), int(row["idB"])): int(row["num_overtakes"])
        for _, row in pair_counts.iterrows()
    }

    rows: List[Dict[str, float]] = []
    for _, row in stable_events.iterrows():
        rows.append(
            {
                "model": label,
                "source_file": str(path),
                "frame": int(row["frame"]),
                "time": float(row["time"]),
                "idA": int(row["idA"]),
                "idB": int(row["idB"]),
                "x": float(row["xA_tplus1"]),
                "y": float(row["yA_tplus1"]),
                "distAB_t": float(row["distAB_t"]),
                "speedA_mps": float(row["speedA_mps"]),
                "speedB_mps": float(row["speedB_mps"]),
                "num_overtakes_pair": int(
                    pair_lookup[(int(row["idA"]), int(row["idB"]))]
                ),
            }
        )
    return rows, geometry


def main() -> None:
    args = parse_args()
    overrides = parse_input_overrides(args.input)
    discovered = discover_experiment_files(args.experiment_dir)
    inputs = discovered | overrides
    if not inputs:
        raise SystemExit("No experimental trajectory files found.")

    measurement_line = load_measurement_line(args.measurement_config)
    geometry = build_experiment_geometry()

    detail_rows_by_label: Dict[str, List[Dict[str, float]]] = {}
    geometries: Dict[str, Polygon] = {}
    for label, path in sorted(inputs.items()):
        rows, polygon = compute_label_rows(
            label=label,
            path=path,
            measurement_line=measurement_line,
            geometry=geometry,
            t_min=args.t_min,
            t_max=args.t_max,
            distance_threshold=args.distance_threshold,
            stability_window=args.stability_window,
            speed_window=args.speed_window,
            rel_speed_threshold=args.rel_speed_threshold,
        )
        detail_rows_by_label[label] = rows
        geometries[label] = polygon

    detail_rows = [row for rows in detail_rows_by_label.values() for row in rows]
    summary_rows = build_summary_rows(detail_rows)
    time_summary_rows = build_time_summary(detail_rows)
    output_dir = Path(args.output_dir)
    write_csv(
        output_dir / tagged_filename("overtaking_details_experimental", ".csv", args.tag),
        detail_rows,
        [
            "model",
            "source_file",
            "frame",
            "time",
            "idA",
            "idB",
            "x",
            "y",
            "distAB_t",
            "speedA_mps",
            "speedB_mps",
            "num_overtakes_pair",
        ],
    )
    write_csv(
        output_dir / tagged_filename("overtaking_summary_experimental", ".csv", args.tag),
        summary_rows,
        [
            "model",
            "num_stable_overtake_events",
            "num_overtaking_pairs",
            "mean_pair_distance_m",
            "mean_relative_speed_mps",
        ],
    )
    write_csv(
        output_dir
        / tagged_filename("overtaking_time_summary_experimental", ".csv", args.tag),
        time_summary_rows,
        [
            "model",
            "time",
            "num_stable_overtake_events",
        ],
    )
    created_heatmaps = plot_heatmaps(
        {label: rows for label, rows in detail_rows_by_label.items() if rows},
        geometries,
        output_dir,
        bins=args.bins,
        vmax=args.vmax,
        tag=args.tag,
    )
    created_comparison = plot_model_comparison(summary_rows, output_dir, args.tag)
    created_time = plot_time_comparison(time_summary_rows, output_dir, args.tag)
    if not created_heatmaps and not created_comparison and not created_time:
        print("CSV outputs were written, but no figures were created.")
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
