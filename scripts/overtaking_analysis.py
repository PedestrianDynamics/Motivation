"""Compute stable overtaking events across simulation models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_ALIASES = {
    "TOGETHER": "PVE",
    "ALL": "PVE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute stable overtaking events from simulation trajectories."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["P", "V", "E", "PVE", "NO_MOTIVATION"],
        help="Models to analyze.",
    )
    parser.add_argument(
        "--search-dir",
        action="append",
        default=["files/base_runs", "files/variations"],
        help="Directories searched for simulation sqlite outputs.",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="MODEL=SQLITE",
        help="Explicitly map a model to a simulation sqlite file.",
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
        default=0.2,
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
        default=None,
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
        default=300.0,
        help="Maximum time included in the analysis.",
    )
    parser.add_argument(
        "--output-dir",
        default="overtaking_results",
        help="Directory for CSV summaries and figures.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix added to output filenames.",
    )
    return parser.parse_args()


def normalize_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name.strip().upper(), name.strip().upper())


def parse_input_overrides(items: Sequence[str]) -> Dict[str, Path]:
    overrides: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --input value '{item}'. Expected MODEL=SQLITE.")
        model_name, raw_path = item.split("=", 1)
        overrides[normalize_model_name(model_name)] = Path(raw_path).expanduser()
    return overrides


def discover_latest_sqlite(model: str, search_dirs: Sequence[str]) -> Optional[Path]:
    pattern = f"base_{model}_*.sqlite"
    candidates: List[Path] = []
    for directory in search_dirs:
        base_dir = Path(directory)
        if not base_dir.exists():
            continue
        candidates.extend(sorted(base_dir.glob(pattern)))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def find_matching_config(sqlite_path: Path) -> Path:
    json_path = sqlite_path.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing matching config json for {sqlite_path}")
    return json_path


def load_measurement_line(config_path: Path):
    import pedpy

    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return pedpy.MeasurementLine(data["measurement_line"]["vertices"])


def compute_speed_mps(
    traj_df,
    agent_id: int,
    start_frame: int,
    frame_rate: float,
    window: int,
) -> float:
    subset = traj_df[
        (traj_df["id"] == agent_id)
        & (traj_df["frame"] >= start_frame)
        & (traj_df["frame"] <= start_frame + window)
    ].sort_values("frame")
    if len(subset) < 2:
        return 0.0
    first = subset.iloc[0]
    last = subset.iloc[-1]
    dt_seconds = (float(last["frame"]) - float(first["frame"])) / float(frame_rate)
    if dt_seconds <= 0:
        return 0.0
    distance = float(np.hypot(last["x"] - first["x"], last["y"] - first["y"]))
    return distance / dt_seconds


def build_distance_lookup(time_distance_df) -> Dict[Tuple[int, int], float]:
    return {
        (int(row["id"]), int(row["frame"])): float(row["distance"])
        for _, row in time_distance_df.iterrows()
    }


def check_overtake_stability(
    distance_lookup: Dict[Tuple[int, int], float],
    id_a: int,
    id_b: int,
    start_frame: int,
    window: int,
) -> bool:
    for frame in range(start_frame, start_frame + window + 1):
        key_a = (id_a, frame)
        key_b = (id_b, frame)
        if key_a not in distance_lookup or key_b not in distance_lookup:
            return False
        if distance_lookup[key_a] >= distance_lookup[key_b]:
            return False
    return True


def compute_overtakes_using_time_distance(
    traj_df,
    time_distance_df,
    distance_threshold: float,
):
    df = traj_df.merge(time_distance_df, on=["id", "frame"], how="inner")
    df_sorted = df.sort_values(by=["frame", "distance"], ascending=[True, True]).copy()
    df_sorted["rank_t"] = df_sorted.groupby("frame")["distance"].rank(
        method="first",
        ascending=True,
    )
    df_sorted = df_sorted.rename(
        columns={"x": "x_t", "y": "y_t", "distance": "dist_t"}
    )

    df_next = df_sorted.copy()
    df_next["frame"] = df_next["frame"] - 1
    df_next = df_next.rename(
        columns={
            "rank_t": "rank_tplus1",
            "x_t": "x_tplus1",
            "y_t": "y_tplus1",
            "dist_t": "dist_tplus1",
        }
    )
    df_merged = df_sorted[["id", "frame", "rank_t", "x_t", "y_t", "dist_t"]].merge(
        df_next[
            ["id", "frame", "rank_tplus1", "x_tplus1", "y_tplus1", "dist_tplus1"]
        ],
        on=["id", "frame"],
        how="inner",
    )

    overtake_events: List[Dict[str, float]] = []
    for frame, group in df_merged.groupby("frame"):
        agents = group.sort_values("rank_t", ascending=True).to_dict("records")
        for index, agent_a in enumerate(agents):
            for agent_b in agents[index + 1 :]:
                if abs(float(agent_a["rank_t"]) - float(agent_b["rank_t"])) > 20:
                    continue
                pair_distance = float(
                    np.hypot(
                        agent_a["x_t"] - agent_b["x_t"],
                        agent_a["y_t"] - agent_b["y_t"],
                    )
                )
                if pair_distance >= distance_threshold:
                    continue

                if (
                    agent_a["dist_t"] > agent_b["dist_t"]
                    and agent_a["dist_tplus1"] < agent_b["dist_tplus1"]
                ):
                    overtake_events.append(
                        {
                            "frame": int(frame),
                            "idA": int(agent_a["id"]),
                            "idB": int(agent_b["id"]),
                            "xA_tplus1": float(agent_a["x_tplus1"]),
                            "yA_tplus1": float(agent_a["y_tplus1"]),
                            "distAB_t": pair_distance,
                            "is_overtake": True,
                        }
                    )
                if (
                    agent_b["dist_t"] > agent_a["dist_t"]
                    and agent_b["dist_tplus1"] < agent_a["dist_tplus1"]
                ):
                    overtake_events.append(
                        {
                            "frame": int(frame),
                            "idA": int(agent_b["id"]),
                            "idB": int(agent_a["id"]),
                            "xA_tplus1": float(agent_b["x_tplus1"]),
                            "yA_tplus1": float(agent_b["y_tplus1"]),
                            "distAB_t": pair_distance,
                            "is_overtake": True,
                        }
                    )

    df_events = df_sorted.iloc[0:0].copy() if not overtake_events else None
    if overtake_events:
        import pandas as pd

        df_events = pd.DataFrame(overtake_events)
        df_summary = (
            df_events.groupby(["idA", "idB"], as_index=False)["is_overtake"]
            .sum()
            .rename(columns={"is_overtake": "num_overtakes"})
        )
        return df_events, df_summary

    import pandas as pd

    return pd.DataFrame(), pd.DataFrame()


def filter_candidate_overtakes(
    traj_df,
    time_distance_df,
    candidate_events,
    frame_rate: float,
    stability_window: int,
    speed_window: int,
    rel_speed_threshold: float,
):
    distance_lookup = build_distance_lookup(time_distance_df)
    confirmed_events: List[Dict[str, float]] = []
    for _, row in candidate_events.iterrows():
        frame = int(row["frame"])
        id_a = int(row["idA"])
        id_b = int(row["idB"])
        if not check_overtake_stability(
            distance_lookup,
            id_a,
            id_b,
            start_frame=frame + 1,
            window=stability_window,
        ):
            continue
        speed_a = compute_speed_mps(
            traj_df,
            agent_id=id_a,
            start_frame=frame + 1,
            frame_rate=frame_rate,
            window=speed_window,
        )
        speed_b = compute_speed_mps(
            traj_df,
            agent_id=id_b,
            start_frame=frame + 1,
            frame_rate=frame_rate,
            window=speed_window,
        )
        if abs(speed_a - speed_b) < rel_speed_threshold:
            continue
        confirmed_event = row.to_dict()
        confirmed_event["speedA_mps"] = speed_a
        confirmed_event["speedB_mps"] = speed_b
        confirmed_events.append(confirmed_event)

    import pandas as pd

    return pd.DataFrame(confirmed_events)


def compute_model_rows(
    model: str,
    sqlite_path: Path,
    t_min: float,
    t_max: float,
    distance_threshold: float,
    stability_window: int,
    speed_window: int,
    rel_speed_threshold: float,
):
    import pedpy
    import pandas as pd

    config_path = find_matching_config(sqlite_path)
    measurement_line = load_measurement_line(config_path)
    traj = pedpy.load_trajectory_from_jupedsim_sqlite(sqlite_path)
    walkable_area = pedpy.load_walkable_area_from_jupedsim_sqlite(sqlite_path)

    traj_df = traj.data.copy()
    time_distance_df = pedpy.compute_time_distance_line(
        traj_data=traj,
        measurement_line=measurement_line,
    )
    candidate_events, _ = compute_overtakes_using_time_distance(
        traj_df=traj_df,
        time_distance_df=time_distance_df,
        distance_threshold=distance_threshold,
    )
    if candidate_events.empty:
        return [], walkable_area.polygon

    stable_events = filter_candidate_overtakes(
        traj_df=traj_df,
        time_distance_df=time_distance_df,
        candidate_events=candidate_events,
        frame_rate=float(traj.frame_rate),
        stability_window=stability_window,
        speed_window=speed_window,
        rel_speed_threshold=rel_speed_threshold,
    )
    if stable_events.empty:
        return [], walkable_area.polygon

    stable_events = stable_events.copy()
    stable_events["time"] = stable_events["frame"] / float(traj.frame_rate)
    stable_events = stable_events[
        (stable_events["time"] >= t_min) & (stable_events["time"] <= t_max)
    ]
    if stable_events.empty:
        return [], walkable_area.polygon

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
                "model": model,
                "source_file": str(sqlite_path),
                "config_file": str(config_path),
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
    return rows, walkable_area.polygon


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def tagged_filename(stem: str, suffix: str, tag: str) -> str:
    return f"{stem}_{tag}{suffix}" if tag else f"{stem}{suffix}"


def build_summary_rows(detail_rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    rows_by_model: Dict[str, List[Dict[str, float]]] = {}
    for row in detail_rows:
        rows_by_model.setdefault(str(row["model"]), []).append(row)

    summary_rows: List[Dict[str, float]] = []
    for model, rows in sorted(rows_by_model.items()):
        pair_count = len({(int(row["idA"]), int(row["idB"])) for row in rows})
        summary_rows.append(
            {
                "model": model,
                "num_stable_overtake_events": len(rows),
                "num_overtaking_pairs": pair_count,
                "mean_pair_distance_m": (
                    sum(float(row["distAB_t"]) for row in rows) / len(rows)
                ),
                "mean_relative_speed_mps": (
                    sum(
                        abs(float(row["speedA_mps"]) - float(row["speedB_mps"]))
                        for row in rows
                    )
                    / len(rows)
                ),
            }
        )
    return summary_rows


def plot_heatmaps(
    rows_by_model: Dict[str, List[Dict[str, float]]],
    geometries: Dict[str, object],
    output_dir: Path,
    bins: int,
    vmax: Optional[float],
    tag: str,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        print(f"Skipping plots: {exc}")
        return False

    if not rows_by_model:
        return False

    figure, axes = plt.subplots(
        len(rows_by_model),
        1,
        figsize=(10, 3.6 * len(rows_by_model)),
        sharex=False,
        sharey=False,
    )
    if len(rows_by_model) == 1:
        axes = [axes]

    computed_vmax = vmax
    if computed_vmax is None:
        maxima: List[float] = []
        for model, rows in rows_by_model.items():
            geometry = geometries[model]
            minx, miny, maxx, maxy = geometry.bounds
            x_bins = np.linspace(minx, maxx, bins)
            y_bins = np.linspace(miny, maxy, bins)
            heatmap, _, _ = np.histogram2d(
                [float(row["x"]) for row in rows],
                [float(row["y"]) for row in rows],
                bins=(x_bins, y_bins),
            )
            maxima.append(float(heatmap.max()) if heatmap.size else 0.0)
        computed_vmax = max(maxima) if maxima else 1.0

    for axis, (model, rows) in zip(axes, rows_by_model.items()):
        geometry = geometries[model]
        minx, miny, maxx, maxy = geometry.bounds
        dx = (maxx - minx) * 0.02
        dy = (maxy - miny) * 0.02
        x_bins = np.linspace(minx - dx, maxx + dx, bins)
        y_bins = np.linspace(miny - dy, maxy + dy, bins)
        heatmap, _, _ = np.histogram2d(
            [float(row["x"]) for row in rows],
            [float(row["y"]) for row in rows],
            bins=(x_bins, y_bins),
        )
        image = axis.imshow(
            heatmap.T,
            extent=[minx - dx, maxx + dx, miny - dy, maxy + dy],
            origin="lower",
            cmap="inferno",
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=computed_vmax,
        )
        x_ext, y_ext = geometry.exterior.xy
        axis.plot(x_ext, y_ext, color="white", linewidth=1.5)
        for interior in geometry.interiors:
            x_int, y_int = interior.xy
            axis.plot(x_int, y_int, color="white", linewidth=1.0)
        axis.set_title(model)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")

    figure.colorbar(image, ax=axes, label="Number of overtakes")
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / tagged_filename("overtaking_heatmaps", ".png", tag), dpi=200)
    plt.close(figure)
    return True


def main() -> None:
    args = parse_args()
    overrides = parse_input_overrides(args.input)
    requested_models = [normalize_model_name(model) for model in args.models]

    detail_rows_by_model: Dict[str, List[Dict[str, float]]] = {}
    geometries: Dict[str, object] = {}
    missing_models: List[str] = []

    for model in requested_models:
        sqlite_path = overrides.get(model) or discover_latest_sqlite(model, args.search_dir)
        if sqlite_path is None:
            missing_models.append(model)
            continue
        rows, geometry = compute_model_rows(
            model=model,
            sqlite_path=sqlite_path,
            t_min=args.t_min,
            t_max=args.t_max,
            distance_threshold=args.distance_threshold,
            stability_window=args.stability_window,
            speed_window=args.speed_window,
            rel_speed_threshold=args.rel_speed_threshold,
        )
        detail_rows_by_model[model] = rows
        geometries[model] = geometry

    if not detail_rows_by_model:
        raise SystemExit("No simulation sqlite files found for the requested models.")

    detail_rows = [row for rows in detail_rows_by_model.values() for row in rows]
    summary_rows = build_summary_rows(detail_rows)
    output_dir = Path(args.output_dir)
    write_csv(
        output_dir / tagged_filename("overtaking_details", ".csv", args.tag),
        detail_rows,
        [
            "model",
            "source_file",
            "config_file",
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
        output_dir / tagged_filename("overtaking_summary", ".csv", args.tag),
        summary_rows,
        [
            "model",
            "num_stable_overtake_events",
            "num_overtaking_pairs",
            "mean_pair_distance_m",
            "mean_relative_speed_mps",
        ],
    )
    created_plots = plot_heatmaps(
        {model: rows for model, rows in detail_rows_by_model.items() if rows},
        geometries,
        output_dir,
        bins=args.bins,
        vmax=args.vmax,
        tag=args.tag,
    )

    if missing_models:
        print("Skipped models without input files:", ", ".join(sorted(missing_models)))
    if not created_plots:
        print("CSV outputs were written, but no figures were created.")
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
