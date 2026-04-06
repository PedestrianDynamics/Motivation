"""Compare Voronoi density time series across simulation models using PedPy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_ALIASES = {
    "TOGETHER": "PVE",
    "ALL": "PVE",
    "NO_MOTIVATION": "BASE_MODEL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Voronoi density in the measurement area for simulation runs."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["PVE", "BASE_MODEL"],
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
        "--t-min",
        type=float,
        default=10.0,
        help="Minimum time included in the comparison.",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=300.0,
        help="Maximum time included in the comparison.",
    )
    parser.add_argument(
        "--output-dir",
        default="voronoi_density_results",
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


def load_measurement_area(config_path: Path) -> Any:
    import pedpy

    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return pedpy.MeasurementArea(data["measurement_area"]["vertices"])


def compute_density_rows(
    model: str,
    sqlite_path: Path,
    t_min: float,
    t_max: float,
) -> List[Dict[str, Any]]:
    import pedpy
    from pedpy.column_identifier import DENSITY_COL, TIME_COL

    config_path = find_matching_config(sqlite_path)
    measurement_area = load_measurement_area(config_path)
    traj = pedpy.load_trajectory_from_jupedsim_sqlite(sqlite_path)
    walkable_area = pedpy.load_walkable_area_from_jupedsim_sqlite(sqlite_path)

    individual = pedpy.compute_individual_voronoi_polygons(
        traj_data=traj,
        walkable_area=walkable_area,
    )
    density_voronoi, _ = pedpy.compute_voronoi_density(
        individual_voronoi_data=individual,
        measurement_area=measurement_area,
    )

    columns = set(density_voronoi.columns)
    time_column = TIME_COL if TIME_COL in columns else None
    density_column = DENSITY_COL if DENSITY_COL in columns else "density"
    frame_column = "frame" if "frame" in columns else None

    rows: List[Dict[str, Any]] = []
    for _, row in density_voronoi.iterrows():
        if time_column is not None:
            time_value = float(row[time_column])
        elif frame_column is not None:
            time_value = int(row[frame_column]) / float(traj.frame_rate)
        elif density_voronoi.index.name in {TIME_COL, "time"}:
            time_value = float(row.name)
        elif density_voronoi.index.name in {"frame", "frame_id"}:
            time_value = int(row.name) / float(traj.frame_rate)
        elif frame_column is None and time_column is None and density_voronoi.index.nlevels == 1:
            # PedPy may return the time axis as a plain unnamed index.
            time_value = float(row.name)
        else:
            raise KeyError(
                "Could not find a time axis in Voronoi density output. "
                f"Columns: {sorted(columns)}, index name: {density_voronoi.index.name!r}"
            )
        if time_value < t_min or time_value > t_max:
            continue
        rows.append(
            {
                "model": model,
                "source_file": str(sqlite_path),
                "config_file": str(config_path),
                "time": time_value,
                "voronoi_density": float(row[density_column]),
            }
        )
    return rows


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def tagged_filename(stem: str, suffix: str, tag: str) -> str:
    return f"{stem}_{tag}{suffix}" if tag else f"{stem}{suffix}"


def plot_density(
    rows_by_model: Dict[str, List[Dict[str, Any]]], output_dir: Path, tag: str
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        print(f"Skipping plots: {exc}")
        return False

    if "PVE" in rows_by_model and "BASE_MODEL" in rows_by_model:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for model in ["PVE", "BASE_MODEL"]:
            rows = rows_by_model[model]
            axis.plot(
                [row["time"] for row in rows],
                [row["voronoi_density"] for row in rows],
                label=model,
            )
        axis.set_xlabel("time [s]")
        axis.set_ylabel(r"Voronoi density [1/m$^2$]")
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_dir / tagged_filename("voronoi_density_base_model_vs_pve", ".png", tag),
            dpi=200,
        )
        plt.close(figure)
        return True

    return False


def main() -> None:
    args = parse_args()
    overrides = parse_input_overrides(args.input)
    requested_models = [normalize_model_name(model) for model in args.models]

    rows_by_model: Dict[str, List[Dict[str, Any]]] = {}
    missing_models: List[str] = []

    for model in requested_models:
        sqlite_path = overrides.get(model) or discover_latest_sqlite(model, args.search_dir)
        if sqlite_path is None:
            missing_models.append(model)
            continue
        rows_by_model[model] = compute_density_rows(
            model=model,
            sqlite_path=sqlite_path,
            t_min=args.t_min,
            t_max=args.t_max,
        )

    if not rows_by_model:
        raise SystemExit("No simulation sqlite files found for the requested models.")

    detail_rows = [row for rows in rows_by_model.values() for row in rows]
    output_dir = Path(args.output_dir)
    write_csv(
        output_dir / tagged_filename("voronoi_density_details", ".csv", args.tag),
        detail_rows,
        [
            "model",
            "source_file",
            "config_file",
            "time",
            "voronoi_density",
        ],
    )
    created_plots = plot_density(rows_by_model, output_dir, args.tag)

    if missing_models:
        print("Skipped models without input files:", ", ".join(sorted(missing_models)))
    if not created_plots:
        print("CSV outputs were written, but no figures were created.")
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
