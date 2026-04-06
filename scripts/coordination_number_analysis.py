"""Analyze Delaunay-based coordination numbers for motivation model outputs."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.coordination_number import coordination_numbers

MODEL_ALIASES = {
    "TOGETHER": "PVE",
    "ALL": "PVE",
    "NO_MOTIVATION": "BASE_MODEL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute coordination numbers Nn from Delaunay neighbors and compare "
            "their discrete distributions across motivation models."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["PVE", "BASE_MODEL"],
        help="Models to analyze. Use PVE for the combined model.",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help="Explicitly map a model to a motivation csv file.",
    )
    parser.add_argument(
        "--search-dir",
        action="append",
        default=["files", "files/variations"],
        help="Directories searched when an explicit input path is not provided.",
    )
    parser.add_argument(
        "--t-min",
        type=float,
        default=10.0,
        help="Minimum time included in the dynamics analysis.",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=20.0,
        help="Maximum time included in the dynamics analysis.",
    )
    parser.add_argument(
        "--output-dir",
        default="coordination_number_results",
        help="Directory for csv summaries and figures.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix added to output filenames.",
    )
    return parser.parse_args()


def normalize_model_name(name: str) -> str:
    model = name.strip().upper()
    return MODEL_ALIASES.get(model, model)


def parse_input_overrides(items: Sequence[str]) -> Dict[str, Path]:
    overrides: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --input value '{item}'. Expected MODEL=PATH.")
        model_name, raw_path = item.split("=", 1)
        overrides[normalize_model_name(model_name)] = Path(raw_path).expanduser()
    return overrides


def discover_latest_file(model: str, search_dirs: Sequence[str]) -> Optional[Path]:
    pattern = f"base_{model}_*_motivation.csv"
    direct_name = f"base_{model}_motivation.csv"
    candidates: List[Path] = []

    for directory in search_dirs:
        base_dir = Path(directory)
        if not base_dir.exists():
            continue
        direct_path = base_dir / direct_name
        if direct_path.exists():
            candidates.append(direct_path)
        candidates.extend(sorted(base_dir.glob(pattern)))

    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_motivation_rows(
    path: Path, t_min: float, t_max: float
) -> Dict[int, List[Dict[str, Any]]]:
    frames: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        for row in reader:
            time_value = float(row["time"])
            if time_value < t_min or time_value > t_max:
                continue
            frame = int(row["frame"])
            frames[frame].append(
                {
                    "frame": frame,
                    "id": int(row["id"]),
                    "time": time_value,
                    "motivation": float(row["motivation"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                }
            )
    return dict(sorted(frames.items()))


def build_coordination_rows(
    model: str, path: Path, t_min: float, t_max: float
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    frames = read_motivation_rows(path, t_min=t_min, t_max=t_max)
    for frame, frame_rows in frames.items():
        positions = {
            int(entry["id"]): (float(entry["x"]), float(entry["y"]))
            for entry in frame_rows
        }
        nn_by_agent = coordination_numbers(positions)
        for entry in frame_rows:
            rows.append(
                {
                    "model": model,
                    "source_file": str(path),
                    "frame": frame,
                    "id": int(entry["id"]),
                    "time": float(entry["time"]),
                    "x": float(entry["x"]),
                    "y": float(entry["y"]),
                    "motivation": float(entry["motivation"]),
                    "coordination_number": int(nn_by_agent.get(int(entry["id"]), 0)),
                }
            )
    return rows


def write_csv(
    path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def tagged_filename(stem: str, suffix: str, tag: str) -> str:
    return f"{stem}_{tag}{suffix}" if tag else f"{stem}{suffix}"


def build_time_summary(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, float], List[float]] = defaultdict(list)
    for row in rows:
        buckets[(str(row["model"]), float(row["time"]))].append(
            float(row["coordination_number"])
        )

    summary: List[Dict[str, Any]] = []
    for (model, time_value), values in sorted(buckets.items()):
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        summary.append(
            {
                "model": model,
                "time": time_value,
                "mean_coordination_number": mean_value,
                "std_coordination_number": variance**0.5,
                "n_agents": len(values),
            }
        )
    return summary


def build_distribution_summary(
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, int], int] = defaultdict(int)
    totals: Dict[str, int] = defaultdict(int)

    for row in rows:
        model = str(row["model"])
        coordination_number = int(row["coordination_number"])
        buckets[(model, coordination_number)] += 1
        totals[model] += 1

    summary: List[Dict[str, Any]] = []
    for (model, coordination_number), count in sorted(buckets.items()):
        total = totals[model]
        summary.append(
            {
                "model": model,
                "coordination_number": coordination_number,
                "count": count,
                "probability": count / total if total else 0.0,
            }
        )
    return summary


def plot_results(
    rows_by_model: Dict[str, List[Dict[str, Any]]],
    summary_rows: Sequence[Dict[str, Any]],
    distribution_rows: Sequence[Dict[str, Any]],
    output_dir: Path,
    tag: str,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        print(f"Skipping plots: {exc}")
        return False

    if distribution_rows:
        selected_models = [
            model for model in ["BASE_MODEL", "PVE"] if model in rows_by_model
        ]
        if not selected_models:
            return False

        probabilities_by_model: Dict[str, Dict[int, float]] = defaultdict(dict)
        coordination_values = set()
        for row in distribution_rows:
            model = str(row["model"])
            if model not in selected_models:
                continue
            coordination_number = int(row["coordination_number"])
            coordination_values.add(coordination_number)
            probabilities_by_model[model][coordination_number] = float(row["probability"])

        x_values = sorted(coordination_values)
        offsets = np.linspace(-0.18, 0.18, num=len(selected_models))
        width = 0.32

        figure, axis = plt.subplots(figsize=(8, 4.5))
        for offset, model in zip(offsets, selected_models):
            axis.bar(
                np.asarray(x_values, dtype=float) + float(offset),
                [probabilities_by_model[model].get(x_value, 0.0) for x_value in x_values],
                width=width,
                label=model,
                alpha=0.85,
            )

        axis.set_xlabel(r"coordination number $N_n$")
        axis.set_ylabel("probability")
        axis.set_xticks(x_values)
        axis.grid(alpha=0.2, axis="y")
        axis.legend()
        figure.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_dir
            / tagged_filename(
                "coordination_number_distribution_no_motivation_vs_pve",
                ".png",
                tag,
            ),
            dpi=200,
        )
        plt.close(figure)

    return True


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    overrides = parse_input_overrides(args.input)
    requested_models = [normalize_model_name(model) for model in args.models]

    rows_by_model: Dict[str, List[Dict[str, Any]]] = {}
    missing_models: List[str] = []

    for model in requested_models:
        input_path = overrides.get(model) or discover_latest_file(
            model, args.search_dir
        )
        if input_path is None:
            missing_models.append(model)
            continue
        rows_by_model[model] = build_coordination_rows(
            model=model,
            path=input_path,
            t_min=args.t_min,
            t_max=args.t_max,
        )

    if not rows_by_model:
        available = ", ".join(sorted(set(requested_models)))
        raise SystemExit(f"No input files found for models: {available}")

    detail_rows = [row for rows in rows_by_model.values() for row in rows]
    summary_rows = build_time_summary(detail_rows)
    distribution_rows = build_distribution_summary(detail_rows)

    write_csv(
        output_dir / tagged_filename("coordination_number_details", ".csv", args.tag),
        detail_rows,
        [
            "model",
            "source_file",
            "frame",
            "id",
            "time",
            "x",
            "y",
            "motivation",
            "coordination_number",
        ],
    )
    write_csv(
        output_dir / tagged_filename("coordination_number_summary", ".csv", args.tag),
        summary_rows,
        [
            "model",
            "time",
            "mean_coordination_number",
            "std_coordination_number",
            "n_agents",
        ],
    )
    write_csv(
        output_dir / tagged_filename(
            "coordination_number_distribution_summary", ".csv", args.tag
        ),
        distribution_rows,
        [
            "model",
            "coordination_number",
            "count",
            "probability",
        ],
    )
    created_plots = plot_results(
        rows_by_model,
        summary_rows,
        distribution_rows,
        output_dir,
        args.tag,
    )

    if missing_models:
        print(
            "Skipped models without input files:",
            ", ".join(sorted(missing_models)),
        )
    if not created_plots:
        print("CSV outputs were written, but no figures were created.")
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
