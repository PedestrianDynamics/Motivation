"""Analyze Delaunay-based coordination numbers for experimental trajectory files."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.coordination_number import coordination_numbers

DEFAULT_EXPERIMENT_DIR = Path("/Users/chraibi/workspace/Writing/Motivation/CroMa videos")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute coordination numbers Nn from Delaunay neighbors for "
            "experimental trajectory files."
        )
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
        default="coordination_number_results_experimental",
        help="Directory for csv summaries and figures.",
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
    paths = sorted(experiment_dir.glob("*_Combined.txt"))
    discovered: Dict[str, Path] = {}
    for path in paths:
        label = path.stem.split("_cam", 1)[0]
        discovered[label] = path
    return discovered


def parse_frame_rate(path: Path) -> float:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("# framerate:"):
                tokens = line.strip().split()
                return float(tokens[2])
    raise ValueError(f"Could not determine frame rate from {path}")


def read_experiment_rows(
    label: str,
    path: Path,
    t_min: float,
    t_max: float,
) -> Dict[int, List[Dict[str, float]]]:
    fps = parse_frame_rate(path)
    frames: Dict[int, List[Dict[str, float]]] = defaultdict(list)

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue

            agent_id = int(parts[0])
            frame = int(parts[1])
            time_value = frame / fps
            if time_value < t_min or time_value > t_max:
                continue

            frames[frame].append(
                {
                    "label": label,
                    "frame": frame,
                    "id": agent_id,
                    "time": time_value,
                    "x": float(parts[2]),
                    "y": float(parts[3]),
                }
            )

    return dict(sorted(frames.items()))


def build_coordination_rows(
    label: str,
    path: Path,
    t_min: float,
    t_max: float,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    frames = read_experiment_rows(label=label, path=path, t_min=t_min, t_max=t_max)
    for frame, frame_rows in frames.items():
        positions = {
            int(entry["id"]): (float(entry["x"]), float(entry["y"]))
            for entry in frame_rows
        }
        nn_by_agent = coordination_numbers(positions)
        for entry in frame_rows:
            rows.append(
                {
                    "label": label,
                    "source_file": str(path),
                    "frame": frame,
                    "id": int(entry["id"]),
                    "time": float(entry["time"]),
                    "x": float(entry["x"]),
                    "y": float(entry["y"]),
                    "coordination_number": int(nn_by_agent.get(int(entry["id"]), 0)),
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


def build_time_summary(rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    buckets: Dict[Tuple[str, float], List[float]] = defaultdict(list)
    for row in rows:
        buckets[(str(row["label"]), float(row["time"]))].append(
            float(row["coordination_number"])
        )

    summary: List[Dict[str, float]] = []
    for (label, time_value), values in sorted(buckets.items()):
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        summary.append(
            {
                "label": label,
                "time": time_value,
                "mean_coordination_number": mean_value,
                "std_coordination_number": variance ** 0.5,
                "n_agents": len(values),
            }
        )
    return summary


def build_distribution_summary(rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    buckets: Dict[Tuple[str, int], int] = defaultdict(int)
    totals: Dict[str, int] = defaultdict(int)

    for row in rows:
        label = str(row["label"])
        coordination_number = int(row["coordination_number"])
        buckets[(label, coordination_number)] += 1
        totals[label] += 1

    summary: List[Dict[str, float]] = []
    for (label, coordination_number), count in sorted(buckets.items()):
        total = totals[label]
        summary.append(
            {
                "label": label,
                "coordination_number": coordination_number,
                "count": count,
                "probability": count / total if total else 0.0,
            }
        )
    return summary


def _kde_curve(values: Sequence[float], xmin: float, xmax: float) -> Tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.array([]), np.array([])

    grid = np.linspace(xmin, xmax, 400)
    if array.size == 1 or float(np.std(array)) < 1e-12:
        bandwidth = 0.25
    else:
        bandwidth = max(1.06 * float(np.std(array)) * (array.size ** (-1.0 / 5.0)), 0.25)

    scaled = (grid[:, None] - array[None, :]) / bandwidth
    density = np.exp(-0.5 * scaled**2).sum(axis=1)
    density /= array.size * bandwidth * np.sqrt(2.0 * np.pi)
    return grid, density


def plot_results(
    rows_by_label: Dict[str, List[Dict[str, float]]],
    summary_rows: Sequence[Dict[str, float]],
    distribution_rows: Sequence[Dict[str, float]],
    output_dir: Path,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        print(f"Skipping plots: {exc}")
        return False

    if rows_by_label:
        figure, axes = plt.subplots(
            len(rows_by_label), 1, figsize=(10, 3.6 * len(rows_by_label)), sharex=True
        )
        if len(rows_by_label) == 1:
            axes = [axes]

        for axis, (label, rows) in zip(axes, rows_by_label.items()):
            times = [row["time"] for row in rows]
            agent_ids = [row["id"] for row in rows]
            coordination = [row["coordination_number"] for row in rows]
            scatter = axis.scatter(
                times,
                agent_ids,
                c=coordination,
                cmap="PiYG",
                s=10,
                vmin=min(coordination),
                vmax=max(coordination),
            )
            axis.set_title(label)
            axis.set_ylabel("#id")
            axis.grid(alpha=0.2)
            figure.colorbar(scatter, ax=axis, label=r"$N_n$")

        axes[-1].set_xlabel("time [s]")
        figure.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_dir / "experimental_coordination_number_timeseries.png", dpi=200)
        plt.close(figure)

    if summary_rows:
        figure, axis = plt.subplots(figsize=(10, 4.5))
        labels = []
        for row in summary_rows:
            if row["label"] not in labels:
                labels.append(row["label"])

        for label in labels:
            series = [row for row in summary_rows if row["label"] == label]
            times = [row["time"] for row in series]
            means = [row["mean_coordination_number"] for row in series]
            stds = [row["std_coordination_number"] for row in series]
            lower = [mean - std for mean, std in zip(means, stds)]
            upper = [mean + std for mean, std in zip(means, stds)]
            axis.plot(times, means, label=label)
            axis.fill_between(times, lower, upper, alpha=0.15)

        axis.set_xlabel("time [s]")
        axis.set_ylabel(r"mean $N_n$")
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_dir / "experimental_coordination_number_model_comparison.png", dpi=200)
        plt.close(figure)

    if distribution_rows:
        labels = []
        for row in distribution_rows:
            if row["label"] not in labels:
                labels.append(row["label"])

        all_coordination_values = [
            int(row["coordination_number"])
            for rows in rows_by_label.values()
            for row in rows
        ]
        xmin = min(all_coordination_values) - 0.5
        xmax = max(all_coordination_values) + 0.5

        figure, axis = plt.subplots(figsize=(8, 4.5))
        for label in labels:
            values = [
                int(row["coordination_number"])
                for row in rows_by_label[label]
            ]
            grid, density = _kde_curve(values, xmin=xmin, xmax=xmax)
            axis.plot(grid, density, label=label)

        axis.set_xlabel(r"$N_n$")
        axis.set_ylabel("density")
        axis.set_title("Experimental Coordination Number KDE")
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_dir / "experimental_coordination_number_distribution_kde_all.png", dpi=200)
        plt.close(figure)

    return True


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    overrides = parse_input_overrides(args.input)

    if overrides:
        inputs = overrides
    else:
        inputs = discover_experiment_files(args.experiment_dir)

    rows_by_label: Dict[str, List[Dict[str, float]]] = {}

    for label, path in inputs.items():
        rows_by_label[label] = build_coordination_rows(
            label=label,
            path=path,
            t_min=args.t_min,
            t_max=args.t_max,
        )

    if not rows_by_label:
        raise SystemExit("No experimental trajectory files found.")

    detail_rows = [row for rows in rows_by_label.values() for row in rows]
    summary_rows = build_time_summary(detail_rows)
    distribution_rows = build_distribution_summary(detail_rows)

    write_csv(
        output_dir / "experimental_coordination_number_details.csv",
        detail_rows,
        [
            "label",
            "source_file",
            "frame",
            "id",
            "time",
            "x",
            "y",
            "coordination_number",
        ],
    )
    write_csv(
        output_dir / "experimental_coordination_number_summary.csv",
        summary_rows,
        [
            "label",
            "time",
            "mean_coordination_number",
            "std_coordination_number",
            "n_agents",
        ],
    )
    write_csv(
        output_dir / "experimental_coordination_number_distribution_summary.csv",
        distribution_rows,
        [
            "label",
            "coordination_number",
            "count",
            "probability",
        ],
    )
    created_plots = plot_results(
        rows_by_label,
        summary_rows,
        distribution_rows,
        output_dir,
    )
    if not created_plots:
        print("CSV outputs were written, but no figures were created.")
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
