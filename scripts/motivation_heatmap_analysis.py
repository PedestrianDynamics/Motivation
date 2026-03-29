"""Plot spatial motivation heatmaps for each model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

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
        description="Create spatial motivation heatmaps for one or more models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["P", "V", "E", "PVE", "NO_MOTIVATION"],
        help="Models to analyze.",
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
        default=["files", "files/variations", "files/base_runs"],
        help="Directories searched when an explicit input path is not provided.",
    )
    parser.add_argument(
        "--t-min",
        type=float,
        default=10.0,
        help="Minimum time included in the heatmap.",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=20.0,
        help="Maximum time included in the heatmap.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Number of spatial bins per axis.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.5,
        help="Gaussian smoothing width in heatmap bins.",
    )
    parser.add_argument(
        "--output-dir",
        default="motivation_heatmap_results",
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix added to the output filename.",
    )
    return parser.parse_args()


def normalize_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name.strip().upper(), name.strip().upper())


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


def load_rows(path: Path, t_min: float, t_max: float) -> List[Tuple[float, float, float]]:
    rows: List[Tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        for row in reader:
            time_value = float(row["time"])
            if time_value < t_min or time_value > t_max:
                continue
            rows.append((float(row["x"]), float(row["y"]), float(row["motivation"])))
    return rows


def build_heatmap(
    rows: Sequence[Tuple[float, float, float]],
    bins: int,
    xrange: Tuple[float, float],
    yrange: Tuple[float, float],
    sigma: float,
) -> np.ndarray:
    if not rows:
        return np.full((bins, bins), np.nan)

    x = np.array([row[0] for row in rows], dtype=float)
    y = np.array([row[1] for row in rows], dtype=float)
    motivation = np.array([row[2] for row in rows], dtype=float)

    weighted_sum, _, _ = np.histogram2d(
        x, y, bins=bins, range=[xrange, yrange], weights=motivation
    )
    counts, _, _ = np.histogram2d(x, y, bins=bins, range=[xrange, yrange])

    if sigma > 0:
        weighted_sum = _smooth_2d(weighted_sum, sigma)
        counts = _smooth_2d(counts, sigma)

    heatmap = np.divide(
        weighted_sum,
        counts,
        out=np.full_like(weighted_sum, np.nan, dtype=float),
        where=counts > 0,
    )
    # Mask locations with very little support after smoothing.
    support_threshold = max(float(np.nanmax(counts)) * 0.03, 1e-12)
    heatmap = np.where(counts >= support_threshold, heatmap, np.nan)
    return heatmap.T


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def _smooth_2d(array: np.ndarray, sigma: float) -> np.ndarray:
    kernel = _gaussian_kernel1d(sigma)
    padded_x = np.pad(array, ((kernel.size // 2, kernel.size // 2), (0, 0)), mode="edge")
    smooth_x = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="valid"), 0, padded_x)
    padded_y = np.pad(smooth_x, ((0, 0), (kernel.size // 2, kernel.size // 2)), mode="edge")
    smooth_xy = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded_y)
    return smooth_xy


def main() -> None:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(f"matplotlib is required to plot heatmaps: {exc}") from exc

    overrides = parse_input_overrides(args.input)
    requested_models = [normalize_model_name(model) for model in args.models]

    rows_by_model: Dict[str, List[Tuple[float, float, float]]] = {}
    for model in requested_models:
        input_path = overrides.get(model) or discover_latest_file(model, args.search_dir)
        if input_path is None:
            continue
        rows = load_rows(input_path, t_min=args.t_min, t_max=args.t_max)
        if rows:
            rows_by_model[model] = rows

    if not rows_by_model:
        raise SystemExit("No motivation csv files found for the requested models.")

    all_x = [row[0] for rows in rows_by_model.values() for row in rows]
    all_y = [row[1] for rows in rows_by_model.values() for row in rows]
    xrange = (min(all_x), max(all_x))
    yrange = (min(all_y), max(all_y))

    heatmaps = {
        model: build_heatmap(
            rows,
            bins=args.bins,
            xrange=xrange,
            yrange=yrange,
            sigma=args.sigma,
        )
        for model, rows in rows_by_model.items()
    }

    nmodels = len(heatmaps)
    figure, axes = plt.subplots(1, nmodels, figsize=(4.5 * nmodels, 5), squeeze=False)
    for axis, (model, heatmap) in zip(axes[0], heatmaps.items()):
        finite = heatmap[np.isfinite(heatmap)]
        if finite.size == 0:
            vmin = 0.0
            vmax = 1.0
        elif float(finite.max()) - float(finite.min()) < 1e-12:
            vmin = float(finite.min()) - 0.05
            vmax = float(finite.max()) + 0.05
        else:
            vmin = float(finite.min())
            vmax = float(finite.max())
        image = axis.imshow(
            heatmap,
            origin="lower",
            extent=(xrange[0], xrange[1], yrange[0], yrange[1]),
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        if finite.size == 0:
            axis.set_title(f"{model}\nno data")
        else:
            axis.set_title(f"{model}\n[{float(finite.min()):.2f}, {float(finite.max()):.2f}]")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        figure.colorbar(image, ax=axis, label="mean motivation")
    figure.tight_layout()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"motivation_heatmap_models_{args.tag}.png"
        if args.tag
        else "motivation_heatmap_models.png"
    )
    output_path = output_dir / filename
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
