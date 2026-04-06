"""Compute crossing density summaries and plots for simulation runs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utilities import calculate_crossing_density, plot_crossing_order_vs_area

MODEL_ALIASES = {
    "TOGETHER": "PVE",
    "ALL": "PVE",
    "E": "SE",
    "NO_MOTIVATION": "BASE_MODEL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute crossing order vs Voronoi area from simulation sqlite outputs."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["P", "V", "SE", "PVE", "BASE_MODEL"],
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
        "--output-dir",
        default="crossing_density_results",
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


def write_csv(
    path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def tagged_filename(stem: str, suffix: str, tag: str) -> str:
    return f"{stem}_{tag}{suffix}" if tag else f"{stem}{suffix}"


def main() -> None:
    args = parse_args()
    overrides = parse_input_overrides(args.input)
    requested_models = [normalize_model_name(model) for model in args.models]
    output_dir = Path(args.output_dir)

    detail_rows: List[Dict[str, object]] = []
    missing_models: List[str] = []
    skipped_models: List[str] = []

    for model in requested_models:
        sqlite_path = overrides.get(model) or discover_latest_sqlite(model, args.search_dir)
        if sqlite_path is None:
            missing_models.append(model)
            continue

        df_merged, _, _, _, _ = calculate_crossing_density(
            sqlite_path, walkable_area=None, file_type="simulation", title=model
        )
        if df_merged.empty:
            skipped_models.append(model)
            continue

        plot_crossing_order_vs_area(
            df_merged,
            tagged_filename(f"crossing_order_vs_area_{model.lower()}", "", args.tag),
            title=model,
            output_dir=str(output_dir),
        )

        indexed_rows = df_merged.rename_axis("id").reset_index().to_dict("records")
        for row in indexed_rows:
            detail_rows.append(
                {
                    "model": model,
                    "source_file": str(sqlite_path),
                    "id": int(row["id"]),
                    "order": int(row["order"]),
                    "density": float(row["density"]),
                    "area": float(row["area"]),
                }
            )

    if detail_rows:
        write_csv(
            output_dir / tagged_filename("crossing_density_details", ".csv", args.tag),
            detail_rows,
            ["model", "source_file", "id", "order", "density", "area"],
        )
    else:
        raise SystemExit("No crossings found in the requested simulation sqlite files.")

    if missing_models:
        print("Skipped models without input files:", ", ".join(sorted(missing_models)))
    if skipped_models:
        print("Skipped models without crossings:", ", ".join(sorted(skipped_models)))
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
