"""Compute final rank summaries and plots for simulation runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utilities import calculate_final_rank_density, plot_final_rank_vs_area


MODEL_ALIASES = {
    "TOGETHER": "PVE",
    "ALL": "PVE",
    "E": "SE",
    "NO_MOTIVATION": "BASE_MODEL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute final rank vs Voronoi area from simulation sqlite outputs."
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
        default="final_rank_results",
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


def load_door_center(config_path: Path) -> Tuple[float, float]:
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    door_vertices = data["motivation_parameters"]["motivation_doors"][0]["vertices"]
    x1, y1 = door_vertices[0]
    x2, y2 = door_vertices[1]
    return (0.5 * float(x1 + x2), 0.5 * float(y1 + y2))


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

    for model in requested_models:
        sqlite_path = overrides.get(model) or discover_latest_sqlite(model, args.search_dir)
        if sqlite_path is None:
            missing_models.append(model)
            continue

        config_path = find_matching_config(sqlite_path)
        door_center = load_door_center(config_path)
        df_merged, _, _, _, _ = calculate_final_rank_density(
            sqlite_path,
            walkable_area=None,
            door_center=door_center,
            file_type="simulation",
            title=model,
        )
        if df_merged.empty:
            continue

        plot_final_rank_vs_area(
            df_merged,
            tagged_filename(f"final_rank_vs_area_{model.lower()}", "", args.tag),
            title=model,
            output_dir=str(output_dir),
        )

        indexed_rows = df_merged.rename_axis("id").reset_index().to_dict("records")
        for row in indexed_rows:
            detail_rows.append(
                {
                    "model": model,
                    "source_file": str(sqlite_path),
                    "config_file": str(config_path),
                    "id": int(row["id"]),
                    "final_rank": int(row["final_rank"]),
                    "density": float(row["density"]),
                    "area": float(row["area"]),
                }
            )

    if detail_rows:
        write_csv(
            output_dir / tagged_filename("final_rank_details", ".csv", args.tag),
            detail_rows,
            ["model", "source_file", "config_file", "id", "final_rank", "density", "area"],
        )
    else:
        raise SystemExit("No simulation sqlite files found for final rank analysis.")

    if missing_models:
        print("Skipped models without input files:", ", ".join(sorted(missing_models)))
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
