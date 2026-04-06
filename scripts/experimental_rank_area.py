"""Crossing-order vs Voronoi area analysis for the CROMA experimental runs.

For each CROMA trajectory, agents are ranked by the first frame in which
they cross the door line (y >= 20). The crossing rank is then related to
the agent's mean Voronoi area over the trajectory, giving the
experimental counterpart of the simulation's rank--area analysis.

Outputs:
 - per-scenario CSV with (id, crossing_order, density, area)
 - combined CSV with all scenarios (for downstream aggregation)
 - summary CSV with Spearman rho, OLS slope, tail ratio per scenario
 - per-scenario scatter plot (rank vs area)
 - overlay plot comparing all scenarios
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utilities import calculate_crossing_density


DEFAULT_INPUT_DIR = PROJECT_ROOT.parent / "trajectories_croma"

DEFAULT_GEOMETRY_XML = PROJECT_ROOT.parent / "trajectories_croma" / "delme" / "geometry.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crossing-order vs area analysis for CROMA trajectories."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing *_Combined.txt trajectory files.",
    )
    parser.add_argument(
        "--geometry",
        default=str(DEFAULT_GEOMETRY_XML),
        help="CROMA geometry.xml used to build the walkable area.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            PROJECT_ROOT / "files" / "experimental_rank_area_results"
        ),
        help="Where to write CSVs and figures.",
    )
    return parser.parse_args()


def scenario_from_filename(path: Path) -> str:
    # 1C070_cam6_cam5_frameshift0_Combined.txt -> 1C070
    return path.name.split("_")[0]


def load_walkable_area(_geometry_xml: Path) -> Any:
    """Build the CROMA walkable area (outer box with corridor-wall cutouts).

    Uses the exact polygon construction from ``index_distance.ipynb``:
    a rectangular outer ring with three interior cutouts (left wall,
    right wall, bottom strip) that represent the corridor walls around
    the entrance funnel leading to the door at y~21.
    """
    import pedpy
    from shapely.geometry import Polygon

    exterior = [
        (-8.88, -11.1),
        (8.3, -11.1),
        (8.3, 27.95),
        (-8.88, 27.95),
        (-8.88, -11.1),
    ]
    interior_rings = [
        # Left cutout
        [
            (-7, -11), (-3.57, -3), (-3.57, 19.57),
            (-1.52, 19.57), (-1.37, 19.71), (-0.87, 19.71),
            (-0.72, 19.57), (-0.42, 19.57), (-0.42, 21.23),
            (-0.72, 21.23), (-0.87, 21.09), (-1.37, 21.09),
            (-1.52, 21.23), (-1.67, 21.23), (-1.67, 21.18),
            (-1.545, 21.18), (-1.42, 21.065), (-1.42, 19.735),
            (-1.545, 19.62), (-3.62, 19.62), (-3.59, -3), (-7, -11),
        ],
        # Right cutout
        [
            (7, -11), (3.57, -3), (3.64, 19.64),
            (1.47, 19.57), (1.32, 19.71), (0.82, 19.71),
            (0.67, 19.57), (0.38, 19.57), (0.38, 21.23),
            (0.67, 21.23), (0.82, 21.09), (1.32, 21.09),
            (1.47, 21.23), (1.62, 21.23), (1.62, 21.18),
            (1.495, 21.18), (1.37, 21.065), (1.37, 19.735),
            (1.495, 19.62), (3.69, 19.69), (3.62, -3), (7, -11),
        ],
        # Bottom strip
        [
            (-6.8, -10.8), (6.8, -10.8), (6.8, -10.6),
            (-6.8, -10.6), (-6.8, -10.8),
        ],
    ]
    geometry = Polygon(exterior, interior_rings)
    return pedpy.WalkableArea(geometry)


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    rx = _rankdata(xs)
    ry = _rankdata(ys)
    return _pearson(rx, ry)


def _rankdata(values: Sequence[float]) -> List[float]:
    ordered = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and values[ordered[j + 1]] == values[ordered[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[ordered[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs)
    dy = sum((y - my) ** 2 for y in ys)
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx * dy) ** 0.5


def ols_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


def tail_ratio(ranks: Sequence[int], areas: Sequence[float]) -> float:
    if not ranks:
        return 0.0
    n = len(ranks)
    q = max(1, n // 4)
    paired = sorted(zip(ranks, areas), key=lambda t: t[0])
    bottom = [a for _, a in paired[:q]]
    top = [a for _, a in paired[-q:]]
    bm = sum(bottom) / len(bottom) if bottom else 0.0
    return (sum(top) / len(top)) / bm if bm else 0.0


def compute_one_scenario(
    txt_path: Path, walkable_area: Any
) -> Tuple[str, List[Dict[str, Any]]]:
    scenario = scenario_from_filename(txt_path)
    df_merged, _, _, _, _ = calculate_crossing_density(
        str(txt_path),
        walkable_area=walkable_area,
        file_type="experiment",
        title=scenario,
    )
    rows: List[Dict[str, Any]] = []
    indexed = df_merged.rename_axis("id").reset_index().to_dict("records")
    for row in indexed:
        rows.append(
            {
                "scenario": scenario,
                "id": int(row["id"]),
                "crossing_order": int(row["order"]),
                "density": float(row["density"]),
                "area": float(row["area"]),
            }
        )
    return scenario, rows


def write_csv(
    path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_scenarios(
    by_scenario: Dict[str, List[Dict[str, Any]]], output_dir: Path
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib missing; skipping plots.")
        return

    # Per-scenario scatter
    for scenario, rows in by_scenario.items():
        ranks = [int(r["crossing_order"]) for r in rows]
        areas = [float(r["area"]) for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(ranks, areas, s=20, alpha=0.7)
        ax.set_xlabel("rank (crossing order)")
        ax.set_ylabel(r"Voronoi area [$\mathrm{m}^2$]")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / f"croma_rank_area_{scenario}.png", dpi=200)
        plt.close(fig)

    # Overlay: all scenarios
    fig, ax = plt.subplots(figsize=(8, 5))
    for scenario, rows in sorted(by_scenario.items()):
        ranks = [int(r["crossing_order"]) for r in rows]
        areas = [float(r["area"]) for r in rows]
        paired = sorted(zip(ranks, areas), key=lambda t: t[0])
        ax.plot(
            [r for r, _ in paired],
            [a for _, a in paired],
            label=scenario,
            alpha=0.75,
            linewidth=1.2,
        )
    ax.set_xlabel("rank (crossing order)")
    ax.set_ylabel(r"Voronoi area [$\mathrm{m}^2$]")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "croma_rank_area_all.png", dpi=200)
    plt.close(fig)

    # Focused comparison: 1C070 vs 2C070 (same door, different motivation),
    # plus 1C060 if present, to mirror the simulation base-vs-PVE framing.
    focus_pairs = [
        ("1C070", "2C070"),  # normal vs high motivation, 70cm door
        ("1C060", "2C070"),  # 60cm normal vs 70cm high (closest available)
    ]
    for normal, high in focus_pairs:
        if normal not in by_scenario or high not in by_scenario:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for scenario, colour, label in [
            (normal, "tab:blue", f"{normal} (normal motivation)"),
            (high, "tab:orange", f"{high} (high motivation)"),
        ]:
            rows = by_scenario[scenario]
            ranks = [int(r["crossing_order"]) for r in rows]
            areas = [float(r["area"]) for r in rows]
            paired = sorted(zip(ranks, areas), key=lambda t: t[0])
            ax.plot(
                [r for r, _ in paired],
                [a for _, a in paired],
                color=colour,
                label=label,
                linewidth=1.8,
            )
        ax.set_xlabel("rank (crossing order)")
        ax.set_ylabel(r"Voronoi area [$\mathrm{m}^2$]")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output_dir / f"croma_rank_area_{normal}_vs_{high}.png", dpi=200
        )
        plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    walkable_area = load_walkable_area(Path(args.geometry))

    txts = sorted(input_dir.glob("*_Combined.txt"))
    if not txts:
        raise SystemExit(f"No *_Combined.txt files found in {input_dir}")

    all_rows: List[Dict[str, Any]] = []
    by_scenario: Dict[str, List[Dict[str, Any]]] = {}
    summary_rows: List[Dict[str, Any]] = []

    skipped: List[Tuple[str, str]] = []
    for txt_path in txts:
        try:
            scenario, rows = compute_one_scenario(txt_path, walkable_area)
        except IndexError as exc:
            # pedpy Voronoi clipping fails when trajectory points fall
            # outside the walkable area (known issue with 2C130/2C150
            # and occasionally other CROMA files in newer pedpy).
            name = scenario_from_filename(txt_path)
            skipped.append((name, str(exc)))
            print(f"WARN: skipped {name} (invalid trajectory points for Voronoi)")
            continue
        by_scenario[scenario] = rows
        all_rows.extend(rows)
        write_csv(
            output_dir / f"croma_rank_area_details_{scenario}.csv",
            rows,
            ["scenario", "id", "crossing_order", "density", "area"],
        )

        ranks = [int(r["crossing_order"]) for r in rows]
        areas = [float(r["area"]) for r in rows]
        summary_rows.append(
            {
                "scenario": scenario,
                "n_agents": len(rows),
                "spearman_rho": spearman_rho(ranks, areas),
                "slope": ols_slope(ranks, areas),
                "tail_ratio": tail_ratio(ranks, areas),
                "source_file": str(txt_path),
            }
        )
        print(
            f"{scenario}: n={len(rows):3d} "
            f"rho={summary_rows[-1]['spearman_rho']:+.3f} "
            f"slope={summary_rows[-1]['slope']:+.4f} "
            f"tail_ratio={summary_rows[-1]['tail_ratio']:.2f}"
        )

    write_csv(
        output_dir / "croma_rank_area_details_all.csv",
        all_rows,
        ["scenario", "id", "crossing_order", "density", "area"],
    )
    write_csv(
        output_dir / "croma_rank_area_summary.csv",
        summary_rows,
        ["scenario", "n_agents", "spearman_rho", "slope", "tail_ratio", "source_file"],
    )

    plot_scenarios(by_scenario, output_dir)
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
