"""Extract initial agent positions from CROMA experimental trajectories.

For each requested run, picks the frame in which the maximum number of
participants is simultaneously visible, and writes (x, y) per agent
into a 2-column CSV that simulation.py can consume via the
'init_positions_file' key.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAJ_DIR = PROJECT_ROOT.parent / "trajectories_croma"
DEFAULT_RUNS = ["1C060", "2C020", "2C070", "2C120"]


def find_traj_file(run: str) -> Path:
    candidates = sorted(TRAJ_DIR.glob(f"{run}_*Combined.txt"))
    if not candidates:
        raise FileNotFoundError(f"No CROMA trajectory file for run {run} in {TRAJ_DIR}")
    return candidates[0]


def parse_rows(path: Path) -> List[Tuple[int, int, float, float]]:
    rows: List[Tuple[int, int, float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                pid = int(float(parts[0]))
                frame = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
            except ValueError:
                continue
            rows.append((pid, frame, x, y))
    return rows


def first_appearance_positions(
    rows: List[Tuple[int, int, float, float]],
    y_max: float = 18.85,
    min_dist: float = 0.4,
) -> List[Tuple[float, float]]:
    """Return each agent at the earliest frame they appear with y < y_max.

    Agents whose first-appearance position lies within ``min_dist`` of an
    already-accepted position are dropped, so JuPedSim's spacing
    constraint (~0.4m centre-to-centre) is satisfied at t=0.
    """
    earliest: Dict[int, Tuple[int, float, float]] = {}
    for pid, frame, x, y in rows:
        if y >= y_max:
            continue
        prev = earliest.get(pid)
        if prev is None or frame < prev[0]:
            earliest[pid] = (frame, x, y)

    accepted: List[Tuple[float, float]] = []
    md2 = min_dist * min_dist
    for _, x, y in sorted(earliest.values()):
        if all((x - ax) ** 2 + (y - ay) ** 2 >= md2 for ax, ay in accepted):
            accepted.append((x, y))
    return accepted


def write_csv(path: Path, positions: List[Tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for x, y in positions:
            w.writerow([f"{x:.4f}", f"{y:.4f}"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    p.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "files" / "init_croma"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        traj = find_traj_file(run)
        rows = parse_rows(traj)
        positions = first_appearance_positions(rows)
        out = args.output_dir / f"init_{run}.csv"
        write_csv(out, positions)
        print(f"{run}: n_agents={len(positions)} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
