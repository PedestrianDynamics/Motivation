"""Which mapping anchors set the discharge after the door opens (V, PVE)?

Diagnostic, not a fit. Starting from the point selected in the low-anchor
fit (low anchors: buffer 1.0 m, desired speed 1.2 m/s, strength 0.1), one
anchor is changed at a time:
  nM runs: low time-gap anchor (reference 2.0 s),
  hM runs: high time-gap anchor (reference 0.01 s) and high desired-speed
           anchor (reference 3.6 m/s).
The reference runs are taken from build/croma_anchor_fit.

  python scripts/croma_discharge_factorial.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import croma_anchor_fit as fit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POINT = (1.0, 1.2, 0.1)
FACTORS = {
    "nM": [("--time-gap-low", [1.5, 1.2, 1.0])],
    "hM": [("--time-gap-high", [0.1, 0.3, 0.5]), ("--v0-high", [2.4, 1.8])],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "build" / "croma_discharge")
    return parser.parse_args()


def replica(job: tuple) -> dict:
    """Run fit.replica with one extra anchor override."""
    mode, run, seed, flag, level, out_dir = job
    sub = out_dir / f"{flag.strip('-')}_{level}"
    sub.mkdir(parents=True, exist_ok=True)
    original = fit.point_args
    fit.point_args = lambda *p: [*original(*p), flag, str(level)]
    try:
        row = fit.replica((mode, run, POINT, seed, sub))
    finally:
        fit.point_args = original
    return {**row, "factor": flag.strip("-"), "level": level}


def main() -> None:
    args = parse_args()
    jobs = [(mode, run, seed, flag, level, args.output_dir)
            for cond, factors in FACTORS.items()
            for (flag, levels), mode in itertools.product(factors, ["V", "PVE"])
            for level, run, seed in itertools.product(levels, fit.RUNS[cond], fit.SEEDS)]
    with ProcessPoolExecutor(args.workers) as pool:
        table = pd.DataFrame(list(pool.map(replica, jobs)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "factorial.csv", index=False)
    print(f"failed: {int((table['status'] != 'ok').sum())} of {len(table)}")
    print(table.groupby(["mode", "factor", "level", "run"])[fit.MEASURES].mean().round(2).to_string())


if __name__ == "__main__":
    main()
