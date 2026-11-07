"""Rank-area correlation of calibrated static models on the CrowdQueue runs.

Runs the CrowdQueue replica of jupedsim-dakota-calibration
(crowdqueue/driver_cq.py, CollisionFreeSpeedModelV2, measured initial
positions and arrivals) with two static parameter sets calibrated there,
and applies the same rank-area measure as crowdqueue_rank_area.py:

  set_c      one parameter set for all runs (calib_h0_s9),
  condition  set C interaction parameters with desired speed and time gap
             fitted per motivation condition (dakota_h0_confirm,
             dakota_hminus_confirm).

Static means every agent has the same, constant parameters, so any
rank-area gradient comes from geometry and crowd dynamics alone.
Needs the dakota project's environment (jupedsim 1.4.2, pedpy 1.4.0,
h5py); the runs excluded there (uncorrected trajectories) are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pedpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crowdqueue_rank_area import rank_area

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAMES = ["desired_speed", "radius", "time_gap", "strength_neighbor", "range_neighbor",
         "strength_geometry", "range_geometry"]
CONDITION_FILES = {"h0": "dakota_h0_confirm.json", "h-": "dakota_hminus_confirm.json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dakota-dir", type=Path, required=True,
                        help="Checkout of jupedsim-dakota-calibration.")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--cutoff", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "build" / "crowdqueue_static")
    return parser.parse_args()


def load_params(results_dir: Path, name: str) -> dict:
    params = json.loads((results_dir / name).read_text())["params"]
    return {k: float(params[k]) for k in NAMES}


def parameter_sets(results_dir: Path) -> dict[str, dict]:
    """model -> {motivation -> parameters}."""
    set_c = load_params(results_dir, "calib_h0_s9.json")
    return {
        "set_c": {"h0": set_c, "h-": set_c, "h+": set_c},
        "condition": {m: load_params(results_dir, f) for m, f in CONDITION_FILES.items()},
    }


def simulate(job: tuple) -> dict:
    run_name, seed, out_file, params = job
    import driver_cq  # imported in the worker; needs the dakota crowdqueue dir on sys.path

    return driver_cq.one(run_name, seed, out_file, params)


def main() -> None:
    args = parse_args()
    cq_dir = args.dakota_dir / "crowdqueue"
    sys.path.insert(0, str(cq_dir))
    import observables_cq as obs

    runs = obs.runs()
    sets = parameter_sets(cq_dir / "results")
    jobs, labels = [], []
    for model, by_condition in sets.items():
        for run_name, run in runs.items():
            params = by_condition.get(run["motivation"])
            if params is None:
                continue
            for seed in range(1, args.seeds + 1):
                out = args.output_dir / "runs" / f"{model}_{run_name}_s{seed}.sqlite"
                out.parent.mkdir(parents=True, exist_ok=True)
                jobs.append((run_name, seed, str(out), params))
                labels.append((model, run))
    with ProcessPoolExecutor(args.workers, initializer=sys.path.insert, initargs=(0, str(cq_dir))) as ex:
        records = list(ex.map(simulate, jobs))

    rows = []
    for (model, run), (run_name, seed, out_file, params), rec in zip(labels, jobs, records):
        row = {"model": model, "run": run_name, "seed": seed, "width": run["width"],
               "motivation": run["motivation"], "status": rec["status"],
               "time_gap": params["time_gap"], "desired_speed": params["desired_speed"]}
        if rec["status"] != "failed":
            traj = pedpy.load_trajectory_from_jupedsim_sqlite(Path(out_file))
            area = pedpy.load_walkable_area_from_jupedsim_sqlite(Path(out_file)).polygon
            summary, _ = rank_area(traj, area, {"run": run_name}, args.cutoff)
            row.update({k: summary[k] for k in ("n_crossed", "rho_area_all", "rho_area_before")})
        rows.append(row)
        print(f"{model} {run_name} s{seed}: {rec['status']} rho_before={row.get('rho_area_before', float('nan')):+.3f}")
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "summary.csv", index=False)
    print(table.groupby(["model", "motivation"])["rho_area_before"].describe().round(3).to_string())


if __name__ == "__main__":
    main()
