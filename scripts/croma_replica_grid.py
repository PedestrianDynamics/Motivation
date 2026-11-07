"""Run CROMA replicas over a parameter grid and measure them at door opening.

For every (run, mode, time gap, seed) the replica (croma_replica.py) is run
until 60 s after the door opens; croma_waiting.waiting_measures and
croma_waiting.discharge are applied to the trajectory, with the
experiment's walkable area. Output:
one CSV row per replica, plus the experimental values from
croma_waiting.py for comparison.

The static arm is selected per condition from crowd shape only (depth and
front fraction at opening, see revision/croma_replica_plan.md); the
correlations are then held-out predictions.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from croma_waiting import DISCHARGE_WINDOW_S, discharge, waiting_measures
from experimental_rank_area import load_walkable_area

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLANK_RUNS = {"1C060": "nM", "2C020": "nM", "2C070": "hM", "2C120": "hM"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", default="BASE_MODEL")
    parser.add_argument("--time-gaps", default="0.3,0.5,0.8,1.0,1.3,1.6,2.0")
    parser.add_argument("--desired-speeds", default="1.2")
    parser.add_argument("--agent-buffers", default="0")
    parser.add_argument("--seeds", default="101,102,103")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tag", default="", help="Suffix for the output CSV.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "build" / "croma_replica")
    return parser.parse_args()


def replica(job: tuple) -> dict:
    run, mode, time_gap, speed, buffer, seed, out_dir = job
    out = out_dir / mode / f"{run}_T{time_gap}_v{speed}_b{buffer}_s{seed}.sqlite"
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "croma_replica.py"), "--run", run,
           "--mode", mode, "--time-gap", str(time_gap), "--desired-speed", str(speed),
           "--agent-buffer", str(buffer), "--seed", str(seed),
           "--after-open", str(DISCHARGE_WINDOW_S), "--output", str(out)]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if proc.returncode != 0:  # e.g. an agent pushed through a wall; counted, not measured
        error = (proc.stderr.strip().splitlines() or ["?"])[-1]
        return {"run": run, "mode": mode, "seed": seed, "time_gap": time_gap,
                "desired_speed": speed, "agent_buffer": buffer, "status": "failed", "error": error[:160]}
    record = json.loads(out.with_suffix(".json").read_text())
    record["status"] = "ok"
    with sqlite3.connect(f"file:{out}?mode=ro", uri=True) as con:
        df = pd.read_sql("SELECT frame, id, pos_x AS x, pos_y AS y FROM trajectory_data", con)
        fps = float(dict(con.execute("SELECT key, value FROM metadata").fetchall())["fps"])
    open_frame = round(record["door_open_s"] * fps)
    record.update(waiting_measures(df, fps, open_frame, load_walkable_area(None)))
    record["discharge_60s"] = discharge(df, fps, open_frame)
    return record


def main() -> None:
    args = parse_args()
    gaps = [float(g) for g in args.time_gaps.split(",")]
    speeds = [float(v) for v in args.desired_speeds.split(",")]
    buffers = [float(b) for b in args.agent_buffers.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    jobs = [(run, args.mode, gap, speed, buffer, seed, args.output_dir)
            for run, gap, speed, buffer, seed in itertools.product(BLANK_RUNS, gaps, speeds, buffers, seeds)]
    with ProcessPoolExecutor(args.workers) as pool:
        records = list(pool.map(replica, jobs))
    table = pd.DataFrame(records)
    table["condition"] = table["run"].map(BLANK_RUNS)
    out = args.output_dir / f"grid_{args.mode}_{args.tag}.csv" if args.tag else args.output_dir / f"grid_{args.mode}.csv"
    table.to_csv(out, index=False)
    cols = ["n_at_open", "depth_med", "frac_front3m", "rho_entry_rank", "rho_rank_area",
            "discharge_60s"]
    ok = table[table["status"] == "ok"]
    keys = ["condition", "time_gap", "desired_speed", "agent_buffer"]
    print(ok.groupby(keys)[cols].mean().round(2).to_string())
    print(table.groupby(keys)["status"].apply(lambda s: (s == "failed").sum()).rename("failed").to_string())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
