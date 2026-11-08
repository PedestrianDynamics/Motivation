"""Fit standing distance, desired speed and neighbour strength on CROMA, per arm.

Design fixed in revision/croma_replica_plan.md (fourth amendment).
V and PVE: the low mapping anchors are fitted on the nM blank runs, then
the selected point is run on hM as a prediction. BASE_MODEL: agent
buffer, normal_v_0 and a_ped are fitted per condition. Score per
condition: mean over runs and seeds of |ln(depth ratio)| + |front
difference| + |ln(discharge ratio)| against croma_waiting.csv.

Output in --output-dir: grid_<mode>.csv (every replica), selected_<mode>.csv.
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

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from croma_waiting import DISCHARGE_WINDOW_S, discharge, waiting_measures
from experimental_rank_area import load_walkable_area

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS = {"nM": ["1C060", "2C020"], "hM": ["2C070", "2C120"]}
SEEDS = [101, 102, 103]
SPEEDS = [0.6, 0.8, 1.0, 1.2]
STRENGTHS = [0.1, 0.3, 0.5]
BUFFERS = {"nM": [1.0, 1.3, 1.6, 2.0], "hM": [0.1, 0.2, 0.3, 0.5]}
MEASURES = ["depth_med", "frac_front3m", "rho_entry_rank", "rho_rank_area", "discharge_60s"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", required=True, choices=["BASE_MODEL", "V", "PVE"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--experiment", type=Path, default=PROJECT_ROOT / "build" / "croma_waiting.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "build" / "croma_anchor_fit")
    return parser.parse_args()


def point_args(mode: str, buffer: float, speed: float, strength: float) -> list[str]:
    """CLI overrides of croma_replica.py for one grid point."""
    if mode == "BASE_MODEL":
        return ["--agent-buffer", str(buffer), "--desired-speed", str(speed), "--a-ped", str(strength)]
    return ["--agent-buffer", "0", "--buffer-low", str(buffer), "--v0-low", str(speed),
            "--strength-low", str(strength)]


def replica(job: tuple) -> dict:
    mode, run, point, seed, out_dir = job
    buffer, speed, strength = point
    out = out_dir / mode / f"{run}_b{buffer}_v{speed}_a{strength}_s{seed}.sqlite"
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "croma_replica.py"), "--run", run,
           "--mode", mode, "--time-gap", "1.0", *point_args(mode, buffer, speed, strength),
           "--seed", str(seed), "--after-open", str(DISCHARGE_WINDOW_S), "--output", str(out)]
    row = {"mode": mode, "run": run, "buffer": buffer, "speed": speed, "strength": strength, "seed": seed}
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        error = (proc.stderr.strip().splitlines() or ["?"])[-1]
        return {**row, "status": "failed", "error": error[:160]}
    record = json.loads(out.with_suffix(".json").read_text())
    with sqlite3.connect(f"file:{out}?mode=ro", uri=True) as con:
        df = pd.read_sql("SELECT frame, id, pos_x AS x, pos_y AS y FROM trajectory_data", con)
        fps = float(dict(con.execute("SELECT key, value FROM metadata").fetchall())["fps"])
    open_frame = round(record["door_open_s"] * fps)
    row.update(waiting_measures(df, fps, open_frame, load_walkable_area(None)))
    row["discharge_60s"] = discharge(df, fps, open_frame)
    return {**row, "status": "ok"}


def score(table: pd.DataFrame, exp: pd.DataFrame) -> pd.Series:
    """Mean score per (condition, buffer, speed, strength)."""
    ok = table[table["status"] == "ok"].merge(exp, on="run", suffixes=("", "_exp"))
    ok["score"] = (np.abs(np.log(ok["depth_med"] / ok["depth_med_exp"]))
                   + np.abs(ok["frac_front3m"] - ok["frac_front3m_exp"])
                   + np.abs(np.log(ok["discharge_60s"].clip(lower=1) / ok["discharge_60s_exp"])))
    return ok.groupby(["condition", "buffer", "speed", "strength"])["score"].mean()


def run_jobs(jobs: list[tuple], workers: int) -> pd.DataFrame:
    with ProcessPoolExecutor(workers) as pool:
        table = pd.DataFrame(list(pool.map(replica, jobs)))
    table["condition"] = table["run"].map({r: c for c, runs in RUNS.items() for r in runs})
    return table


def grid_jobs(mode: str, condition: str, out_dir: Path) -> list[tuple]:
    points = itertools.product(BUFFERS[condition], SPEEDS, STRENGTHS)
    return [(mode, run, p, seed, out_dir)
            for p, run, seed in itertools.product(points, RUNS[condition], SEEDS)]


def main() -> None:
    args = parse_args()
    exp = pd.read_csv(args.experiment)[["run", *MEASURES]]
    conditions = ["nM", "hM"] if args.mode == "BASE_MODEL" else ["nM"]
    table = run_jobs([j for c in conditions for j in grid_jobs(args.mode, c, args.output_dir)],
                     args.workers)
    scores = score(table, exp)
    selected = {c: scores.loc[c].idxmin() for c in conditions}
    if args.mode != "BASE_MODEL":  # hM is predicted from the nM point
        selected["hM"] = selected["nM"]
        hm_jobs = [(args.mode, run, selected["hM"], seed, args.output_dir)
                   for run, seed in itertools.product(RUNS["hM"], SEEDS)]
        table = pd.concat([table, run_jobs(hm_jobs, args.workers)], ignore_index=True)
    table.to_csv(args.output_dir / f"grid_{args.mode}.csv", index=False)
    print(f"failed: {int((table['status'] != 'ok').sum())} of {len(table)}")
    for c in conditions:
        print(f"\n{args.mode} {c}: best 5 (buffer, speed, strength) -> score")
        print(scores.loc[c].sort_values().head(5).round(3).to_string())
    rows = []
    for c, point in selected.items():
        sel = table[(table["condition"] == c) & (table["status"] == "ok")
                    & (table[["buffer", "speed", "strength"]].apply(tuple, axis=1) == tuple(point))]
        for run, g in sel.groupby("run"):
            rows.append({"mode": args.mode, "condition": c, "buffer": point[0], "speed": point[1],
                         "strength": point[2], "run": run, "n": len(g),
                         **g[MEASURES].mean().round(2).to_dict()})
    out = pd.DataFrame(rows)
    out.to_csv(args.output_dir / f"selected_{args.mode}.csv", index=False)
    print("\n" + out.to_string(index=False))


if __name__ == "__main__":
    main()
