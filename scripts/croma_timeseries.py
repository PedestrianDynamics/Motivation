"""Time series of the waiting crowd in the CROMA runs and replicas, up to door opening.

Every STEP_S seconds, for the persons inside the walkable area in front of
the door line (y < 20 m), measure:
  n               persons present,
  depth_med/p90   median and 90th percentile of y_front - y [m],
  frac_front3m    fraction within 3 m of the front-most person,
  sd_x            lateral standard deviation [m],
  nn_med          median nearest-neighbour distance [m],
  frac_moving     fraction with speed > MOVING_SPEED over the last STEP_S.
Per person (queue then advance): the first time the person has been still
(speed < STILL_SPEED) for STILL_S seconds, and the forward distance
y(open) - y(stop) covered after that. Per run: frac_adv1m and frac_adv3m,
the fractions of stopped persons that advanced more than 1 m and 3 m.

  python scripts/croma_timeseries.py --output-dir build/croma_timeseries
measures the experiment; --replica-dir build/croma_anchor_fit adds the
selected replica points of every mode (selected_<mode>.csv).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from croma_ordering import condition
from croma_waiting import CROMA_FPS, DOOR, FRONT_BAND, load_experiment
from experimental_rank_area import load_walkable_area

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP_S = 1.0
MOVING_SPEED = 0.2
STILL_SPEED = 0.1
STILL_S = 5
SEEDS = (101, 102, 103)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--traj-dir", type=Path, default=PROJECT_ROOT.parent / "trajectories_croma"
    )
    parser.add_argument("--replica-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def nearest_neighbour(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return float("nan")
    dist, _ = cKDTree(xy).query(xy, k=2)
    return float(np.median(dist[:, 1]))


def snapshot(now: pd.DataFrame, before: pd.DataFrame) -> dict:
    depth = now["y"].max() - now["y"]
    shift = now.join(before, rsuffix="_prev", how="inner")
    speed = (
        np.hypot(shift["x"] - shift["x_prev"], shift["y"] - shift["y_prev"]) / STEP_S
    )
    return {
        "n": len(now),
        "depth_med": float(depth.median()),
        "depth_p90": float(depth.quantile(0.9)),
        "frac_front3m": float((depth <= FRONT_BAND).mean()),
        "sd_x": float(now["x"].std(ddof=0)),
        "nn_med": nearest_neighbour(now[["x", "y"]].to_numpy()),
        "frac_moving": float((speed > MOVING_SPEED).mean())
        if len(speed)
        else float("nan"),
    }


def waiting_samples(
    df: pd.DataFrame, fps: float, open_frame: int, walkable
) -> pd.DataFrame:
    """Persons in front of the door, sampled every STEP_S on a grid ending at open_frame."""
    step = round(STEP_S * fps)
    inside = shapely.contains_xy(
        walkable.polygon, df["x"].to_numpy(), df["y"].to_numpy()
    )
    keep = inside & (df["y"] < DOOR[1]) & (df["frame"] <= open_frame)
    keep &= (open_frame - df["frame"]) % step == 0
    return df[keep]


def series(samples: pd.DataFrame, fps: float, open_frame: int) -> list[dict]:
    step = round(STEP_S * fps)
    by_frame = {f: g.set_index("id")[["x", "y"]] for f, g in samples.groupby("frame")}
    rows = []
    for frame in sorted(by_frame):
        before = by_frame.get(frame - step)
        if before is None:
            continue
        rows.append(
            {
                "t_to_open_s": (frame - open_frame) / fps,
                **snapshot(by_frame[frame], before),
            }
        )
    return rows


def advance_after_stop(g: pd.DataFrame) -> float:
    """Forward distance covered after the first STILL_S-second stop; NaN if never still."""
    x, y = g["x"].to_numpy(), g["y"].to_numpy()
    still = np.hypot(np.diff(x), np.diff(y)) / STEP_S < STILL_SPEED
    if len(still) < STILL_S:
        return float("nan")
    window = np.convolve(still.astype(int), np.ones(STILL_S, int), "valid") == STILL_S
    if not window.any():
        return float("nan")
    return float(y[-1] - y[int(np.argmax(window))])


def advance_summary(samples: pd.DataFrame) -> dict:
    adv = (
        samples.sort_values("frame")
        .groupby("id")[["x", "y"]]
        .apply(advance_after_stop)
        .dropna()
    )
    return {
        "n_stopped": len(adv),
        "frac_adv1m": float((adv > 1).mean()),
        "frac_adv3m": float((adv > 3).mean()),
        "adv_med": float(adv.median()),
    }


def measure(df: pd.DataFrame, fps: float, open_frame: int, walkable, label: dict):
    samples = waiting_samples(df, fps, open_frame, walkable)
    rows = [{**label, **r} for r in series(samples, fps, open_frame)]
    return rows, {**label, **advance_summary(samples)}


def experiment(traj_dir: Path, walkable):
    for path in sorted(traj_dir.glob("*_Combined.txt")):
        df, open_frame = load_experiment(path)
        label = {
            "source": "experiment",
            "run": path.name[:5],
            "condition": condition(path),
            "seed": 0,
        }
        yield measure(df, CROMA_FPS, open_frame, walkable, label)


def load_replica(sqlite_path: Path) -> tuple[pd.DataFrame, float, int]:
    record = json.loads(sqlite_path.with_suffix(".json").read_text())
    with sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True) as con:
        df = pd.read_sql(
            "SELECT frame, id, pos_x AS x, pos_y AS y FROM trajectory_data", con
        )
        fps = float(
            dict(con.execute("SELECT key, value FROM metadata").fetchall())["fps"]
        )
    return df, fps, round(record["door_open_s"] * fps)


def replicas(replica_dir: Path, walkable):
    for selected in sorted(replica_dir.glob("selected_*.csv")):
        for p in pd.read_csv(selected).itertuples():
            for seed in SEEDS:
                path = (
                    replica_dir
                    / p.mode
                    / f"{p.run}_b{p.buffer}_v{p.speed}_a{p.strength}_s{seed}.sqlite"
                )
                df, fps, open_frame = load_replica(path)
                label = {
                    "source": p.mode,
                    "run": p.run,
                    "condition": p.condition,
                    "seed": seed,
                }
                yield measure(df, fps, open_frame, walkable, label)


def main() -> None:
    args = parse_args()
    walkable = load_walkable_area(None)
    results = list(experiment(args.traj_dir, walkable))
    if args.replica_dir is not None:
        results += list(replicas(args.replica_dir, walkable))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r for rows, _ in results for r in rows]).to_csv(
        args.output_dir / "timeseries.csv", index=False
    )
    summary = pd.DataFrame([s for _, s in results])
    summary.to_csv(args.output_dir / "advance.csv", index=False)
    print(
        summary.groupby(["source", "run", "condition"])[
            ["n_stopped", "frac_adv1m", "frac_adv3m", "adv_med"]
        ]
        .mean()
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
