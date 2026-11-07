"""Diagnostics for two referee concerns on the rank-area result.

Check 1 (boundary artefact): is the rank-area correlation still present
when boundary effects are removed? Per agent we compute
  * area_all      mean Voronoi area over all frames (the paper's measure),
  * area_interior mean Voronoi area over frames where the agent is not a
                  vertex of the crowd's convex hull,
  * nn_dist       mean nearest-neighbour distance (no cells involved),
each over the full run and over the last ``--window`` seconds.
The Spearman correlation with final rank is reported per seed and model.

Check 2 (circularity): does the static value V predict final rank once
initial rank is controlled for? We report the partial Spearman
correlation rho(V, final_rank | initial_rank). BASE_MODEL draws V but
does not use it, so it is the placebo.

Inputs are the stored multi-seed runs under
files/coordination_scenarios/agents_<N>_open_100/base_runs.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pedpy
from pedpy import Cutoff, compute_individual_voronoi_polygons
from scipy.spatial import ConvexHull, cKDTree
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = ["BASE_MODEL", "P", "V", "SE", "PVE"]
RUN_RE = re.compile(r"^base_(?P<model>.+)_seed(?P<seed>\d+)_(?P=model)_[\d_-]+\.sqlite$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agents", type=int, default=80)
    parser.add_argument("--window", type=float, default=30.0, help="Seconds at end of run.")
    parser.add_argument("--cutoff", type=float, default=1.0, help="Voronoi cutoff radius (paper: 1 m).")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "files" / "r3_diagnostics"
    )
    return parser.parse_args()


def discover_runs(runs_dir: Path) -> list[tuple[str, int, Path]]:
    runs = []
    for path in sorted(runs_dir.glob("base_*_seed*.sqlite")):
        match = RUN_RE.match(path.name)
        if match is None:
            continue
        runs.append((match["model"], int(match["seed"]), path))
    return runs


def door_center(config_path: Path) -> np.ndarray:
    import json

    data = json.loads(config_path.read_text())
    vertices = np.asarray(data["motivation_parameters"]["motivation_doors"][0]["vertices"])
    return vertices.mean(axis=0)


def hull_flags(df: pd.DataFrame) -> pd.Series:
    """True where the agent is a vertex of its frame's convex hull."""
    flags = pd.Series(False, index=df.index)
    for _, frame in df.groupby("frame"):
        if len(frame) < 4:
            flags.loc[frame.index] = True
            continue
        hull = ConvexHull(frame[["x", "y"]].to_numpy())
        flags.loc[frame.index[hull.vertices]] = True
    return flags


def nn_distance(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    for _, frame in df.groupby("frame"):
        if len(frame) < 2:
            continue
        dist, _ = cKDTree(frame[["x", "y"]].to_numpy()).query(frame[["x", "y"]].to_numpy(), k=2)
        out.loc[frame.index] = dist[:, 1]
    return out


def per_frame_measures(sqlite_path: Path, cutoff: float) -> tuple[pd.DataFrame, float]:
    traj = pedpy.load_trajectory_from_jupedsim_sqlite(sqlite_path)
    walkable = pedpy.load_walkable_area_from_jupedsim_sqlite(sqlite_path)
    cells = compute_individual_voronoi_polygons(
        traj_data=traj, walkable_area=walkable, cut_off=Cutoff(radius=cutoff, quad_segments=3)
    )
    cells["area"] = cells["polygon"].apply(lambda poly: poly.area)
    df = traj.data[["id", "frame", "x", "y"]].merge(
        cells[["id", "frame", "area"]], on=["id", "frame"], how="inner"
    )
    df["on_hull"] = hull_flags(df)
    df["nn_dist"] = nn_distance(df)
    return df, float(traj.frame_rate)


def final_and_initial_rank(df: pd.DataFrame, door: np.ndarray) -> pd.DataFrame:
    def rank_at(frame_df: pd.DataFrame) -> pd.Series:
        d = np.hypot(frame_df["x"] - door[0], frame_df["y"] - door[1])
        return pd.Series(d.to_numpy(), index=frame_df["id"]).rank(method="first")

    first = df[df["frame"] == df["frame"].min()]
    last = df[df["frame"] == df["frame"].max()]
    return pd.DataFrame({"initial_rank": rank_at(first), "final_rank": rank_at(last)})


def agent_measures(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("id")
    interior = df[~df["on_hull"]].groupby("id")["area"].mean()
    return pd.DataFrame(
        {
            "area_all": grouped["area"].mean(),
            "area_interior": interior,
            "nn_dist": grouped["nn_dist"].mean(),
            "hull_fraction": grouped["on_hull"].mean(),
        }
    )


def partial_spearman(x: pd.Series, y: pd.Series, z: pd.Series) -> float:
    """Spearman partial correlation rho(x, y | z) via residuals of ranks."""
    rx, ry, rz = (s.rank().to_numpy() for s in (x, y, z))
    design = np.column_stack([np.ones_like(rz), rz])
    res_x = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    res_y = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    return float(np.corrcoef(res_x, res_y)[0, 1])


def static_values(sqlite_path: Path) -> pd.Series:
    csv_path = sqlite_path.with_name(sqlite_path.stem + "_motivation.csv")
    mot = pd.read_csv(csv_path, skipinitialspace=True)
    first = mot[mot["frame"] == mot["frame"].min()]
    return first.set_index("id")["value"]


def analyse_run(model: str, seed: int, sqlite_path: Path, args: argparse.Namespace) -> list[dict]:
    df, fps = per_frame_measures(sqlite_path, args.cutoff)
    ranks = final_and_initial_rank(df, door_center(sqlite_path.with_suffix(".json")))
    window_start = df["frame"].max() - args.window * fps
    rows = []
    for window, sub in (("all", df), (f"last{int(args.window)}s", df[df["frame"] >= window_start])):
        agents = agent_measures(sub).join(ranks, how="inner")
        for measure in ("area_all", "area_interior", "nn_dist", "hull_fraction"):
            valid = agents[[measure, "final_rank"]].dropna()
            rho = spearmanr(valid["final_rank"], valid[measure]).statistic
            rows.append(row(model, seed, "rank_vs_" + measure, window, rho, len(valid)))

    values = static_values(sqlite_path).reindex(ranks.index)
    rho_v = partial_spearman(values, ranks["final_rank"], ranks["initial_rank"])
    rows.append(row(model, seed, "V_vs_final_rank|initial_rank", "all", rho_v, len(ranks)))
    rho_raw = spearmanr(values, ranks["final_rank"]).statistic
    rows.append(row(model, seed, "V_vs_final_rank", "all", rho_raw, len(ranks)))
    return rows


def row(model: str, seed: int, metric: str, window: str, rho: float, n: int) -> dict:
    return {"model": model, "seed": seed, "metric": metric, "window": window, "rho": rho, "n": n}


def summarise(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return (
        df.groupby(["metric", "window", "model"])["rho"]
        .agg(median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75), seeds="count")
        .reset_index()
    )


def main() -> None:
    args = parse_args()
    runs_dir = PROJECT_ROOT / "files" / "coordination_scenarios" / f"agents_{args.agents}_open_100" / "base_runs"
    runs = [r for r in discover_runs(runs_dir) if r[0] in MODELS]
    if not runs:
        raise SystemExit(f"No runs found in {runs_dir}")

    rows: list[dict] = []
    for model, seed, path in runs:
        print(f"{model} seed {seed}")
        rows.extend(analyse_run(model, seed, path, args))

    out_dir = args.output_dir / f"agents_{args.agents}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarise(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summary.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
