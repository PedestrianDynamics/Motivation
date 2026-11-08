"""Select the standing distance per arm and report held-out CROMA measures.

Rule fixed in revision/croma_replica_plan.md (third amendment): score per
condition = mean over its blank runs and seeds of
|ln(depth_sim / depth_exp)| + |front_sim - front_exp|, with the experimental
values of the same run from croma_waiting.csv. The static arm (BASE_MODEL)
selects its agent buffer per condition; V and PVE select the low buffer
anchor on nM only and keep it for hM.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEASURES = ["depth_med", "frac_front3m", "rho_entry_rank", "rho_rank_area", "discharge_60s"]
FIT_PARAM = {"BASE_MODEL": "agent_buffer", "V": "buffer_low", "PVE": "buffer_low"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid-dir", type=Path, default=PROJECT_ROOT / "build" / "croma_fit")
    parser.add_argument("--experiment", type=Path, default=PROJECT_ROOT / "build" / "croma_waiting.csv")
    return parser.parse_args()


def score(grid: pd.DataFrame, exp: pd.DataFrame) -> pd.DataFrame:
    ok = grid[grid["status"] == "ok"].merge(exp, on="run", suffixes=("", "_exp"))
    ok["score"] = (np.abs(np.log(ok["depth_med"] / ok["depth_med_exp"]))
                   + np.abs(ok["frac_front3m"] - ok["frac_front3m_exp"]))
    return ok


def select(ok: pd.DataFrame, mode: str) -> dict[str, float]:
    """condition -> selected value of the fitted parameter."""
    param = FIT_PARAM[mode]
    by = ok.groupby(["condition", param])["score"].mean()
    if mode == "BASE_MODEL":
        return {c: float(by.loc[c].idxmin()) for c in ("nM", "hM")}
    best = float(by.loc["nM"].idxmin())
    return {"nM": best, "hM": best}


def main() -> None:
    args = parse_args()
    exp_all = pd.read_csv(args.experiment)
    exp = exp_all[["run", *MEASURES]]
    rows = []
    for mode, param in FIT_PARAM.items():
        grid = pd.read_csv(args.grid_dir / f"grid_{mode}.csv")
        ok = score(grid, exp)
        print(f"\n{mode}: score by condition and {param}")
        print(ok.groupby(["condition", param])["score"].mean().unstack(0).round(3).to_string())
        failed = grid.groupby(param)["status"].apply(lambda s: int((s != "ok").sum()))
        print(f"failed runs by {param}: {failed.to_dict()}")
        for condition, value in select(ok, mode).items():
            sel = ok[(ok["condition"] == condition) & (ok[param] == value)]
            for run, g in sel.groupby("run"):
                rows.append({"mode": mode, "condition": condition, param: value, "run": run,
                             "n": len(g), **g[MEASURES].mean().round(2).to_dict()})
    table = pd.DataFrame(rows)
    exp_rows = exp_all[exp_all["run"].isin(table["run"])][["run", "condition", *MEASURES]]
    exp_rows = exp_rows.assign(mode="experiment")
    table = pd.concat([table, exp_rows], ignore_index=True)
    out = args.grid_dir / "selection.csv"
    table.to_csv(out, index=False)
    cols = ["mode", "condition", "agent_buffer", "buffer_low", "run", *MEASURES]
    print("\n" + table.reindex(columns=cols).sort_values(["run", "mode"]).to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
