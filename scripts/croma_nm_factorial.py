"""Which low-motivation mapping value makes the CROMA nM replica reorder?

Static V3 replicas of the nM blank runs at agent buffer 1.3 m (the V
arm's fitted low anchor), moving one parameter at a time from the static
values (v0 1.2 m/s, T 1.0 s, a_ped 0.5) to the values the mapping gives
every nM agent (v0 0.5, T 1.9, a_ped 0.1). Usage: croma_nm_factorial.py OUT_DIR
"""
import sys, json, sqlite3, subprocess, itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from croma_waiting import DISCHARGE_WINDOW_S, discharge, waiting_measures
from experimental_rank_area import load_walkable_area
S = Path(sys.argv[1])
ARMS = {"static_b1.3": (1.2, 1.0, 0.5), "v0=0.5": (0.5, 1.0, 0.5), "T=1.9": (1.2, 1.9, 0.5),
        "a_ped=0.1": (1.2, 1.0, 0.1), "all(V-nM)": (0.5, 1.9, 0.1)}
def job(a):
    arm, run, seed = a
    v0, T, ap = ARMS[arm]
    out = S / "fac" / f"{arm}_{run}_s{seed}.sqlite"
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "croma_replica.py"), "--run", run, "--mode", "BASE_MODEL",
           "--time-gap", str(T), "--desired-speed", str(v0), "--agent-buffer", "1.3",
           "--seed", str(seed), "--after-open", str(DISCHARGE_WINDOW_S), "--output", str(out)]
    if ap != 0.5:
        cmd += ["--config", str(S / "fac" / f"cfg_{run}_a0.1.json")]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if p.returncode:
        return {"arm": arm, "run": run, "seed": seed, "status": "failed"}
    rec = json.loads(out.with_suffix(".json").read_text())
    con = sqlite3.connect(out)
    df = pd.read_sql("select frame,id,pos_x x,pos_y y from trajectory_data", con)
    fps = float(dict(con.execute("select key,value from metadata").fetchall())["fps"])
    of = round(rec["door_open_s"] * fps)
    r = {"arm": arm, "run": run, "seed": seed, "status": "ok"}
    r.update(waiting_measures(df, fps, of, load_walkable_area(None))); r["discharge_60s"] = discharge(df, fps, of)
    return r
if __name__ == '__main__':
    (S / "fac").mkdir(parents=True, exist_ok=True)
    for run in ("1C060", "2C020"):
        cfg = json.loads((PROJECT_ROOT / "files" / f"base_testing_{run}.json").read_text())
        cfg["velocity_init_parameters"]["a_ped"] = 0.1
        (S / "fac" / f"cfg_{run}_a0.1.json").write_text(json.dumps(cfg))
    jobs = list(itertools.product(ARMS, ["1C060", "2C020"], [101, 102, 103]))
    with ProcessPoolExecutor(8) as ex:
        t = pd.DataFrame(list(ex.map(job, jobs)))
    t.to_csv(S / "fac" / "factorial.csv", index=False)
    print(t.groupby(["arm", "run"])[["depth_med", "frac_front3m", "rho_entry_rank", "rho_rank_area", "discharge_60s"]].mean().round(2).to_string())
    print(t.status.value_counts().to_dict())
