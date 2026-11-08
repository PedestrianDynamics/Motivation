"""Mapping of value to CFSM parameters and value-rank correlation in the CROMA fit runs.

Reads the selected V (low anchor 1.3) and PVE (1.6) replicas in build/croma_fit,
rebuilds each agent value from its seed as EVPStrategy does, and correlates it with
the rank at door opening and with the advance over entry order. Run from the repo root.
"""
import sys, json, random, sqlite3
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from src.motivation_mapping import MotivationParameterMapper
from croma_waiting import DOOR

ALPHA = 14/3
def mapper(run, low):
    mp = json.loads(Path(f"files/base_testing_{run}.json").read_text())["motivation_parameters"]
    mp["buffer_anchors"]["low"] = low
    return MotivationParameterMapper(mapping_block=mp, normal_v_0=1.2, range_default=0.2), mp
print("mapping for nM values 1..3 and hM 4..7 (low anchor 1.3):")
m_, mp = mapper("1C060", 1.3)
for v in [1, 2, 3, 4, 5.5, 7]:
    m = min(v/ALPHA, 3); print(f" V={v}: m={m:.2f} buffer={m_.buffer(m):.2f} v0={m_.desired_speed(m):.2f} T={m_.time_gap(m, 1.0):.2f}")

rows = []
for mode, low in [("V", 1.3), ("PVE", 1.6)]:
    for run in ["1C060", "2C020", "2C070", "2C120"]:
        mp = json.loads(Path(f"files/base_testing_{run}.json").read_text())["motivation_parameters"]
        for seed in (101, 102, 103):
            f = Path(f"build/croma_fit/{mode}/{run}_T1.0_v1.2_b0.0_bl{low}_s{seed}.sqlite")
            rec = json.loads(f.with_suffix(".json").read_text())
            con = sqlite3.connect(f)
            df = pd.read_sql("select frame,id,pos_x x,pos_y y from trajectory_data", con)
            fps = float(dict(con.execute("select key,value from metadata").fetchall())["fps"])
            of = round(rec["door_open_s"] * fps)
            w = df[df.frame <= of]
            entry = w.groupby("id").frame.min().rank(method="first")
            at = w[w.frame == of].set_index("id")
            rank = np.hypot(at.x - DOOR[0], at.y - DOOR[1]).rank(method="first")
            ids = rank.index
            val = pd.Series({i: (random.seed(seed*1000+i), random.uniform(mp["min_value"], mp["max_value"]))[1] for i in ids})
            adv = entry[ids].rank() - rank  # >0: moved forward relative to entry order
            rows.append(dict(mode=mode, run=run, seed=seed,
                             rho_entry_rank=spearmanr(entry[ids], rank).statistic,
                             rho_value_rank=spearmanr(val, rank).statistic,
                             rho_value_advance=spearmanr(val, adv).statistic))
t = pd.DataFrame(rows)
print(t.groupby(["mode","run"])[["rho_entry_rank","rho_value_rank","rho_value_advance"]].mean().round(2).to_string())
