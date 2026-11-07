"""Replay a CROMA entrance run in the EVP simulation.

Design fixed in revision/croma_replica_plan.md. Every participant enters
at the position and time of their first observation in the experiment;
the door stays closed until the measured opening time (first person at
y >= 20 m), then opens. The run ends when everyone has passed the door
or after --after-open seconds.

The simulation itself is simulation.py's: the same geometry (closed and
open), CollisionFreeSpeedModelV3, agent parameters, motivation model and
per-agent parameter update. Only the population (entries replayed
instead of placed at t = 0) and the door timing differ.

  --mode BASE_MODEL  static: every agent keeps normal_v_0 / normal_time_gap
  --mode V           static individual values from [min_value, max_value]
  --mode PVE         full EVP
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import simulation as sim_mod

CROMA_FPS = 50.0
DOOR_Y = 20.0
FREE_SPACING = 0.02  # extra clearance when placing an entrant [m]
GIVE_UP_S = 10.0  # drop an entrant that finds no free spot for this long


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="CROMA run, e.g. 2C070.")
    parser.add_argument("--traj-dir", type=Path, default=PROJECT_ROOT.parent / "trajectories_croma")
    parser.add_argument("--config", type=Path, help="Default: files/base_testing_<run>.json.")
    parser.add_argument("--mode", default="BASE_MODEL", choices=["BASE_MODEL", "V", "PVE"])
    parser.add_argument("--time-gap", type=float, help="Override normal_time_gap.")
    parser.add_argument("--desired-speed", type=float, help="Override normal_v_0.")
    parser.add_argument("--agent-buffer", type=float, help="Override agent_buffer.")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--after-open", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True, help="Trajectory sqlite to write.")
    return parser.parse_args()


def experimental_entries(path: Path) -> tuple[list[tuple[float, float, float]], float]:
    """(t, x, y) of every participant's first observation, and the door-opening time [s]."""
    df = pd.read_csv(path, sep="\t", names=["id", "frame", "x", "y", "z", "m"], comment="#")
    t0 = df["frame"].min()
    first = df.loc[df.groupby("id")["frame"].idxmin()]
    entries = sorted(((f - t0) / CROMA_FPS, x, y) for f, x, y in zip(first["frame"], first["x"], first["y"]))
    t_open = (df.loc[df["y"] >= DOOR_Y, "frame"].min() - t0) / CROMA_FPS
    return entries, float(t_open)


def load_config(args: argparse.Namespace, n_agents: int) -> dict:
    config = args.config or PROJECT_ROOT / "files" / f"base_testing_{args.run}.json"
    data = json.loads(config.read_text())
    data["simulation_parameters"]["number_agents"] = n_agents
    data["motivation_parameters"]["motivation_mode"] = args.mode
    data["motivation_parameters"]["seed"] = float(args.seed)
    if args.time_gap is not None:
        data["motivation_parameters"]["normal_time_gap"] = args.time_gap
    if args.desired_speed is not None:
        data["motivation_parameters"]["normal_v_0"] = args.desired_speed
    if args.agent_buffer is not None:
        data["motivation_parameters"]["agent_buffer"] = args.agent_buffer
    return data


def free_spot(x: float, y: float, live: list[tuple[float, float]], region, min_dist: float) -> bool:
    if not region.contains(shapely.Point(x, y)):
        return False
    return all(np.hypot(x - px, y - py) > min_dist for px, py in live)


class Entrances:
    """Adds experimental entrants to the simulation once their time has come."""

    def __init__(self, simulation, template, entries, region, radius, rng):
        self.simulation = simulation
        self.template = template
        self.pending = list(entries)
        self.region = region
        self.min_dist = 2 * radius + FREE_SPACING
        self.rng = rng
        self.next_id = 1
        self.dropped = 0

    def add_due(self, now: float) -> None:
        live = [tuple(a.position) for a in self.simulation.agents()]
        still = []
        for t0, x, y in self.pending:
            if t0 > now:
                still.append((t0, x, y))
                continue
            placed = self.try_place(x, y, live)
            if placed is None and now - t0 <= GIVE_UP_S:
                still.append((t0, x, y))
            elif placed is None:
                self.dropped += 1
        self.pending = still

    def try_place(self, x: float, y: float, live: list) -> tuple[float, float] | None:
        for _ in range(20):  # nudge until the spot is free of walls and agents
            if free_spot(x, y, live, self.region, self.min_dist):
                self.template.position = (x, y)
                agent_id = self.simulation.add_agent(self.template)
                if agent_id != self.next_id:
                    raise RuntimeError(f"Expected agent id {self.next_id}, got {agent_id}")
                self.next_id += 1
                live.append((x, y))
                return x, y
            x, y = x + self.rng.normal(0, 0.1), y + self.rng.normal(0, 0.1)
        return None


def run(args: argparse.Namespace) -> dict:
    entries, t_open = experimental_entries(next(args.traj_dir.glob(f"{args.run}_*_Combined.txt")))
    data = load_config(args, len(entries))
    dt = sim_mod.parse_time_step(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)
    simulation, geometry_open = sim_mod.init_simulation(
        data, dt, sim_mod.parse_fps(data), args.output, from_file=True
    )
    templates, _ = sim_mod.create_agent_parameters(data, simulation)
    radius = data["velocity_init_parameters"]["radius"]
    # entrants appear near y = -1 m, far from the door, so the open geometry
    # is the right placement region
    region = geometry_open.buffer(-(radius + 0.03))
    ids = list(range(1, len(entries) + 1))
    model = sim_mod.init_motivation_model(data, ids, [(x, y) for _, x, y in entries])
    door = (0.5 * (model.door_point1[0] + model.door_point2[0]),
            0.5 * (model.door_point1[1] + model.door_point2[1]))
    entrances = Entrances(simulation, templates[0], entries, region, radius,
                          np.random.default_rng(args.seed))
    every_nth = data["simulation_parameters"]["fps"]
    open_iteration = round(t_open / dt)
    end_time = t_open + args.after_open
    door_open = False
    while simulation.elapsed_time() < end_time:
        if entrances.pending:
            entrances.add_due(simulation.elapsed_time())
        if not door_open and simulation.iteration_count() >= open_iteration:
            simulation.switch_geometry(geometry_open)
            door_open = True
        if door_open and not entrances.pending and simulation.agent_count() == 0:
            break
        positions = {a.id: tuple(a.position) for a in simulation.agents()}
        model.motivation_strategy.update_payoff_cache(
            iteration_count=simulation.iteration_count(),
            agent_positions=positions,
            number_agents_in_simulation=simulation.agent_count(),
        )
        if simulation.iteration_count() % every_nth == 0:
            for agent in simulation.agents():
                sim_mod.process_agent(agent, door, simulation, model, None, 0, 0, data)
        simulation.iterate()
    record = {
        "run": args.run, "mode": args.mode, "seed": args.seed,
        "time_gap": data["motivation_parameters"]["normal_time_gap"],
        "desired_speed": data["motivation_parameters"]["normal_v_0"],
        "agent_buffer": data["motivation_parameters"]["agent_buffer"],
        "entries": len(entries), "added": entrances.next_id - 1, "dropped": entrances.dropped,
        "door_open_s": t_open, "end_s": simulation.elapsed_time(),
        "remaining": simulation.agent_count(),
    }
    # the writer buffers 100 frames and writes them only on close();
    # init_simulation does not return it
    simulation._writer.close()
    return record


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    args = parse_args()
    record = run(args)
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
