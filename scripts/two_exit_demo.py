"""Two-exit demonstration: EVP motivation with distance and rank per goal.

A room with two exits in the top wall. Each agent's goal is the exit
nearest to its start position, fixed for the run; choosing the exit is the
tactical level and is not modelled here. Each exit gets its own EVP
strategy (simulation.init_motivation_model with that exit as the
motivation door and the agents heading to it), so spatial expectancy uses
the distance to the agent's own exit and the payoff uses the rank among the
agents heading to the same exit. Values come from the seed and the agent
id, as in the single-exit simulations. The model code in src/ is unchanged.

The motivation-to-parameter mapping is the one used in the CroMa replica
(files/base_testing_2C070.json, value range [4, 7]) with the low anchors
of the selected fit point (buffer 1.0 m, desired speed 1.2 m/s,
neighbour strength 0.1). This is a demonstration, not a validation.

  python scripts/two_exit_demo.py
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import jupedsim as jps
import shapely

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import simulation as sim_mod

ROOM = shapely.box(-8.0, 0.0, 8.0, 10.0)
EXIT_X = (-4.0, 4.0)
DOOR_HALF_WIDTH = 0.4
CORRIDOR_LENGTH = 2.0
START_AREA = shapely.box(-7.5, 0.5, 3.5, 6.0)
FIT_POINT = {"buffer": 1.0, "desired_speed": 1.2, "strength_neighbor_repulsion": 0.1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "files" / "base_testing_2C070.json")
    parser.add_argument("--agents", type=int, default=100)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--max-time", type=float, default=150.0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "build" / "two_exit_demo")
    return parser.parse_args()


def load_config(path: Path, seed: int) -> dict:
    data = json.loads(path.read_text())
    mp = data["motivation_parameters"]
    mp["motivation_mode"] = "PVE"
    mp["seed"] = float(seed)
    mp["buffer_anchors"]["low"] = FIT_POINT["buffer"]
    mp["desired_speed_anchors"]["low"] = FIT_POINT["desired_speed"]
    mp["strength_neighbor_repulsion_anchors"]["low"] = FIT_POINT["strength_neighbor_repulsion"]
    return data


def geometry() -> tuple[shapely.Polygon, list[shapely.Polygon]]:
    top = ROOM.bounds[3]
    corridors = [shapely.box(x - DOOR_HALF_WIDTH, top, x + DOOR_HALF_WIDTH, top + CORRIDOR_LENGTH)
                 for x in EXIT_X]
    exits = [shapely.box(x - DOOR_HALF_WIDTH, top + CORRIDOR_LENGTH - 0.5,
                         x + DOOR_HALF_WIDTH, top + CORRIDOR_LENGTH) for x in EXIT_X]
    return shapely.union_all([ROOM, *corridors]), exits


def agent_template(data: dict, journey_id: int, stage_id: int) -> jps.CollisionFreeSpeedModelV3AgentParameters:
    a_ped, d_ped, a_wall, d_wall = sim_mod.parse_velocity_init_parameters(data)
    return jps.CollisionFreeSpeedModelV3AgentParameters(
        journey_id=journey_id,
        stage_id=stage_id,
        radius=data["velocity_init_parameters"]["radius"],
        desired_speed=sim_mod.parse_normal_v_0(data),
        time_gap=sim_mod.parse_normal_time_gap(data),
        strength_neighbor_repulsion=a_ped,
        range_neighbor_repulsion=d_ped,
        strength_geometry_repulsion=a_wall,
        range_geometry_repulsion=d_wall,
        agent_buffer=data["motivation_parameters"]["agent_buffer"],
    )


def goal_model(data: dict, door_x: float, ids: list[int], positions: list[tuple[float, float]]):
    """EVP model whose motivation door is this exit and whose agents head to it."""
    goal_data = copy.deepcopy(data)
    top = ROOM.bounds[3]
    goal_data["motivation_parameters"]["motivation_doors"] = [
        {"id": 1, "label": "door", "vertices": [[door_x - DOOR_HALF_WIDTH, top], [door_x + DOOR_HALF_WIDTH, top]]}
    ]
    goal_data["simulation_parameters"]["number_agents"] = len(ids)
    return sim_mod.init_motivation_model(goal_data, ids, positions)


def run(args: argparse.Namespace) -> Path:
    data = load_config(args.config, args.seed)
    dt = sim_mod.parse_time_step(data)
    every_nth = data["simulation_parameters"]["fps"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    traj_path = args.output_dir / f"two_exit_seed{args.seed}.sqlite"
    traj_path.unlink(missing_ok=True)

    walkable, exits = geometry()
    simulation = jps.Simulation(
        model=jps.CollisionFreeSpeedModelV3(),
        geometry=walkable,
        dt=dt,
        trajectory_writer=jps.SqliteTrajectoryWriter(output_file=traj_path, every_nth_frame=every_nth),
    )
    templates = []
    for exit_polygon in exits:
        stage_id = simulation.add_exit_stage(exit_polygon)
        journey_id = simulation.add_journey(jps.JourneyDescription([stage_id]))
        templates.append(agent_template(data, journey_id, stage_id))

    radius = data["velocity_init_parameters"]["radius"]
    starts = jps.distribute_by_number(polygon=START_AREA, number_of_agents=args.agents,
                                      distance_to_agents=2.5 * radius, distance_to_polygon=radius,
                                      seed=args.seed)
    goal_of: dict[int, int] = {}
    members: list[tuple[list[int], list[tuple[float, float]]]] = [([], []) for _ in EXIT_X]
    for x, y in starts:
        goal = min(range(len(EXIT_X)), key=lambda k: abs(x - EXIT_X[k]))
        template = templates[goal]
        template.position = (x, y)
        agent_id = simulation.add_agent(template)
        goal_of[agent_id] = goal
        members[goal][0].append(agent_id)
        members[goal][1].append((x, y))
    models = [goal_model(data, EXIT_X[k], ids, pos) for k, (ids, pos) in enumerate(members)]
    doors = [(x, ROOM.bounds[3]) for x in EXIT_X]

    rows = []
    frame = 0
    while simulation.agent_count() > 0 and simulation.elapsed_time() < args.max_time:
        flags = []
        for k, model in enumerate(models):
            positions = {a.id: tuple(a.position) for a in simulation.agents() if goal_of[a.id] == k}
            flags.append(model.motivation_strategy.update_payoff_cache(
                iteration_count=simulation.iteration_count(),
                agent_positions=positions,
                number_agents_in_simulation=len(positions),
            ))
        if simulation.iteration_count() % every_nth == 0:
            for agent in simulation.agents():
                k = goal_of[agent.id]
                line = sim_mod.process_agent(agent, doors[k], simulation, models[k], None, frame,
                                             int(flags[k]), data)
                rows.append(f"{line}, {k}")
            frame += 1
        simulation.iterate()

    # the writer buffers frames and writes them only on close()
    simulation._writer.close()
    csv_path = args.output_dir / f"two_exit_seed{args.seed}_motivation.csv"
    csv_path.write_text(sim_mod.MOTIVATION_CSV_HEADER + ",goal\n" + "\n".join(rows) + "\n")
    record = {
        "seed": args.seed, "agents": args.agents, "per_goal": [len(ids) for ids, _ in members],
        "end_s": simulation.elapsed_time(), "remaining": simulation.agent_count(),
        "fit_point": FIT_POINT, "config": args.config.name,
    }
    csv_path.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record))
    return csv_path


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    run(parse_args())


if __name__ == "__main__":
    main()
