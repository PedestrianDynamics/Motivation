"""Simulation model using jpscore API."""

# Copyright © 2012-2022 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: LGPL-3.0-or-later
import _io
import contextlib
import csv
import json
import logging
import pathlib
import time
from typing import Any, Dict, Iterator, List, Tuple, TypeAlias, cast

import jupedsim as jps
import typer
from jupedsim.distributions import distribute_by_number
from shapely import from_wkt

from src import motivation_model as mm
from src.inifile_parser import (
    parse_accessible_areas,
    parse_destinations,
    parse_distribution_polygons,
    parse_fps,
    parse_motivation_doors,
    parse_motivation_strategy,
    parse_normal_time_gap,
    parse_normal_v_0,
    parse_number_agents,
    parse_radius,
    parse_simulation_time,
    parse_time_step,
    parse_velocity_init_parameters,
    parse_way_points,
)
from src.logger_config import init_logger
from src.utilities import (
    build_geometry,
    calculate_distance,
    distribute_and_add_agents,
    init_journey,
)

# import cProfile
# import pstats


Point: TypeAlias = Tuple[float, float]


def write_value_to_file(file_handle: _io.TextIOWrapper, value: str) -> None:
    """Write motivation information for ploting as heatmap."""
    file_handle.write(value + "\n")


@contextlib.contextmanager
def profile_function(name: str) -> Iterator[None]:
    """Profile function. use with <with> and name it <name>."""
    start_time = time.perf_counter_ns()
    yield  # <-- your code will execute here
    total_time = time.perf_counter_ns() - start_time
    logging.info(f"{name}: {total_time / 1000000.0:.4f} ms")


def init_motivation_model(
    _data: Dict[str, Any], ped_ids: List[int]
) -> mm.MotivationModel:
    """Init motivation model based on parsed strategy."""
    width = _data["motivation_parameters"]["width"]
    height = _data["motivation_parameters"]["height"]
    seed = _data["motivation_parameters"]["seed"]
    motivation_doors = parse_motivation_doors(_data)
    if not motivation_doors:
        logging.info("json file does not contain any motivation door.")

    normal_v_0 = parse_normal_v_0(_data)
    normal_time_gap = parse_normal_time_gap(_data)
    choose_motivation_strategy = parse_motivation_strategy(_data)
    number_agents = parse_number_agents(_data)
    competition_max = _data["motivation_parameters"]["competition_max"]
    competition_decay_reward = _data["motivation_parameters"][
        "competition_decay_reward"
    ]
    percent = _data["motivation_parameters"]["percent"]
    # =================
    motivation_strategy: mm.MotivationStrategy
    if choose_motivation_strategy == "default":
        motivation_strategy = mm.DefaultMotivationStrategy(width=width, height=height)
    if choose_motivation_strategy == "EVC":
        logging.info(f"init EVC with {width = }, {height = }")
        motivation_strategy = mm.EVCStrategy(
            width=width,
            height=height,
            max_reward=number_agents,
            seed=seed,
            max_value_high=float(_data["motivation_parameters"]["max_value_high"]),
            min_value_high=float(_data["motivation_parameters"]["min_value_high"]),
            max_value_low=float(_data["motivation_parameters"]["max_value_low"]),
            min_value_low=float(_data["motivation_parameters"]["min_value_low"]),
            number_high_value=int(_data["motivation_parameters"]["number_high_value"]),
            nagents=number_agents,
            agent_ids=ped_ids,
            competition_decay_reward=competition_decay_reward,
            competition_max=competition_max,
            percent=percent,
            evc=True,
        )
    if choose_motivation_strategy == "EC-V":
        motivation_strategy = mm.EVCStrategy(
            width=width,
            height=height,
            max_reward=number_agents,
            seed=seed,
            max_value_high=float(_data["motivation_parameters"]["max_value_high"]),
            min_value_high=float(_data["motivation_parameters"]["min_value_high"]),
            max_value_low=float(_data["motivation_parameters"]["max_value_low"]),
            min_value_low=float(_data["motivation_parameters"]["min_value_low"]),
            number_high_value=int(_data["motivation_parameters"]["number_high_value"]),
            nagents=number_agents,
            agent_ids=ped_ids,
            competition_decay_reward=competition_decay_reward,
            competition_max=competition_max,
            percent=percent,
            evc=False,
        )
    # =================
    motivation_model = mm.MotivationModel(
        door_point1=(motivation_doors[0][0][0], motivation_doors[0][0][1]),
        door_point2=(motivation_doors[0][1][0], motivation_doors[0][1][1]),
        normal_v_0=normal_v_0,
        normal_time_gap=normal_time_gap,
        motivation_strategy=motivation_strategy,
    )
    motivation_model.print_details()
    return motivation_model


def init_simulation(
    _data: Dict[str, Any],
    _time_step: float,
    _fps: int,
    _trajectory_path: pathlib.Path,
    from_file: bool = True,
) -> Any:
    """Initialize geometry.

    :param data:
    :type data: str
    :param time_step:
    :type time_step: float
    :returns:
    """
    accessible_areas = parse_accessible_areas(_data)

    if from_file:
        logging.info(f"Init geometry from WKT")
        geometry = from_wkt(
            "POLYGON ((-8.88 -7.63, 8.3 -7.63, 8.3 27.95, -8.88 27.95, -8.88 -7.63), (-3.54 -1.13, -3.57 19.57, -1.52 19.57, -1.37 19.71,  -0.87 19.71, -0.72 19.57, -0.42 19.57, -0.27 19.71, -0.27 21.09, -0.42 21.23, -0.72 21.23, -0.87 21.09, -1.37 21.09, -1.52 21.23, -1.67 21.23, -1.67 21.18, -1.545 21.18, -1.4200000000000002 21.065, -1.4200000000000002 19.735, -1.545 19.62, -3.6199999999999997 19.62, -3.59 -1.13, -3.54 -1.13), (3.57 -0.89, 3.64 19.64, 1.47 19.57, 1.32 19.71, 0.82 19.71, 0.67 19.57, 0.38 19.57, 0.23 19.71, 0.23 21.09, 0.38 21.23, 0.67 21.23, 0.82 21.09, 1.32 21.09, 1.47 21.23, 1.62 21.23, 1.62 21.18, 1.4949999999999999 21.18, 1.37 21.065, 1.37 19.735, 1.4949999999999999 19.62, 3.69 19.69, 3.6199999999999997 -0.89, 3.57 -0.89))"
        )
    else:
        logging.info("Init geometry from data")
        geometry = build_geometry(accessible_areas)
    # areas = build_areas(destinations, labels)
    simulation = jps.Simulation(
        model=jps.CollisionFreeSpeedModelV2(),
        geometry=geometry,
        dt=_time_step,
        trajectory_writer=jps.SqliteTrajectoryWriter(
            output_file=pathlib.Path(_trajectory_path), every_nth_frame=_fps
        ),
    )
    logging.info("Init simulation done")
    return simulation


def adjust_parameter_linearly(
    motivation_i: float,
    min_value: float = 0.01,
    default_value: float = 0.5,
    max_value: float = 1.0,
) -> float:
    """
    Adjust the a parameter based on agent's motivation level (0 < motivation_i < 1).

    :param motivation_i: The agent's motivation level, expected to be a positive value less than 1.
    :param min_value: Minimum repulsion range for very low motivation.
    :param default_value: Default repulsion range for mid motivation.
    :param max_value: Maximum repulsion range for high motivation.
    :return: Adjusted range_neighbor_repulsion value.
    """
    # Linear interpolation between min_value and max_value based on motivation_i
    return min_value + (max_value - min_value) * motivation_i


def process_agent(
    agent: jps.Agent,
    door: Point,
    simulation: jps.Simulation,
    motivation_model: mm.MotivationModel,
    a_ped_min: float,
    a_ped_max: float,
    d_ped_min: float,
    d_ped_max: float,
    default_strength: float,
    default_range: float,
    file_handle: _io.TextIOWrapper,
    frame_to_write: int,
) -> str:
    """Process an individual agent by calculating motivation and updating model parameters."""
    position = agent.position
    distance = calculate_distance(position, door)

    params = {
        "agent_id": agent.id,
        "distance": distance,
        "number_agents_in_simulation": simulation.agent_count(),
    }

    motivation_i = motivation_model.motivation_strategy.motivation(params)

    if motivation_i > 1:
        logging.error(f"{simulation.iteration_count()}: {agent.id}: {motivation_i = }")

    v_0, time_gap = motivation_model.calculate_motivation_state(motivation_i, agent.id)

    # Adjust agent parameters based on motivation
    agent.model.strength_neighbor_repulsion = adjust_parameter_linearly(
        motivation_i=motivation_i,
        min_value=a_ped_min,
        default_value=default_strength,
        max_value=a_ped_max,
    )

    agent.model.range_neighbor_repulsion = adjust_parameter_linearly(
        motivation_i=motivation_i,
        min_value=d_ped_min,
        default_value=default_range,
        max_value=d_ped_max,
    )

    agent.model.v0 = v_0
    agent.model.time_gap = time_gap
    return f"{frame_to_write}, {agent.id}, {simulation.elapsed_time():.2f}, {motivation_i:.2f}, {position[0]:.2f}, {position[1]:.2f}"


def run_simulation_loop(
    simulation: jps.Simulation,
    door: Point,
    motivation_model: mm.MotivationModel,
    simulation_time: float,
    a_ped_min: float,
    a_ped_max: float,
    d_ped_min: float,
    d_ped_max: float,
    default_strength: float,
    default_range: float,
    every_nth_frame: int,
    motivation_file: pathlib.Path,
) -> None:
    """Run the simulation loop to process agents and write motivation information to a CSV file.

    Args:
        simulation (jps.Simulation): The simulation instance.
        door (Point): The coordinates of the door.
        motivation_model (mm.MotivationModel): The motivation model used for agents.
        simulation_time (float): The total simulation time.
        a_ped_min (float): Minimum value for adjusting agent strength based on motivation.
        a_ped_max (float): Maximum value for adjusting agent strength based on motivation.
        d_ped_min (float): Minimum value for adjusting agent range based on motivation.
        d_ped_max (float): Maximum value for adjusting agent range based on motivation.
        default_strength (float): Default strength value for agents.
        default_range (float): Default range value for agents.
        every_nth_frame (int): Write to file every nth frame.
        motivation_file (pathlib.Path): Path to the motivation file to write.

    Returns:
        None
    """
    buffer = []
    with open(motivation_file, "w", encoding="utf-8") as file_handle:
        frame_to_write = 0

        while (
            simulation.elapsed_time() < simulation_time and simulation.agent_count() > 0
        ):
            print(f"time: {simulation.elapsed_time():.2f}", end="\r")

            if simulation.iteration_count() % every_nth_frame == 0:
                for agent in simulation.agents():
                    ret = process_agent(
                        agent,
                        door,
                        simulation,
                        motivation_model,
                        a_ped_min,
                        a_ped_max,
                        d_ped_min,
                        d_ped_max,
                        default_strength,
                        default_range,
                        file_handle,
                        frame_to_write,
                    )
                    buffer.append(ret)
                frame_to_write += 1
            simulation.iterate()

        with profile_function("Writing to csv file"):
            for items in buffer:
                write_value_to_file(file_handle, items)


def create_agent_parameters(
    _data: Dict[str, Any], simulation: jps.Simulation
) -> Tuple[List[jps.CollisionFreeSpeedModelV2AgentParameters], List[List[Point]]]:
    """Create the model parameters."""
    way_points = parse_way_points(_data)
    destinations_dict = parse_destinations(_data)
    destinations: List[List[Point]] = cast(
        List[List[Point]], list(destinations_dict.values())
    )
    journey_id, exit_ids = init_journey(simulation, way_points, destinations)

    normal_v_0 = parse_normal_v_0(_data)
    normal_time_gap = parse_normal_time_gap(_data)
    radius = parse_radius(_data)
    agent_parameters_list = []
    a_ped, d_ped, a_wall, d_wall, a_ped_min, a_ped_max, d_ped_min, d_ped_max = (
        parse_velocity_init_parameters(_data)
    )
    for exit_id in exit_ids:
        agent_parameters = jps.CollisionFreeSpeedModelV2AgentParameters(
            journey_id=journey_id,
            stage_id=exit_id,
            radius=radius,
            v0=normal_v_0,
            time_gap=normal_time_gap,
            strength_neighbor_repulsion=a_ped,
            range_neighbor_repulsion=d_ped,
            strength_geometry_repulsion=a_wall,
            range_geometry_repulsion=d_wall,
        )
        agent_parameters_list.append(agent_parameters)

    return (agent_parameters_list, destinations)


def init_positions(_data: Dict[str, Any], _number_agents: int) -> List[Point]:
    """Randomly create positions for distribution of pedestrians."""
    distribution_polygons = parse_distribution_polygons(_data)
    positions = []
    seed = int(_data["motivation_parameters"]["seed"])
    total_agents = _number_agents
    for s_polygon in distribution_polygons.values():
        logging.info(f"Distribute {total_agents} agents")
        pos = distribute_by_number(
            polygon=s_polygon,
            number_of_agents=total_agents,
            distance_to_agents=0.4,
            distance_to_polygon=0.2,
            seed=seed,
        )
        total_agents -= _number_agents
        positions += pos
        if not total_agents:
            break

    return positions


def read_positions_from_csv(file_path: str = "points.csv") -> List[Point]:
    """Read positions generated by notebook from a CSV file if it exists."""
    path = pathlib.Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"The file {file_path} does not exist yet.")

    with path.open("r") as f:
        reader = csv.reader(f)
        points: List[Point] = []
        for row in reader:
            if len(row) == 2:
                try:
                    x, y = float(row[0]), float(row[1])
                    points.append((x, y))
                except ValueError:
                    raise FileNotFoundError(f"The file {file_path} does not exist yet.")

    return points


def get_agent_positions(_data: Dict[str, Any]) -> Tuple[List[Point], int]:
    """Get agent positions either from a file or generate them."""
    if "init_positions_file" in _data:
        positions_file = _data["init_positions_file"]
        if pathlib.Path(positions_file).exists():
            logging.info(f"Reading positions from file: {positions_file}")
            positions = read_positions_from_csv(file_path=positions_file)
            num_agents = len(positions)
            _data["simulation_parameters"]["number_agents"] = num_agents
            logging.info(f"Number of agents from file: {num_agents}")
        else:
            raise FileNotFoundError(f"Positions file {positions_file} does not exist!")
    else:
        num_agents = parse_number_agents(_data)
        logging.info(f"Generating {num_agents} agent positions")
        positions = init_positions(_data, num_agents)

    return positions, num_agents


def print_hello(msg):
    print(f"{msg}")


def init_and_run_simulation(
    _fps: int,
    _time_step: float,
    _simulation_time: float,
    _data: Dict[str, Any],
    _trajectory_path: pathlib.Path,
    msg: Any,
) -> float:
    """Implement simulation loop.

    :param fps:
    :param dt:
    :param ini_file:
    :param trajectory_file:
    :returns:
    """
    motivation_file = _trajectory_path.with_name(
        _trajectory_path.stem + "_motivation.csv"
    )
    logging.info(f"{motivation_file = }")
    simulation = init_simulation(
        _data, _time_step, _fps, _trajectory_path, from_file=True
    )
    a_ped, d_ped, a_wall, d_wall, a_ped_min, a_ped_max, d_ped_min, d_ped_max = (
        parse_velocity_init_parameters(_data)
    )
    agent_parameters_list, exit_positions = create_agent_parameters(_data, simulation)

    positions, num_agents = get_agent_positions(_data)
    ped_ids = distribute_and_add_agents(
        simulation=simulation,
        agent_parameters_list=agent_parameters_list,
        positions=positions,
        exit_positions=exit_positions,
    )
    motivation_model = init_motivation_model(_data, ped_ids)
    x_door = 0.5 * (motivation_model.door_point1[0] + motivation_model.door_point2[0])
    y_door = 0.5 * (motivation_model.door_point1[1] + motivation_model.door_point2[1])
    motivation_door: Point = (x_door, y_door)
    logging.info(f"Running simulation for {len(ped_ids)} agents:")
    logging.info(f"{motivation_model.motivation_strategy.width = }")
    start_time = time.time()
    run_simulation_loop(
        simulation=simulation,
        door=motivation_door,
        motivation_model=motivation_model,
        simulation_time=_simulation_time,
        a_ped_min=a_ped_min,
        a_ped_max=a_ped_max,
        d_ped_min=d_ped_min,
        d_ped_max=d_ped_max,
        default_strength=a_ped,
        default_range=d_ped,
        every_nth_frame=_data["simulation_parameters"]["fps"],
        motivation_file=motivation_file,
    )
    end_time = time.time()
    logging.info(f"Run time: {end_time - start_time:.2f} seconds")
    logging.info(
        f"Simulation completed after {simulation.iteration_count()} iterations"
    )
    logging.info(f"simulation time: {simulation.iteration_count()*_time_step:.2f} [s]")
    # logging.info(f"Trajectory: {_trajectory_path}")
    return float(simulation.iteration_count() * _time_step)


def start_simulation(config_path: str, output_path: str) -> float:
    """Call main function."""
    with open(config_path, "r", encoding="utf8") as f:
        data = json.load(f)
        fps = parse_fps(data)
        time_step = parse_time_step(data)
        simulation_time = parse_simulation_time(data)
        dummy = ""
        if fps and time_step:
            evac_time = init_and_run_simulation(
                fps,
                time_step,
                simulation_time,
                data,
                pathlib.Path(output_path),
                dummy,
            )
        return evac_time


def modify_and_save_config(
    base_config: float, modification_dict: Dict[str, Any], new_config_path: str
) -> None:
    """Modify base configuration and save as a new JSON file."""
    config = json.loads(json.dumps(base_config))  # Deep copy
    for key, value in modification_dict.items():
        nested_keys = key.split("/")
        last_key = nested_keys.pop()
        temp = config
        for nk in nested_keys:
            temp = temp[nk]
        temp[last_key] = value
    with open(new_config_path, "w", encoding="utf8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def main(
    inifile: pathlib.Path = typer.Option(
        pathlib.Path("files/inifile.json"),
        help="Path to the initial configuration file",
    ),
) -> None:
    """Implement Main function. Create variations and start simulations."""
    init_logger()
    logging.info(f"Base config = {inifile}")
    with open(inifile, "r", encoding="utf8") as f:
        base_config = json.load(f)

    variations = [
        {"motivation_parameters/width": 1.0, "motivation_parameters/seed": 1.0},
        # {"motivation_parameters/width": 2.0, "motivation_parameters/seed": 300.0},
    ]

    output_dir = pathlib.Path("files/variations")
    output_dir.mkdir(parents=True, exist_ok=True)

    variations_file = output_dir / "variations.json"
    with open(variations_file, "w") as f:
        json.dump(variations, f, indent=4)

    for i, variation in enumerate(variations, start=1):
        logging.info(f"running variation {i:03d}: {variation}")
        new_config_path = f"{output_dir}/config_variation_{i:03d}.json"
        output_path = f"files/trajectory_variation_{i:03d}.sqlite"
        logging.info(f"{output_path = }")
        # Modify and save the new configuration
        modify_and_save_config(base_config, variation, new_config_path)

        evac_time = start_simulation(new_config_path, output_path)
        logging.info(f"Variation {i:03d}: {evac_time = }")


if __name__ == "__main__":
    typer.run(main)
