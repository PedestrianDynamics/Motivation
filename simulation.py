"""Simulation model using jpscore API."""

# Copyright © 2012-2022 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: LGPL-3.0-or-later

import contextlib
import json
import pathlib
import sys
import time
from typing import Any, Dict, Iterator, List, Tuple, TypeAlias

import _io
import jupedsim as jps
from jupedsim.distributions import distribute_by_number
from jupedsim.serialization import JpsCoreStyleTrajectoryWriter

from src import motivation_model as mm
from src.inifile_parser import (
    is_motivation_active,
    parse_accessible_areas,
    parse_destinations,
    parse_distribution_polygons,
    parse_fps,
    parse_motivation_doors,
    parse_motivation_parameter,
    parse_motivation_strategy,
    parse_normal_time_gap,
    parse_normal_v_0,
    parse_number_agents,
    parse_simulation_time,
    parse_time_step,
    parse_velocity_init_parameters,
    parse_way_points,
)
from src.logger_config import init_logger, log_debug, log_error, log_info
from src.utilities import (
    build_geometry,
    build_velocity_model,
    distribute_and_add_agents,
    init_journey,
    init_velocity_agent_parameters,
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
    log_debug(f"{name}: {total_time / 1000000.0:.4f} ms")


def init_simulation(
    _data: Dict[str, Any], _time_step: float
) -> Tuple[Any, mm.MotivationModel]:
    """Initialise geometry.

    :param data:
    :type data: str
    :param time_step:
    :type time_step: float
    :returns:

    """
    width = parse_motivation_parameter(data, "width")
    height = parse_motivation_parameter(data, "height")
    seed = parse_motivation_parameter(data, "seed")
    min_value = parse_motivation_parameter(data, "min_value")
    max_value = parse_motivation_parameter(data, "max_value")

    accessible_areas = parse_accessible_areas(_data)
    geometry = build_geometry(accessible_areas)
    # areas = build_areas(destinations, labels)
    a_ped, d_ped, a_wall, d_wall = parse_velocity_init_parameters(_data)
    init_parameters = {
        "a_ped": a_ped,
        "d_ped": d_ped,
        "a_wall": a_wall,
        "d_wall": d_wall,
    }
    model = build_velocity_model(
        init_parameters,
    )

    simulation = jps.Simulation(model=model, geometry=geometry, dt=_time_step)
    normal_v_0 = parse_normal_v_0(_data)
    normal_time_gap = parse_normal_time_gap(_data)
    motivation_doors = parse_motivation_doors(_data)
    if not motivation_doors:
        log_error("json file does not contain any motivation door")

    choose_motivation_strategy = parse_motivation_strategy(data)

    # =================
    if choose_motivation_strategy == "default":
        motivation_strategy = mm.DefaultMotivationStrategy(width=width, height=height)
    if choose_motivation_strategy == "EVC":
        motivation_strategy = mm.EVCStrategy(
            width=width,
            height=height,
            max_reward=parse_number_agents(data),
            seed=seed,
            max_value=max_value,
            min_value=min_value,
        )
    # =================

    motivation_model = mm.MotivationModel(
        door_point1=(motivation_doors[0][0][0], motivation_doors[0][0][1]),
        door_point2=(motivation_doors[0][1][0], motivation_doors[0][1][1]),
        normal_v_0=normal_v_0,
        normal_time_gap=normal_time_gap,
        active=is_motivation_active(_data),
        motivation_strategy=motivation_strategy,
    )
    motivation_model.print_details()
    log_info("Init simulation done")
    return simulation, motivation_model


def run_simulation(
    simulation: Any,
    writer: Any,
    motivation_model: mm.MotivationModel,
    _simulation_time: float,
) -> None:
    """Run simulation logic.

    :param simulation:
    :type simulation:
    :param writer:
    :type writer:
    :returns:

    """
    # profiler = cProfile.Profile()
    # profiler.enable()
    # deltas = []
    with open("values.txt", "w", encoding="utf-8") as file_handle:
        while (
            simulation.agent_count() > 0 and simulation.elapsed_time() < simulation_time
        ):
            simulation.iterate()

            if simulation.iteration_count() % 10 == 0:
                writer.write_iteration_state(simulation)


def main(
    _number_agents: int,
    _fps: int,
    _time_step: float,
    _simulation_time: float,
    _data: Dict[str, Any],
    _trajectory_path: pathlib.Path,
) -> None:
    """Main simulation loop

    :param fps:
    :param dt:
    :param ini_file:
    :param trajectory_file:
    :returns:

    """
    simulation, motivation_model = init_simulation(_data, _time_step)
    way_points = parse_way_points(_data)
    destinations_dict = parse_destinations(_data)
    destinations = list(destinations_dict.values())
    journey_id = init_journey(simulation, way_points, destinations[0])

    agent_parameters = init_velocity_agent_parameters(
        phi_x=1, phi_y=0, journey=journey_id
    )
    distribution_polygons = parse_distribution_polygons(_data)
    positions = []

    total_agents = _number_agents
    for s_polygon in distribution_polygons.values():
        log_info(f"Distribute {total_agents} agents in {s_polygon}")
        pos = distribute_by_number(
            polygon=s_polygon,
            number_of_agents=total_agents,
            distance_to_agents=0.4,
            distance_to_polygon=0.2,
            seed=45131502,
        )
        total_agents -= _number_agents
        positions += pos
        if not total_agents:
            break

    ped_ids = distribute_and_add_agents(simulation, agent_parameters, positions)

    log_info(f"Running simulation for {len(ped_ids)} agents:")
    writer = JpsCoreStyleTrajectoryWriter(_trajectory_path)
    writer.begin_writing(_fps)
    run_simulation(simulation, writer, motivation_model, _simulation_time)
    writer.end_writing()
    log_info(f"Simulation completed after {simulation.iteration_count()} iterations")
    log_info(
        f"{time_step}, {simulation.iteration_count()=},  simulation time: {simulation.iteration_count()*time_step} [s]"
    )
    # log_info(f"Trajectory: {_trajectory_path}")


if __name__ == "__main__":
    init_logger()
    if len(sys.argv) < 3:
        sys.exit(f"usage: {sys.argv[0]} inifile.json trajectory.txt")

    with open(sys.argv[1], "r", encoding="utf8") as f:
        json_str = f.read()
        data = json.loads(json_str)
        fps = parse_fps(data)
        time_step = parse_time_step(data)
        number_agents = parse_number_agents(data)
        simulation_time = parse_simulation_time(data)
        if fps and time_step:
            main(
                number_agents,
                fps,
                time_step,
                simulation_time,
                data,
                pathlib.Path(sys.argv[2]),
            )
