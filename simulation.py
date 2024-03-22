"""Simulation model using jpscore API."""

# Copyright © 2012-2022 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: LGPL-3.0-or-later
import contextlib
import json
import logging
import pathlib
import sys
import time
from typing import Any, Dict, Iterator, List, Tuple, TypeAlias

import _io
import jupedsim as jps
import streamlit as st
from jupedsim.distributions import distribute_by_number

from src import motivation_model as mm
from src.inifile_parser import (parse_accessible_areas, parse_destinations,
                                parse_distribution_polygons, parse_fps,
                                parse_motivation_doors,
                                parse_motivation_parameter,
                                parse_motivation_strategy,
                                parse_normal_time_gap, parse_normal_v_0,
                                parse_number_agents, parse_radius,
                                parse_simulation_time, parse_time_step,
                                parse_velocity_init_parameters,
                                parse_way_points)
from src.logger_config import init_logger, log_debug, log_error
from src.utilities import (build_geometry, distribute_and_add_agents,
                           init_journey)

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


def init_motivation_model(
    _data: Dict[str, Any], ped_ids: List[int]
) -> mm.MotivationModel:
    """Init motuvation model based on parsed streategy."""
    width = parse_motivation_parameter(_data, "width")
    height = parse_motivation_parameter(_data, "height")
    seed = parse_motivation_parameter(_data, "seed")
    min_value = parse_motivation_parameter(_data, "min_value")
    max_value = parse_motivation_parameter(_data, "max_value")
    motivation_doors = parse_motivation_doors(_data)
    if not motivation_doors:
        log_error("json file does not contain any motivation door")

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
        motivation_strategy = mm.EVCStrategy(
            width=width,
            height=height,
            max_reward=number_agents,
            seed=seed,
            max_value=max_value,
            min_value=min_value,
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
            max_value=max_value,
            min_value=min_value,
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
    fig = motivation_model.plot()
    st.pyplot(fig)
    motivation_model.print_details()
    return motivation_model


def init_simulation(
    _data: Dict[str, Any],
    _time_step: float,
    _fps: int,
    _trajectory_path: pathlib.Path,
) -> Any:
    """Initialise geometry.

    :param data:
    :type data: str
    :param time_step:
    :type time_step: float
    :returns:
    """
    accessible_areas = parse_accessible_areas(_data)
    geometry = build_geometry(accessible_areas)
    # areas = build_areas(destinations, labels)
    a_ped, d_ped, a_wall, d_wall = parse_velocity_init_parameters(_data)
    simulation = jps.Simulation(
        model=jps.CollisionFreeSpeedModel(
            strength_neighbor_repulsion=a_ped,
            range_neighbor_repulsion=d_ped,
            strength_geometry_repulsion=a_wall,
            range_geometry_repulsion=d_wall,
        ),
        geometry=geometry,
        dt=_time_step,
        trajectory_writer=jps.SqliteTrajectoryWriter(
            output_file=pathlib.Path(_trajectory_path), every_nth_frame=_fps
        ),
    )
    logging.info("Init simulation done")
    return simulation


def run_simulation(
    simulation: jps.Simulation,
    motivation_model: mm.MotivationModel,
    _simulation_time: float,
    ped_ids: List[int],
    msg: Any,
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
    x_door = 0.5 * (motivation_model.door_point1[0] + motivation_model.door_point2[0])
    y_door = 0.5 * (motivation_model.door_point1[1] + motivation_model.door_point2[1])
    door = [x_door, y_door]
    logging.info("init initial speed")
    for agent_id in ped_ids:
        value_agent = motivation_model.motivation_strategy.get_value(agent_id=agent_id)
        simulation.agent(agent_id).model.v0 *= value_agent

    with open("values.txt", "w", encoding="utf-8") as file_handle:
        while (
            simulation.agent_count() > 0
            and simulation.elapsed_time() < _simulation_time
        ):
            simulation.iterate()

            msg.code(f"Agents in the simulation: {simulation.agent_count()}")
            if simulation.iteration_count() % 100 == 0:
                agents = simulation.agents()
                number_agents_in_simulation = simulation.agent_count()
                for agent_id, agent in zip(ped_ids, agents):
                    position = agent.position
                    distance = (
                        (position[0] - door[0]) ** 2 + (position[1] - door[1]) ** 2
                    ) ** 0.5
                    params = {
                        "agent_id": agent_id,
                        "distance": distance,
                        "number_agents_in_simulation": number_agents_in_simulation,
                    }
                    motivation_i = motivation_model.motivation_strategy.motivation(
                        params
                    )
                    v_0, time_gap = motivation_model.calculate_motivation_state(
                        motivation_i,
                        agent_id
                    )
                    agent.model.v0 = v_0
                    agent.model.time_gap = time_gap
                    # if agent.id == 1:
                    # logging.info(
                    #     f"Agents: {agent.id},{v_0 = :.2f}, {time_gap = :.2f}, {motivation_i = }, Pos: {position[0]:.2f} {position[1]:.2f}"
                    # )
                    write_value_to_file(
                        file_handle,
                        f"{position[0]} {position[1]} {motivation_i} {v_0} {time_gap} {distance}",
                    )


def main(
    _number_agents: int,
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
    simulation = init_simulation(_data, _time_step, _fps, _trajectory_path)
    way_points = parse_way_points(_data)
    destinations_dict = parse_destinations(_data)
    destinations = list(destinations_dict.values())
    journey_id, stage_id = init_journey(simulation, way_points, destinations[0])
    distribution_polygons = parse_distribution_polygons(_data)
    positions = []

    total_agents = _number_agents
    for s_polygon in distribution_polygons.values():
        logging.info(f"Distribute {total_agents} agents in {s_polygon}")
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

    normal_v_0 = parse_normal_v_0(_data)
    normal_time_gap = parse_normal_time_gap(_data)
    radius = parse_radius(_data)
    agent_parameters = jps.CollisionFreeSpeedModelAgentParameters(
        journey_id=journey_id,
        stage_id=stage_id,
        radius=radius,
        v0=normal_v_0,
        time_gap=normal_time_gap,
    )
    ped_ids = distribute_and_add_agents(simulation, agent_parameters, positions)
    motivation_model = init_motivation_model(_data, ped_ids)
    logging.info(f"Running simulation for {len(ped_ids)} agents:")
    run_simulation(simulation, motivation_model, _simulation_time, ped_ids, msg)
    logging.info(
        f"Simulation completed after {simulation.iteration_count()} iterations"
    )
    logging.info(f"simulation time: {simulation.iteration_count()*_time_step} [s]")
    # logging.info(f"Trajectory: {_trajectory_path}")
    return float(simulation.iteration_count() * _time_step)


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
        dummy = ""
        if fps and time_step:
            evac_time = main(
                number_agents,
                fps,
                time_step,
                simulation_time,
                data,
                pathlib.Path(sys.argv[2]),
                dummy,
            )
        print(f"{evac_time = }")
