"""
Simulation model using jpscore API
"""
# Copyright © 2012-2022 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import pathlib
import sys
from typing import Any, Dict, List, Tuple, TypeAlias
import src.profiles as pp

import py_jupedsim as jps
from jupedsim.distributions import distribute_by_number
from jupedsim.serialization import JpsCoreStyleTrajectoryWriter

import src.motivation_model as mm
from src.inifile_parser import (
    parse_accessible_areas,
    parse_destinations,
    parse_distribution_polygons,
    parse_fps,
    parse_time_step,
    #parse_velocity_model_parameter_profiles,
    parse_way_points,
)
from src.logger_config import init_logger, log_error, log_info
from src.utilities import (
    build_areas,
    build_geometry,
    build_velocity_model,
    distribute_and_add_agents,
    init_journey,
    init_velocity_agent_parameters,
)

Point: TypeAlias = Tuple[float, float]


def init_simulation(_data: Dict[str, Any], _time_step: float) -> Tuple[Any, pp.ParameterGrid]:
    """Setup geometry and parameter profiles,

    :param data:
    :type data: str
    :param time_step:
    :type time_step: float
    :returns:

    """
    accessible_areas = parse_accessible_areas(_data)
    destinations = parse_destinations(_data)
    labels = ["exit"]  # todo --> json file
    #parameter_profiles = parse_velocity_model_parameter_profiles(_data)
    grid = pp.ParameterGrid(
        min_v_0=1.0,
        max_v_0=2.0,
        v_0_step=0.1,
        min_time_gap=0.1,
        max_time_gap=1,
        time_gap_step=0.1,
    )
    velocity_profiles = grid.velocity_profiles
    parameter_profiles: Dict[int, List[float]] = {}    
    for velocity_profile in velocity_profiles:     
        parameter_profiles[velocity_profile.number] = [
            velocity_profile.time_gap,
            velocity_profile.tau,
            velocity_profile.v_0,
            velocity_profile.radius,
        ]

    print(f"{parameter_profiles=}")
    geometry = build_geometry(accessible_areas)
    areas = build_areas(destinations, labels)
    init_parameters = {"a_ped": 8, "d_ped": 0.1, "a_wall": 5, "d_wall": 0.02}
    model = build_velocity_model(
        init_parameters,
        parameter_profiles=parameter_profiles,
    )
    # todo: here we already need to know the profiles, and can not calculate them on the fly.
    simulation = jps.Simulation(
        model=model, geometry=geometry, areas=areas, dt=_time_step
    )
    log_info("Init simulation done")
    return simulation, grid


def update_profiles(
        simulation: Any, peds_ids: List[int], positions: List[Point], grid:pp.ParameterGrid
) -> None:
    """Switch profile of pedestrian depending on its motivation"""

    for ped_id, position in zip(peds_ids, positions):
        actual_profile = mm.get_profile_number(position, grid)
        try:
            simulation.switch_agent_profile(agent_id=ped_id, profile_id=actual_profile)
        except RuntimeError:
            log_error(
                f"""Can not change Profile of Agent {ped_id}
                to Profile={actual_profile} at
                Iteration={simulation.iteration_count()}."""
            )


def run_simulation(
        simulation: Any, writer: Any, ped_ids: List[int], positions: List[Point], grid:pp.ParameterGrid
) -> None:
    """Run simulation logic

    :param simulation:
    :type simulation:
    :param writer:
    :type writer:
    :param ped_ids:
    :type ped_ids:
    :returns:

    """
    while simulation.agent_count() > 0:
        simulation.iterate()
        # TODO: maybe not every time step
        update_profiles(simulation, ped_ids, positions, grid)

    writer.end_writing()
    log_info(f"Simulation completed after {simulation.iteration_count()} iterations")


def main(
    _fps: int, _time_step: float, _data: Dict[str, Any], _trajectory_path: pathlib.Path
) -> None:
    """Main simulation loop

    :param fps:
    :param dt:
    :param ini_file:
    :param trajectory_file:
    :returns:

    """
    simulation, grid = init_simulation(_data, _time_step)
    print(f"{parse_way_points(_data)=}")
    way_points = parse_way_points(_data)
    print(f"{way_points=}")
    journey_id = init_journey(simulation, way_points)

    agent_parameters = init_velocity_agent_parameters(
        phi_x=1, phi_y=0, journey=journey_id, profile=1
    )
    distribution_polygons = parse_distribution_polygons(_data)
    positions = []

    for s_polygon in distribution_polygons.values():
        log_info(f"Distribute agents in {s_polygon}")
        pos = distribute_by_number(
            polygon=s_polygon,
            number_of_agents=10,
            distance_to_agents=0.30,
            distance_to_polygon=0.20,
            seed=45131502,
        )
        positions += pos

    ped_ids = distribute_and_add_agents(simulation, agent_parameters, positions)

    log_info(f"Running simulation for {len(ped_ids)} agents:")
    writer = JpsCoreStyleTrajectoryWriter(_trajectory_path)
    writer.begin_writing(_fps)

    run_simulation(simulation, writer, ped_ids, positions, grid)
    log_info(f"Trajectory: {_trajectory_path}")


if __name__ == "__main__":
    init_logger()
    if len(sys.argv) < 3:
        sys.exit(f"usage: {sys.argv[0]} inifile.json trajectory.txt")

    with open(sys.argv[1], "r", encoding="utf8") as f:
        json_str = f.read()
        data = json.loads(json_str)
        fps = parse_fps(data)
        time_step = parse_time_step(data)
        if fps and time_step:
            main(
                fps,
                time_step,
                data,
                pathlib.Path(sys.argv[2]),
            )
