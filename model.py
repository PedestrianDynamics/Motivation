"""
Simulation model using jpscore API
"""
# Copyright © 2012-2022 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import pathlib
import sys
from typing import Any

import py_jupedsim as jps
from jupedsim.distributions import (
    distribute_by_number,
)

from jupedsim.serialization import JpsCoreStyleTrajectoryWriter

from src.logger_config import init_logger, log_error, log_info
from src.inifile_parser import (
    parse_accessible_areas,
    parse_destinations,
    parse_distribution_polygons,
    parse_fps,
    parse_time_step,
    parse_velocity_model_parameter_profiles,
    parse_way_points,
)
from src.utilities import (
    build_areas,
    build_geometry,
    build_velocity_model,
    distribute_and_add_agents,
    init_journey,
    init_velocity_agent_parameters,
)


def init_simulation(_data: dict, _time_step: float) -> Any:
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
    parameter_profiles = parse_velocity_model_parameter_profiles(_data)

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
    return simulation


def run_simulation(
    simulation,
    writer,
    ped_ids,
):
    """Run simulation logic

    :param simulation:
    :type simulation:
    :param writer:
    :type writer:
    :param ped_ids:
    :type ped_ids:
    :returns:

    """
    test_id = ped_ids[0]
    actual_profile = 1
    while simulation.agent_count() > 0:
        simulation.iterate()
        if simulation.iteration_count() % fps == 0:
            writer.write_iteration_state(simulation)

        if simulation.iteration_count() > 0 and simulation.iteration_count() < 500:
            actual_profile = 2

        if simulation.iteration_count() > 500 and simulation.iteration_count() < 700:
            actual_profile = 1

        try:
            simulation.switch_agent_profile(agent_id=test_id, profile_id=actual_profile)
        except RuntimeError:
            log_error(
                f"""Can not change Profile of Agent
                {test_id} to Profile={actual_profile} at
                Iteration={simulation.iteration_count()}."""
            )
            # end the simulation
            break

    writer.end_writing()
    log_info(f"Simulation completed after {simulation.iteration_count()} iterations")


def main(_fps: int, _time_step: float, _data: dict, _trajectory_path: pathlib.Path):
    """Main simulation loop

    :param fps:
    :param dt:
    :param ini_file:
    :param trajectory_file:
    :returns:

    """
    simulation = init_simulation(_data, _time_step)

    way_points = list(parse_way_points(_data).values())
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

    run_simulation(simulation, writer, ped_ids)
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
