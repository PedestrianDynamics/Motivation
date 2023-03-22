# Copyright © 2012-2022 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: LGPL-3.0-or-later
import pathlib
from sys import argv
import py_jupedsim as jps
from configs import init_logger, log_error, log_info
from jupedsim.serialization import JpsCoreStyleTrajectoryWriter
from utilities import (
    build_areas,
    build_geometry,
    build_velocity_model,
    distribute_and_add_agents,
    init_journey,
    init_velocity_agent_parameters,
)

from inifile_parser import (
    parse_accessible_areas,
    parse_destinations,
    parse_fps,
    parse_velocity_model_parameter_profiles,
    parse_way_points,
    parse_fps,
    parse_dt,
)
import json


def main(fps: int, dt: float, data: str, trajectory_path: pathlib.Path):
    """Main simulation loop

    :param fps:
    :param dt:
    :param ini_file:
    :param trajectory_file:
    :returns:

    """
    accessible_areas = parse_accessible_areas(data)
    geometry = build_geometry(accessible_areas.values())
    destinations = parse_destinations(data)
    labels = ["exit", "other-label"]  # todo --> json file
    areas = build_areas(destinations, labels)
    parameter_profiles = parse_velocity_model_parameter_profiles(data)
    print(parameter_profiles)
    # way_points = [ ((19, 5), 0.5)]
    way_points = list(parse_way_points(data).values())
    model = build_velocity_model(
        a_ped=8,
        d_ped=0.1,
        a_wall=5,
        d_wall=0.02,
        parameter_profiles=parameter_profiles,
    )
    # todo: here we already need to know the profiles, and can not calculate them on the fly.
    log_info(f"Init simulation with dt={dt} [s] and fps={fps}")
    simulation = jps.Simulation(model=model, geometry=geometry, areas=areas, dt=dt)
    log_info("Init simulation done")

    journey_id = init_journey(simulation, way_points)
    agent_parameters = init_velocity_agent_parameters(
        phi_x=1, phi_y=0, journey=journey_id, profile=1
    )
    positions = [(7, 7), (1, 3), (1, 5), (1, 7), (2, 7)]
    ped_ids = distribute_and_add_agents(simulation, agent_parameters, positions)
    log_info(f"Running simulation for {len(ped_ids)} agents:")
    writer = JpsCoreStyleTrajectoryWriter(trajectory_path)
    writer.begin_writing(fps)
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
                f"Can not change Profile of Agent {test_id} to Profile={actual_profile} at Iteration={simulation.iteration_count()}."
            )
            # end the simulation
            break

    writer.end_writing()
    log_info(f"Simulation completed after {simulation.iteration_count()} iterations")
    log_info(f"Trajectory: {trajectory_path}")


if __name__ == "__main__":
    init_logger()
    if len(argv) < 2:
        exit(f"usage: {argv[0]} inifile.json")

    f = open(argv[1], "r")
    json_str = f.read()
    f.close()
    data = json.loads(json_str)
    fps = parse_fps(data)
    dt = parse_dt(data)
    main(
        fps=fps,
        dt=dt,
        data=data,
        trajectory_path=pathlib.Path("out.txt"),
    )
