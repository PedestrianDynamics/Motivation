# Copyright © 2012-2022 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import pathlib
from sys import argv, path

from src.inifile_parser import parse_time_step

path.append("./src")
import py_jupedsim as jps
from jupedsim.distributions import (
    distribute_by_number,
    distribute_by_percentage,
    distribute_till_full,
)
from jupedsim.serialization import JpsCoreStyleTrajectoryWriter

from configs import init_logger, log_error, log_info
from inifile_parser import (
    parse_accessible_areas,
    parse_destinations,
    parse_distribution_polygons,
    parse_time_step,
    parse_fps,
    parse_velocity_model_parameter_profiles,
    parse_way_points,
)
from utilities import (
    build_areas,
    build_geometry,
    build_velocity_model,
    distribute_and_add_agents,
    init_journey,
    init_velocity_agent_parameters,
)


def main(fps: int, dt: float, data: str, trajectory_path: pathlib.Path):
    """Main simulation loop

    :param fps:
    :param dt:
    :param ini_file:
    :param trajectory_file:
    :returns:

    """
    accessible_areas = parse_accessible_areas(data)
    geometry = build_geometry(accessible_areas)
    destinations = parse_destinations(data)
    labels = ["exit"]  # todo --> json file
    areas = build_areas(destinations, labels)
    parameter_profiles = parse_velocity_model_parameter_profiles(data)
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
    distribution_polygons = parse_distribution_polygons(data)
    positions = []

    for s_polygon in distribution_polygons.values():
        log_info(f"Distribute agents in {s_polygon}")
        # pos = distribute_by_percentage(
        #     polygon=s_polygon,
        #     percent=20,
        #     distance_to_agents=0.30,
        #     distance_to_polygon=0.20,
        #     seed=45131502,
        # )
        # pos = distribute_till_full(polygon=s_polygon,
        #     distance_to_agents=0.30,
        #     distance_to_polygon=0.20,
        #     seed=45131502,)
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
    if len(argv) < 3:
        exit(f"usage: {argv[0]} inifile.json trajectory.txt")

    f = open(argv[1], "r")
    json_str = f.read()
    f.close()
    data = json.loads(json_str)
    fps = parse_fps(data)
    time_step = parse_time_step(data)
    main(
        fps=fps,
        dt=time_step,
        data=data,
        trajectory_path=pathlib.Path(argv[2]),
    )
