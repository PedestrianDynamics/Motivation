"""
Simulation model using jpscore API
"""
# Copyright © 2012-2022 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import pathlib
import sys
from typing import Any, Dict, List, Tuple, TypeAlias

import jupedsim as jps
from jupedsim.distributions import distribute_by_number
from jupedsim.serialization import JpsCoreStyleTrajectoryWriter

from src import motivation_model as mm, profiles as pp

from src.inifile_parser import (
    parse_accessible_areas,
    parse_destinations,
    parse_distribution_polygons,
    parse_fps,
    parse_time_step,
    parse_way_points,
    parse_number_agents,
    parse_normal_v_0,
    parse_normal_time_gap,
    parse_motivation_doors,
    parse_grid_time_gap_step,
    parse_grid_max_time_gap,
    parse_grid_min_time_gap,
    parse_grid_min_v0,
    parse_grid_max_v0,
    parse_grid_step_v0,
    is_motivation_active,
)
from src.logger_config import init_logger, log_error, log_info, log_debug
from src.utilities import (
    build_geometry,
    build_velocity_model,
    distribute_and_add_agents,
    init_journey,
    init_velocity_agent_parameters,
)

Point: TypeAlias = Tuple[float, float]


def init_simulation(
    _data: Dict[str, Any], _time_step: float
) -> Tuple[Any, pp.ParameterGrid, mm.MotivationModel]:
    """Setup geometry and parameter profiles,

    :param data:
    :type data: str
    :param time_step:
    :type time_step: float
    :returns:

    """
    accessible_areas = parse_accessible_areas(_data)

    grid = pp.ParameterGrid(
        min_v_0=parse_grid_min_v0(_data),
        max_v_0=parse_grid_max_v0(_data),
        v_0_step=parse_grid_step_v0(_data),
        min_time_gap=parse_grid_min_time_gap(_data),
        max_time_gap=parse_grid_max_time_gap(_data),
        time_gap_step=parse_grid_time_gap_step(_data),
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

    geometry = build_geometry(accessible_areas)
    # areas = build_areas(destinations, labels)
    init_parameters = {"a_ped": 8, "d_ped": 0.1, "a_wall": 5, "d_wall": 0.02}
    model = build_velocity_model(
        init_parameters,
        parameter_profiles=parameter_profiles,
    )
    simulation = jps.Simulation(model=model, geometry=geometry, dt=_time_step)
    normal_v_0 = parse_normal_v_0(_data)
    normal_time_gap = parse_normal_time_gap(_data)
    motivation_doors = parse_motivation_doors(_data)
    if not motivation_doors:
        log_error("json file does not contain any motivation door")

    motivation_model = mm.MotivationModel(
        door_point1=(motivation_doors[0][0][0], motivation_doors[0][0][1]),
        door_point2=(motivation_doors[0][1][0], motivation_doors[0][1][1]),
        normal_v_0=normal_v_0,
        normal_time_gap=normal_time_gap,
        active=is_motivation_active(_data),
    )
    motivation_model.print_details()
    log_info("Init simulation done")
    return simulation, grid, motivation_model


def update_profiles(
    simulation: Any, grid: pp.ParameterGrid, motivation_model: mm.MotivationModel
) -> None:
    """Switch profile of pedestrian depending on its motivation"""

    # TODO get neighbors
    # JPS_Simulation_AgentsInRange(JPS_Simulation handle, JPS_Point position, double distance);
    agents = simulation.agents()

    for agent in agents:
        position = agent.position
        actual_profile = agent.profile_id
        (
            new_profile,
            motivation_i,
            v_0,
            time_gap,
            distance,
        ) = motivation_model.get_profile_number(position, grid)
        try:
            # TODO is motivation is active

            simulation.switch_agent_profile(agent_id=agent.id, profile_id=new_profile)

            # log_error(
            #     f"{agent.id}, {position}, {distance=:.2f}, {new_profile=}, {actual_profile=}, {motivation_i=:.2}, {v_0=:.2}, {time_gap=:.2}"
            # )
        except RuntimeError:
            # pass
            log_error(
                f"""Can not change Profile of Agent {agent.id}
                to Profile={actual_profile} at
                Iteration={simulation.iteration_count()}."""
            )


def run_simulation(
    simulation: Any,
    writer: Any,
    grid: pp.ParameterGrid,
    motivation_model: mm.MotivationModel,
) -> None:
    """Run simulation logic

    :param simulation:
    :type simulation:
    :param writer:
    :type writer:
    :returns:

    """
    while simulation.agent_count() > 0:
        simulation.iterate()
        if motivation_model.active and simulation.iteration_count() % 10 == 0:
            update_profiles(simulation, grid, motivation_model)

        if simulation.iteration_count() % 10 == 0:
            writer.write_iteration_state(simulation)


def main(
    _number_agents: int,
    _fps: int,
    _time_step: float,
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
    simulation, grid, motivation_model = init_simulation(_data, _time_step)
    way_points = parse_way_points(_data)
    destinations_dict = parse_destinations(_data)
    destinations = list(destinations_dict.values())
    journey_id = init_journey(simulation, way_points, destinations[0])

    agent_parameters = init_velocity_agent_parameters(
        phi_x=1, phi_y=0, journey=journey_id, profile=1
    )
    distribution_polygons = parse_distribution_polygons(_data)
    positions = []

    total_agents = _number_agents
    for s_polygon in distribution_polygons.values():
        log_info(f"Distribute agents in {s_polygon}")
        pos = distribute_by_number(
            polygon=s_polygon,
            number_of_agents=total_agents,
            distance_to_agents=0.30,
            distance_to_polygon=0.20,
            seed=45131502,
        )
        total_agents -= _number_agents
        positions += pos

    ped_ids = distribute_and_add_agents(simulation, agent_parameters, positions)

    log_info(f"Running simulation for {len(ped_ids)} agents:")
    writer = JpsCoreStyleTrajectoryWriter(_trajectory_path)
    writer.begin_writing(_fps)
    run_simulation(simulation, writer, grid, motivation_model)
    writer.end_writing()
    log_info(f"Simulation completed after {simulation.iteration_count()} iterations")
    log_info(
        f"{time_step}, {simulation.iteration_count()},  {simulation.iteration_count()*time_step}"
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
        if fps and time_step:
            main(
                number_agents,
                fps,
                time_step,
                data,
                pathlib.Path(sys.argv[2]),
            )
