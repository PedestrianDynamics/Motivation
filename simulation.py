"""
Simulation model using jpscore API
"""
import contextlib

# Copyright © 2012-2022 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import pathlib
import sys
import time
from typing import Any, Dict, List, Tuple, TypeAlias

import jupedsim as jps

# import cProfile
# import pstats

from jupedsim.distributions import distribute_by_number
from jupedsim.serialization import JpsCoreStyleTrajectoryWriter

from src import motivation_model as mm
from src import profiles as pp
from src.inifile_parser import (
    is_motivation_active,
    parse_accessible_areas,
    parse_destinations,
    parse_distribution_polygons,
    parse_fps,
    parse_grid_max_time_gap,
    parse_grid_max_v0,
    parse_grid_min_time_gap,
    parse_grid_min_v0,
    parse_grid_step_v0,
    parse_grid_time_gap_step,
    parse_motivation_doors,
    parse_normal_time_gap,
    parse_normal_v_0,
    parse_number_agents,
    parse_simulation_time,
    parse_time_step,
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

Point: TypeAlias = Tuple[float, float]


def write_value_to_file(file_handle, value):
    file_handle.write(str(value) + "\n")


@contextlib.contextmanager
def profile_function(name: str):
    start_time = time.perf_counter_ns()
    yield  # <-- your code will execute here
    total_time = time.perf_counter_ns() - start_time
    log_debug(f"{name}: {total_time / 1000000.0:.4f} ms")


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
    simulation: Any,
    grid: pp.ParameterGrid,
    motivation_model: mm.MotivationModel,
    file_handle,
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
            simulation.switch_agent_profile(agent_id=agent.id, profile_id=new_profile)
            write_value_to_file(
                file_handle, f"{position[0]} {position[1]} {motivation_i}"
            )
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

    def update_profiles(
        simulation: Any,
        grid: pp.ParameterGrid,
        motivation_model: mm.MotivationModel,
        file_handle,
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
                simulation.switch_agent_profile(
                    agent_id=agent.id, profile_id=new_profile
                )
                write_value_to_file(
                    file_handle, f"{position[0]} {position[1]} {motivation_i}"
                )
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
    _simulation_time: float,
) -> None:
    """Run simulation logic

    :param simulation:
    :type simulation:
    :param writer:
    :type writer:
    :returns:

    """
    # profiler = cProfile.Profile()
    # profiler.enable()
    # deltas = []
    with open("values.txt", "w") as file_handle:
        while (
            simulation.agent_count() > 0 and simulation.elapsed_time() < simulation_time
        ):
            simulation.iterate()

            if motivation_model.active and simulation.iteration_count() % 100 == 0:
                with profile_function("update profiles"):
                    # time_1 = time.perf_counter_ns()
                    update_profiles(simulation, grid, motivation_model, file_handle)
                    # time_2 = time.perf_counter_ns()
                    # delta = time_2 - time_1
                    # deltas.append(delta / 1000000)
                    # print(
                    #    f"{simulation.agent_count()}: {simulation.iteration_count()}: {delta=}"
                    # )

            if simulation.iteration_count() % 10 == 0:
                writer.write_iteration_state(simulation)

    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.dump_stats("profile_stats.prof")  # Save to file
    # if deltas:
    #     print(
    #         f"{np.min(deltas)=}, {np.max(deltas)=}, {np.mean(deltas)=}, {np.median(deltas)=}"
    #     )


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
    run_simulation(simulation, writer, grid, motivation_model, _simulation_time)
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
