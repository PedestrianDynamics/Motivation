"""Simulation model using jpscore API."""

# Copyright © 2012-2022 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: LGPL-3.0-or-later

import subprocess
from datetime import datetime
import numpy as np
import pandas as pd
import _io
import contextlib
import csv
import json
import logging
import pathlib
import time
from typing import Any, Dict, Iterator, List, Tuple, TypeAlias, cast
import random
import jupedsim as jps
import typer
from jupedsim.distributions import distribute_by_number
from shapely import from_wkt, Polygon
from typing import Optional
from xml.etree.ElementTree import Element, ElementTree, SubElement
from jupedsim.internal.notebook_utils import read_sqlite_file

from src import motivation_model as mm
from src import motivation_mapping as mmap
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
    calculate_crossing_density,
    plot_crossing_order_vs_area,
)

# import cProfile
# import pstats


Point: TypeAlias = Tuple[float, float]


def polygon_to_xml(
    walkable_area,
    output_file="geometry3.xml",
):
    """
    Converts a Shapely polygon to an XML format.
    """
    # Root geometry element
    polygon = walkable_area.polygon
    geometry = Element(
        "geometry",
        {
            "version": "0.8",
            "caption": "Projectname",
            "gridSizeX": "20.000000",
            "gridSizeY": "20.000000",
            "unit": "m",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://134.94.2.137/jps_geoemtry.xsd",
        },
    )

    # Rooms container
    rooms = SubElement(geometry, "rooms")
    room = SubElement(
        rooms,
        "room",
        {"id": "0", "caption": "walkable_area", "zpos": "0.000000"},
    )

    # Subroom containing the polygon
    subroom = SubElement(
        room, "subroom", {"id": "0", "closed": "0", "class": "subroom"}
    )
    poly_element = SubElement(subroom, "polygon", {"caption": "walkable_area"})

    # Add vertices to the polygon
    for x, y in polygon.exterior.coords:
        SubElement(poly_element, "vertex", {"px": f"{x:.6f}", "py": f"{y:.6f}"})

    tree = ElementTree(geometry)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"XML file saved to {output_file}")


def compute_speed(traj, df, fps):
    """
    Calculates the speed from the trajectory points with proper padding.
    """
    size = traj.shape[0]
    speed = np.zeros(size)

    if size < df:
        print(
            f"Warning: Trajectory length {size} is shorter than frame difference {df}"
        )
        return np.ones(size)

    # Calculate speeds for the main portion of the trajectory
    delta = traj[df:, :] - traj[:-df, :]
    delta_square = np.square(delta)
    delta_x_square = delta_square[:, 0]
    delta_y_square = delta_square[:, 1]
    s = np.sqrt(delta_x_square + delta_y_square)

    # Place the calculated speeds in the correct position
    speed[df // 2 : -df // 2] = s / df * fps

    # Handle the edges with forward/backward differences
    # Start (use forward difference)
    for i in range(df // 2):
        delta = traj[df : df + 1, :] - traj[i : i + 1, :]
        speed[i] = np.sqrt(np.sum(np.square(delta))) / df * fps

    # End (use backward difference)
    for i in range(size - df // 2, size):
        delta = traj[i : i + 1, :] - traj[size - df - 1 : size - df, :]
        speed[i] = np.sqrt(np.sum(np.square(delta))) / df * fps

    return speed


def export_trajectory_to_txt(
    trajectory_data,
    motivation_model: mm.MotivationModel,
    output_file="output.txt",
    geometry_file="geometry.xml",
    df=10,
    v0=1.2,
    radius=0.18,
    by_speed=True,
):
    """
    Exports trajectory data from a SQLite file to a formatted .txt file, including speed and color.
    """
    df_data = trajectory_data.data
    fps = trajectory_data.frame_rate
    print(f"export_trajectory_to_txt {fps = }")
    # Extract trajectories for speed calculations
    # trajectories = df_data.groupby("id").apply(
    #     lambda group: group.sort_values(by="frame")[["x", "y"]].values
    # )

    speeds = []
    frame_indices = []
    values = []
    for traj_id, group in df_data.groupby("id"):
        # Sort by frame within each trajectory
        group = group.sort_values(by="frame")
        traj = group[["x", "y"]].values
        agent_value = motivation_model.motivation_strategy.get_value(agent_id=traj_id)

        # Calculate speed for this trajectory
        speed = compute_speed(traj, df, fps)
        # if agent_value > 1.0:#green
        values.extend([-1] * len(speed))
        # else:#blue
        #    values.extend([10]*len(speed))

        # Store speed and corresponding frame indices
        speeds.extend(speed)
        frame_indices.extend(group.index)

    df_data["angle"] = np.degrees(np.arctan2(df_data["oy"], df_data["ox"]))
    # Calculate color based on speed
    if by_speed:
        speed_series = pd.Series(speeds, index=frame_indices)
        df_data.loc[frame_indices, "speed"] = speed_series
        df_data["color"] = (df_data["speed"] / v0 * 255).clip(0, 255).astype(int)
    else:
        value_series = pd.Series(values, index=frame_indices)
        df_data.loc[frame_indices, "value"] = value_series
        df_data["color"] = df_data["value"]

    # Write the formatted data to the output file
    with open(output_file, "w") as f:
        # Write the header
        f.write(f"#framerate: {fps}\n")
        f.write("#unit: m\n")
        f.write(f"#geometry: {geometry_file}\n")
        f.write("#ID\tFR\tX\tY\tZ\tA\tB\tANGLE\tCOLOR\n")

        # Write each row of trajectory data
        for _, row in df_data.iterrows():
            f.write(
                f"{row['id']}\t{row['frame']}\t{row['x']:.4f}\t{row['y']:.4f}\t0\t"
                f"{radius:.4f}\t{radius:.4f}\t{row['angle']:.4f}\t{row['color']}\n"
            )


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
    _data: Dict[str, Any],
    ped_ids: List[int],
    ped_positions: List[Point],
) -> mm.MotivationModel:
    """Init motivation model based on parsed strategy."""
    width = _data["motivation_parameters"]["width"]
    height = _data["motivation_parameters"]["height"]
    seed = _data["motivation_parameters"]["seed"]
    motivation_doors = parse_motivation_doors(_data)
    logging.info("Enter init motivation model")
    if not motivation_doors:
        logging.info("json file does not contain any motivation door.")

    door_point1 = (motivation_doors[0][0][0], motivation_doors[0][0][1])
    door_point2 = (motivation_doors[0][1][0], motivation_doors[0][1][1])
    x_door = 0.5 * (door_point1[0] + door_point2[0])
    y_door = 0.5 * (door_point1[1] + door_point2[1])
    motivation_door_center: Point = (x_door, y_door)

    normal_v_0 = parse_normal_v_0(_data)
    normal_time_gap = parse_normal_time_gap(_data)
    mapping_block = mmap.ensure_mapping_block(_data["motivation_parameters"])
    choose_motivation_strategy = parse_motivation_strategy(_data)
    number_agents = parse_number_agents(_data)
    competition_max = _data["motivation_parameters"]["competition_max"]
    competition_decay_reward = _data["motivation_parameters"][
        "competition_decay_reward"
    ]
    percent = _data["motivation_parameters"]["percent"]
    logging.info(f"{choose_motivation_strategy = }")
    # =================
    motivation_strategy: mm.MotivationStrategy
    if choose_motivation_strategy == "default":
        motivation_strategy = mm.DefaultMotivationStrategy(width=width, height=height)
    if choose_motivation_strategy == "EVC":
        logging.info(f"init EVC with {width = }, {height = }, {seed = }")
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
            agent_positions=ped_positions,
            motivation_door_center=motivation_door_center,
            competition_decay_reward=competition_decay_reward,
            competition_max=competition_max,
            percent=percent,
            evc=True,
            value_probability=_data["motivation_parameters"][
                "value_probability_sorting"
            ],
            motivation_min=float(mapping_block["motivation_min"]),
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
            agent_positions=ped_positions,
            competition_decay_reward=competition_decay_reward,
            competition_max=competition_max,
            percent=percent,
            motivation_door_center=motivation_door_center,
            evc=False,
            motivation_min=float(mapping_block["motivation_min"]),
        )

    a_ped, d_ped, a_wall, d_wall, a_ped_min, a_ped_max, d_ped_min, d_ped_max = (
        parse_velocity_init_parameters(_data)
    )
    parameter_mapper = mmap.MotivationParameterMapper(
        mapping_block=mapping_block,
        normal_v_0=normal_v_0,
        strength_default=a_ped,
        strength_min=a_ped_min,
        strength_max=a_ped_max,
        range_default=d_ped,
    )
    # =================
    motivation_model = mm.MotivationModel(
        door_point1=(motivation_doors[0][0][0], motivation_doors[0][0][1]),
        door_point2=(motivation_doors[0][1][0], motivation_doors[0][1][1]),
        normal_v_0=normal_v_0,
        normal_time_gap=normal_time_gap,
        motivation_strategy=motivation_strategy,
        parameter_mapper=parameter_mapper,
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
    accessible_areas = parse_accessible_areas(_data)  # TODO not used.

    if from_file:
        logging.info(f"Init geometry from WKT")

        # Original exterior ring
        exterior = [
            (-8.88, -20.1),
            (8.3, -20.1),
            (8.3, 27.95),
            (-8.88, 27.95),
            (-8.88, -20.1),
        ]

        # Interior rings (excluding the door)
        interior_rings = [
            # Left cutout
            [
                (-10, -20),
                (-3.57, -3),
                (-2, -3),
                (-2, -2.8),
                (-3.57, -2.8),
                (-3.57, 19.57),
                (-1.52, 19.57),
                (-1.37, 19.57),
                (-0.87, 19.57),
                (-0.72, 19.57),
                (-0.42, 19.57),
                (-0.42, 21.23),
                (-0.72, 21.23),
                (-0.87, 21.09),
                (-1.37, 21.09),
                (-1.52, 21.23),
                (-1.67, 21.23),
                (-1.67, 21.18),
                (-1.545, 21.18),
                (-1.42, 21.065),
                (-1.42, 19.735),
                (-1.545, 19.62),
                (-3.62, 19.62),
                (-3.59, -3),
                (-10, -20),
            ],
            # Right cutout
            [
                (10, -20),
                (3.57, -3),
                (2, -3),
                (2, -2.8),
                (3.57, -2.8),
                (3.64, 19.64),
                (1.47, 19.57),
                (1.32, 19.57),
                (0.82, 19.57),
                (0.67, 19.57),
                (0.38, 19.57),
                (0.38, 21.23),
                (0.67, 21.23),
                (0.82, 21.09),
                (1.32, 21.09),
                (1.47, 21.23),
                (1.62, 21.23),
                (1.62, 21.18),
                (1.495, 21.18),
                (1.37, 21.065),
                (1.37, 19.735),
                (1.495, 19.62),
                (3.69, 19.69),
                (3.62, -3),
                (10, -20),
            ],
            # Bottom strip
            # [(3.55, -3), (2, -3), (2, -3.2), (3.55, -3.2),(3.55, -3)],
            # [(-3.55, -3), (-2, -3), (-2, -3.2), (-3.55, -3.2),(-3.55, -3)]
        ]

        # Door coordinates
        door = [(-0.41, 20), (0.37, 20), (0.37, 21), (-0.41, 21), (-0.41, 20)]
        door2 = [
            (-0.41, 19.9),
            (0.37, 19.9),
            (0.37, 19.8),
            (-0.41, 19.8),
            (-0.41, 19.9),
        ]

        # Create closed geometry (with door)
        geometry_closed = Polygon(exterior, interior_rings + [door])

        # Create open geometry (without door)
        geometry_open = Polygon(exterior, interior_rings)
    else:
        logging.info("Init geometry from data")
        geometry_open = build_geometry(accessible_areas)
        geometry_closed = build_geometry(accessible_areas)
    # areas = build_areas(destinations, labels)
    simulation = jps.Simulation(
        model=jps.CollisionFreeSpeedModelV2(),
        geometry=geometry_closed,
        dt=_time_step,
        trajectory_writer=jps.SqliteTrajectoryWriter(
            output_file=pathlib.Path(_trajectory_path), every_nth_frame=_fps
        ),
    )
    logging.info("Init simulation done.")
    return simulation, geometry_open


def adjust_radius_with_distance(
    position_y: float,
    motivation_i: float,
    min_value: float = 0.1,
    max_value: float = 0.5,
    y_min: float = -1,  # Start reducing radius from this y position
    y_max: float = 19,  # Exit position where radius should be min_value
    min_motivation: float = 1,
) -> float:
    """
    Adjust the radius based on agent's motivation level and distance to exit.

    :param position_y: The pedestrian's current y position.
    :param motivation_i: The agent's motivation level (1 <= motivation_i <= 3).
    :param min_value: Minimum radius near the exit (y_max).
    :param max_value: Maximum radius when far from the exit.
    :param y_min: Position where radius starts decreasing.
    :param y_max: Position where radius is minimal.
    :return: Adjusted radius.
    """
    max_motivation = 3.6 / 1.2
    # Ensure position_y is within the defined range
    position_y = max(min(position_y, y_max), y_min)

    # Compute distance-based factor (1 when far, 0 when at exit)
    distance_factor = (y_max - position_y) / (y_max - y_min)

    # Compute motivation-dependent base radius (inverse relationship)
    motivation_radius = max_value - (max_value - min_value) * (
        motivation_i - min_motivation
    ) / (max_motivation - min_motivation)
    # print("-----------------------")
    # print(f"{y_min = }, {y_max = }")
    # print(f"{min_value = }, {max_value = }")
    # print(
    #     f"{position_y = }, {motivation_i = }, {min_motivation = }, {motivation_radius = } "
    # )
    # print(
    #     f"{min_value = }, {motivation_radius = }, {distance_factor = }, --> {min_value + (motivation_radius - min_value) * distance_factor}"
    # )
    # input()

    # Scale radius based on distance factor
    return min_value + (motivation_radius - min_value) * distance_factor


def adjust_parameter_linearly(
    motivation_i: float,
    min_value: float = 0.01,
    default_value: float = 0.5,
    max_value: float = 1.0,
    min_motivation: float = 1,
) -> float:
    """
    Adjust the a parameter based on agent's motivation level (1 <= motivation_i <= 3).

    :param motivation_i: The agent's motivation level, expected to be a positive value less than 1.
    :param min_value: Minimum repulsion range for very low motivation.
    :param default_value: Default repulsion range for mid motivation.
    :param max_value: Maximum repulsion range for high motivation.
    :return: Adjusted range_neighbor_repulsion value.
    """
    # Linear interpolation between min_value and max_value based on motivation_i
    return min_value + (max_value - min_value) * 0.5 * (motivation_i - min_motivation)


def adjust_buffer_size_linearly(
    value_i: float,
    min_size: float = 0.1,
    max_size: float = 1.5,
    min_motivation: float = 0.5,
    max_motivation: float = 3.0,
) -> float:
    """
    Adjust the buffer size based on agent's motivation level (1 <= motivation_i <= 3).

    :param motivation_i: The agent's motivation level.
    :param min_size: The minimum buffer size at highest motivation.
    :param max_size: The maximum buffer size at lowest motivation.
    :param min_motivation: Motivation level corresponding to max_size.
    :param max_motivation: Motivation level corresponding to min_size.
    :return: Adjusted buffer size.
    """
    # Clamp motivation_i between min and max motivation
    value_i = max(min_motivation, min(value_i, max_motivation))

    # Inverse linear interpolation
    t = (value_i - min_motivation) / (max_motivation - min_motivation)
    return max_size - (max_size - min_size) * t


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
    _data: Dict[str, Any],
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
    if motivation_model.parameter_mapper is not None:
        motivation_i = motivation_model.parameter_mapper.clamp_motivation(motivation_i)
    agent_value = motivation_model.motivation_strategy.get_value(agent_id=agent.id)
    # if motivation_i > 1:
    #     logging.error(
    #         f"Motivation too high. Count: {simulation.iteration_count()}. Agent: {agent.id}. Motivation: {motivation_i = }"
    #     )

    v_0, time_gap = motivation_model.calculate_motivation_state(motivation_i, agent.id)
    if motivation_model.parameter_mapper is not None:
        agent.model.strength_neighbor_repulsion = (
            motivation_model.parameter_mapper.strength_neighbor_repulsion(motivation_i)
        )
        agent.model.range_neighbor_repulsion = (
            motivation_model.parameter_mapper.range_neighbor_repulsion(motivation_i)
        )
    else:
        min_motivtion = _data["motivation_parameters"]["min_value_low"]
        # Adjust agent parameters based on motivation
        agent.model.strength_neighbor_repulsion = adjust_parameter_linearly(
            motivation_i=motivation_i,
            min_value=a_ped_min,
            default_value=default_strength,
            max_value=a_ped_max,
            min_motivation=min_motivtion,
        )
        agent.model.range_neighbor_repulsion = adjust_parameter_linearly(
            motivation_i=motivation_i,
            min_value=d_ped_min,
            default_value=default_range,
            max_value=d_ped_max,
            min_motivation=min_motivtion,
        )
    # if simulation.elapsed_time() > 25:
    #     agent.model.strength_neighbor_repulsion = 0.6
    #     agent.model.range_neighbor_repulsion = 0.2
    # if simulation.elapsed_time() > 30:
    #     agent.model.strength_neighbor_repulsion = 0.6
    #     agent.model.range_neighbor_repulsion = 0.22
    # if simulation.elapsed_time() > 35:
    #     agent.model.strength_neighbor_repulsion = 0.6
    #     agent.model.range_neighbor_repulsion = 0.24
    # if simulation.elapsed_time() > 40:
    #     agent.model.strength_neighbor_repulsion = 0.6
    #     agent.model.range_neighbor_repulsion = 0.28
    # if simulation.elapsed_time() > 45:
    #     agent.model.strength_neighbor_repulsion = 0.6
    #     agent.model.range_neighbor_repulsion = 0.39

    # else:
    #    agent.model.strength_neighbor_repulsion = 0.6
    #    agent.model.range_neighbor_repulsion = 0.1

    do_adjust_buffer = True
    if "do_adjust_buffer" in _data["motivation_parameters"]:
        do_adjust_buffer = _data["motivation_parameters"]["do_adjust_buffer"]

    if do_adjust_buffer:
        if motivation_model.parameter_mapper is not None:
            agent.model.agent_buffer = motivation_model.parameter_mapper.buffer(
                motivation_i
            )
        else:
            agent.model.agent_buffer = adjust_buffer_size_linearly(motivation_i)
        if False and agent.position[1] > 15:
            print(
                f"{agent.id}: ({agent.position[0]}, {agent.position[1]}), {motivation_i = :.2f}, {agent.model.agent_buffer =}"
            )

    # Usage in the agent model
    do_adjust_radius = False
    if "do_adjust_radius" in _data["motivation_parameters"]:
        do_adjust_radius = _data["motivation_parameters"]["do_adjust_radius"]

    if do_adjust_radius:  # Adjust radius based on distance to exit
        min_value_y = _data["motivation_parameters"]["adjust_radius_y_min"]
        max_value_y = _data["motivation_parameters"]["adjust_radius_y_max"]
        min_value_radius = _data["motivation_parameters"]["min_radius"]
        max_value_radius = _data["motivation_parameters"]["max_radius"]
        min_motivation = _data["motivation_parameters"]["min_value_low"]
        agent.model.radius = adjust_radius_with_distance(
            position_y=position[1],
            motivation_i=motivation_i,
            min_value=min_value_radius,  # Smallest radius near the exit
            max_value=max_value_radius,  # Largest radius when far away
            y_min=min_value_y,  # Start decreasing radius here
            y_max=max_value_y,  # Minimum radius at exit
            min_motivation=min_motivation,
        )
    else:
        agent.model.radius = _data["velocity_init_parameters"][
            "radius"
        ]  # Fixed radius at the exit

        # 0,3
        # 1,6
    agent.model.v0 = v_0
    agent.model.time_gap = time_gap
    # print(
    #     f"{agent.id}, {simulation.elapsed_time():.2f}, {motivation_i:.2f}, {agent.model.v0 =}, {agent.model.time_gap = }"
    # )
    return f"{frame_to_write}, {agent.id}, {simulation.elapsed_time():.2f}, {motivation_i:.2f}, {position[0]:.2f}, {position[1]:.2f}, {agent_value:.2f}"


def run_simulation_loop(
    simulation: jps.Simulation,
    geometry_open,
    door: Point,
    motivation_model: mm.MotivationModel,
    simulation_time: float,
    open_door_time: float,
    a_ped_min: float,
    a_ped_max: float,
    d_ped_min: float,
    d_ped_max: float,
    default_strength: float,
    default_range: float,
    every_nth_frame: int,
    motivation_file: pathlib.Path,
    data: Dict[str, Any],
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
            print(f"Elapsed time: {simulation.elapsed_time():.2f}", end="\r")
            # open the gate after some time
            if simulation.elapsed_time() == open_door_time:
                logging.info(f"Open Door at {open_door_time} s")
                simulation.switch_geometry(geometry_open)

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
                        data,
                    )
                    buffer.append(ret)
                frame_to_write += 1
            simulation.iterate()

        with profile_function("Writing motivation data to csv file"):
            for items in buffer:
                write_value_to_file(file_handle, items)


def create_agent_parameters(
    _data: Dict[str, Any], simulation: jps.Simulation
) -> Tuple[List[jps.CollisionFreeSpeedModelV2AgentParameters], List[List[Point]]]:
    """Create the model parameters."""
    way_points = parse_way_points(_data)
    way_points = []
    destinations_dict = parse_destinations(_data)
    destinations: List[List[Point]] = cast(
        List[List[Point]], list(destinations_dict.values())
    )
    journey_id, exit_ids, wp_ids = init_journey(simulation, way_points, destinations)
    normal_v_0 = parse_normal_v_0(_data)
    normal_time_gap = parse_normal_time_gap(_data)
    radius = parse_radius(_data)
    agent_buffer = _data["motivation_parameters"]["agent_buffer"]
    agent_parameters_list = []
    a_ped, d_ped, a_wall, d_wall, a_ped_min, a_ped_max, d_ped_min, d_ped_max = (
        parse_velocity_init_parameters(_data)
    )

    if not wp_ids:
        stage_id = exit_ids[0]
    else:
        stage_id = wp_ids[0]

    for exit_id in exit_ids:
        agent_parameters = jps.CollisionFreeSpeedModelV2AgentParameters(
            journey_id=journey_id,
            stage_id=stage_id,
            radius=radius,
            v0=normal_v_0,
            time_gap=normal_time_gap,
            strength_neighbor_repulsion=a_ped,
            range_neighbor_repulsion=d_ped,
            strength_geometry_repulsion=a_wall,
            range_geometry_repulsion=d_wall,
            agent_buffer=agent_buffer,
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
            num_positions = len(positions)
            num_agents_config = parse_number_agents(_data)
            logging.info(f"Number of agents from inifile: {num_agents_config = }")
            if num_agents_config < num_positions:
                positions = random.sample(positions, num_agents_config)
                num_agents = num_agents_config
            if num_agents_config >= num_positions:
                positions = positions
                num_agents = num_positions

            logging.info(f"Number of agents from position file: {num_agents = }")
        else:
            raise FileNotFoundError(f"Positions file {positions_file} does not exist!")
    else:
        num_agents = parse_number_agents(_data)
        logging.info(f"Generating {num_agents} agent positions")
        positions = init_positions(_data, num_agents)

    return positions, num_agents


def init_and_run_simulation(
    _fps: int,
    _time_step: float,
    _simulation_time: float,
    _open_door_time: float,
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
    logging.info(f"Motivation file: {motivation_file}")
    simulation, geometry_open = init_simulation(
        _data, _time_step, _fps, _trajectory_path, from_file=True
    )
    a_ped, d_ped, a_wall, d_wall, a_ped_min, a_ped_max, d_ped_min, d_ped_max = (
        parse_velocity_init_parameters(_data)
    )
    agent_parameters_list, exit_positions = create_agent_parameters(_data, simulation)

    positions, _ = get_agent_positions(_data)
    ped_ids = distribute_and_add_agents(
        simulation=simulation,
        agent_parameters_list=agent_parameters_list,
        positions=positions,
        exit_positions=exit_positions,
    )
    motivation_model = init_motivation_model(_data, ped_ids, positions)
    x_door = 0.5 * (motivation_model.door_point1[0] + motivation_model.door_point2[0])
    y_door = 0.5 * (motivation_model.door_point1[1] + motivation_model.door_point2[1])
    motivation_door: Point = (x_door, y_door)
    logging.info(f"Running simulation for {len(ped_ids)} agents:")
    start_time = time.time()
    run_simulation_loop(
        simulation=simulation,
        geometry_open=geometry_open,
        door=motivation_door,
        motivation_model=motivation_model,
        simulation_time=_simulation_time,
        open_door_time=_open_door_time,
        a_ped_min=a_ped_min,
        a_ped_max=a_ped_max,
        d_ped_min=d_ped_min,
        d_ped_max=d_ped_max,
        default_strength=a_ped,
        default_range=d_ped,
        every_nth_frame=_data["simulation_parameters"]["fps"],
        motivation_file=motivation_file,
        data=_data,
    )
    end_time = time.time()
    logging.info(f"Run time: {end_time - start_time:.2f} seconds")
    logging.info(
        f"Simulation completed after {simulation.iteration_count()} iterations"
    )
    logging.info(
        f"Simulation time: {simulation.iteration_count() * _time_step:.2f} [s]"
    )
    # logging.info(f"Trajectory: {_trajectory_path}")
    return float(simulation.iteration_count() * _time_step), motivation_model


def start_simulation(config_path: str, output_path: str) -> float:
    """Call main function."""
    logging.info(f"Start simulation with config file: {config_path}")
    with open(config_path, "r", encoding="utf8") as f:
        data = json.load(f)
        fps = parse_fps(data)
        time_step = parse_time_step(data)
        simulation_time = parse_simulation_time(data)

        if "open_door_time" in data["simulation_parameters"]:
            open_door_time = data["simulation_parameters"]["open_door_time"]
        else:
            open_door_time = 0

        logging.info(f"Open door time: {open_door_time} s")
        dummy = ""
        if fps and time_step:
            evac_time, motivation_model = init_and_run_simulation(
                fps,
                time_step,
                simulation_time,
                open_door_time,
                data,
                pathlib.Path(output_path),
                dummy,
            )
        return evac_time, motivation_model


def load_variations(variations_path: pathlib.Path) -> List[Dict[str, Any]]:
    """Load parameter variations from a JSON file."""
    if not variations_path.exists():
        raise FileNotFoundError(f"Variations file not found: {variations_path}")

    with open(variations_path, "r", encoding="utf8") as f:
        variations = json.load(f)

    # Validate variations format
    for var in variations:
        if "parameters" not in var:
            raise ValueError(f"Missing 'parameters' in variation: {var}")

    return variations


def modify_and_save_config(
    base_config: Dict[str, Any], variation: Dict[str, Any], output_path: pathlib.Path
) -> None:
    """
    Modify base configuration with variation parameters and save to new file.

    Args:
        base_config: Original configuration dictionary
        variation: Dictionary with parameter paths and their new values
        output_path: Where to save the modified configuration
    """
    # Create a deep copy of the base config
    new_config = json.loads(json.dumps(base_config))

    # Apply all parameter changes
    for param_path, value in variation.items():
        keys = param_path.split("/")
        current = new_config
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = value

    # Save the modified configuration
    with open(output_path, "w", encoding="utf8") as f:
        json.dump(new_config, f, indent=4)


def main(
    inifile: pathlib.Path = typer.Option(
        pathlib.Path("files/inifile.json"),
        help="Path to the initial configuration file",
    ),
    variations_file: Optional[pathlib.Path] = typer.Option(
        None,
        help="Path to the variations file (optional). If not provided, the simulation will run with the base configuration only.",
    ),
    output_dir: pathlib.Path = typer.Option(
        pathlib.Path("files/variations"),
        help="Directory for output files",
    ),
) -> None:
    """Run simulations with parameter variations or, if no variations file is provided, run a single simulation using the base configuration."""
    init_logger()

    # Load base configuration
    logging.info(f"Loading base configuration from {inifile}.")
    try:
        with open(inifile, "r", encoding="utf8") as f:
            base_config = json.load(f)
    except FileNotFoundError:
        logging.error(f"Base configuration file not found: {inifile}.")
        raise typer.Exit(code=1)

    # Create output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if variations_file:
        # Load variations from the provided file
        logging.info(f"Loading variations from {variations_file}.")
        try:
            variations = load_variations(variations_file)
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            logging.error(f"Error loading variations: {e}.")
            raise typer.Exit(code=2)

        # Save a copy of the variations used for this run
        run_info = {
            "timestamp": timestamp,
            "base_config": str(inifile),
            "variations_file": str(variations_file),
            "variations": variations,
        }
        run_info_file = output_dir / f"run_info_{timestamp}.json"
        with open(run_info_file, "w") as f:
            json.dump(run_info, f, indent=4)

        # Run simulations for each variation
        results = []
        total_variations = len(variations)

        for i, variation in enumerate(variations, start=1):
            var_name = variation.get("name", f"variation_{i:03d}")
            var_desc = variation.get("description", "")

            logging.info(f"\nRunning variation {i}/{total_variations}: {var_name}")
            if var_desc:
                logging.info(f"Description: {var_desc}")

            # Log parameter changes
            for param, value in variation["parameters"].items():
                original = variation.get("original_value", "unknown")
                logging.info(f"  >>  {param}: {original} -> {value}")

            # Create variation-specific filenames
            new_config_path = output_dir / f"{inifile.stem}_{var_name}.json"
            output_path = output_dir / f"{inifile.stem}_{var_name}.sqlite"

            # Modify and save the new configuration for this variation
            modify_and_save_config(
                base_config, variation["parameters"], new_config_path
            )

            # Run simulation for this variation
            try:
                evac_time = start_simulation(str(new_config_path), str(output_path))
                status = "completed"
            except Exception as e:
                logging.error(f"Error in simulation: {e}.")
                evac_time = None
                status = "failed"

            # Store the result
            result = {
                "variation_name": var_name,
                "description": var_desc,
                "parameters": variation["parameters"],
                "evac_time": evac_time,
                "status": status,
                "config_file": str(new_config_path),
                "output_file": str(output_path),
            }
            results.append(result)

            logging.info(f"Status: {status}.")
            if evac_time is not None:
                logging.info(f"Evacuation time: {evac_time:.2f} [s].")

        # Save all simulation results
        results_file = output_dir / f"results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

        logging.info(f"\nSimulation batch completed. Results saved to {results_file}.")

        completed = sum(1 for r in results if r["status"] == "completed")
        failed = sum(1 for r in results if r["status"] == "failed")
        logging.info("\nSummary:")
        logging.info(f"Total variations: {total_variations}")
        logging.info(f"Completed: {completed}")
        logging.info(f"Failed: {failed}")

    else:
        # No variations file provided; run simulation with the base configuration only.
        logging.info(
            "No variations file provided. Running simulation with base configuration."
        )

        # Optionally, you can save a copy of the base configuration in the output directory.
        config_file = output_dir / f"{inifile.stem}_base_{timestamp}.json"
        with open(config_file, "w") as f:
            json.dump(base_config, f, indent=4)

        output_path = output_dir / f"{inifile.stem}_base_{timestamp}.sqlite"

        try:
            evac_time, motivation_model = start_simulation(
                str(config_file), str(output_path)
            )
            status = "completed"
        except Exception as e:
            logging.error(f"Error in simulation: {e}.")
            evac_time = None
            status = "failed"

        # Save run info for the base configuration simulation
        run_info = {
            "timestamp": timestamp,
            "base_config": str(inifile),
            "config_file": str(config_file),
            "output_file": str(output_path),
            "status": status,
            "evac_time": evac_time,
        }
        run_info_file = output_dir / f"run_info_{timestamp}.json"
        with open(run_info_file, "w") as f:
            json.dump(run_info, f, indent=4)

        logging.info(f"Status: {status}.")
        if evac_time is not None:
            logging.info(f"Evacuation time: {evac_time:.2f} [s].")
        logging.info(
            f"\nSimulation completed. Run info saved to {run_info_file}.\n Timestamp:"
        )
        print(timestamp)

        if status == "completed":
            logging.info("JPSVIS")
            trajectory_data, walkable_area = read_sqlite_file(output_path)
            output_file = "jpsvis_files" + pathlib.Path(output_path).stem + ".txt"
            geometry_file = pathlib.Path(output_path).stem + "_geometry.xml"
            v0_mean = 1.2
            export_trajectory_to_txt(
                trajectory_data,
                motivation_model,
                output_file=output_file,
                geometry_file="geometry.xml",
                df=10,
                v0=v0_mean,
                by_speed=True,
            )

            # polygon_to_xml(walkable_area=walkable_area, output_file=geometry_file)
            print(">>> ", output_file)
            print(">>> ", geometry_file)
            command = ["/Applications/jpsvis.app/Contents/MacOS/jpsvis", output_file]
            result = subprocess.run(command, capture_output=True, text=True)

            (
                df_merged_simulation,
                density_over_time_simulation,
                crossing_info_simulation,
                individual_simulation,
                title_simulation,
            ) = calculate_crossing_density(
                output_path, walkable_area, file_type="simulation", title="Simulation"
            )

            plot_crossing_order_vs_area(
                df_merged_simulation, output_path.stem, title=None, color="blue"
            )

        else:
            logging.warning(f"Status: {status}")


if __name__ == "__main__":
    typer.run(main)
