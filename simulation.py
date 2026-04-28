"""Simulation model using jpscore API."""

# Copyright © 2012-2022 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: LGPL-3.0-or-later

import _io
import contextlib
import csv
import json
import logging
import pathlib
import random
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple, TypeAlias, cast
from xml.etree.ElementTree import Element, ElementTree, SubElement

import jupedsim as jps
import numpy as np
import pandas as pd
import typer
from jupedsim.distributions import distribute_by_number
from jupedsim.internal.notebook_utils import read_sqlite_file
from shapely import Polygon

from src import motivation_mapping as mmap
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
    parse_payoff_update_interval,
    parse_radius,
    parse_simulation_time,
    parse_theta_max_upper_bound,
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
MOTIVATION_CSV_HEADER = (
    "frame,id,time,motivation,x,y,value,rank_abs,rank_q,payoff_p,rank_update_flag"
)


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
    trajectory_data: Any,
    motivation_model: mm.MotivationModel,
    output_file: str = "output.txt",
    geometry_file: str = "geometry.xml",
    motivation_csv: pathlib.Path | None = None,
    df: int = 10,
    v0: float = 1.2,
    radius: float = 0.18,
    by_speed: bool = True,
) -> None:
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

        # Calculate speed for this trajectory
        speed = compute_speed(traj, df, fps)
        values.extend([0.0] * len(speed))

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
        if motivation_csv is not None and motivation_csv.exists():
            motivation_df = pd.read_csv(motivation_csv)
            if not ("frame" in motivation_df.columns and "id" in motivation_df.columns):
                motivation_df = pd.read_csv(
                    motivation_csv,
                    names=[
                        "frame",
                        "id",
                        "time",
                        "motivation",
                        "x",
                        "y",
                        "value",
                        "rank_abs",
                        "rank_q",
                        "payoff_p",
                        "rank_update_flag",
                    ],
                )
            motivation_slice = motivation_df[["frame", "id", "motivation"]].copy()
            df_data = df_data.merge(
                motivation_slice,
                on=["frame", "id"],
                how="left",
                suffixes=("", "_m"),
            )
            if "motivation_m" in df_data.columns:
                df_data["motivation"] = df_data["motivation"].fillna(
                    df_data["motivation_m"]
                )
                df_data.drop(columns=["motivation_m"], inplace=True)
            df_data["motivation"] = df_data["motivation"].fillna(0.0)
        else:
            value_series = pd.Series(values, index=frame_indices)
            df_data.loc[frame_indices, "motivation"] = value_series

        motivation_mode = str(
            motivation_model.motivation_strategy.motivation_mode
        ).upper()
        if motivation_mode == "BASE_MODEL":
            df_data["color"] = 128
        else:
            if motivation_mode == "V":
                motivation_min = float(df_data["motivation"].min())
                motivation_max = float(df_data["motivation"].max())
            elif motivation_mode == "P":
                motivation_min = 0.0
                motivation_max = 1.0
            else:
                motivation_min = float(
                    motivation_model.motivation_strategy.motivation_min
                )
                motivation_max = float(mmap.MOTIVATION_HIGH)
            motivation_range = max(motivation_max - motivation_min, 1e-9)
            normalized_motivation = (
                (df_data["motivation"] - motivation_min) / motivation_range
            ).clip(0, 1)
            df_data["color"] = ((1 - normalized_motivation) * 255).astype(int)
        df_data["id"] = df_data["id"].astype(int)
        df_data["frame"] = df_data["frame"].astype(int)
        df_data["color"] = df_data["color"].astype(int)

    # Write the formatted data to the output file
    geometry_reference = pathlib.Path(geometry_file).name
    with open(output_file, "w") as f:
        # Write the header
        f.write(f"#framerate: {fps}\n")
        f.write("#unit: m\n")
        f.write(f"#geometry: {geometry_reference}\n")
        f.write("#ID\tFR\tX\tY\tZ\tA\tB\tANGLE\tCOLOR\n")

        # Write each row of trajectory data
        for row in df_data.itertuples(index=False):
            f.write(
                f"{row.id}\t{row.frame}\t{row.x:.4f}\t{row.y:.4f}\t0\t"
                f"{radius:.4f}\t{radius:.4f}\t{row.angle:.4f}\t{row.color}\n"
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
    time_step = parse_time_step(_data)
    mapping_block = mmap.ensure_mapping_block(_data["motivation_parameters"])
    motivation_mode = parse_motivation_strategy(_data)
    number_agents = parse_number_agents(_data)
    payoff_params = _data["motivation_parameters"]["payoff"]
    payoff_update_interval_s = parse_payoff_update_interval(_data)
    logging.info(f"{motivation_mode = }")
    # =================
    motivation_strategy: mm.EVPStrategy = mm.EVPStrategy(
        width=width,
        height=height,
        max_reward=number_agents,
        seed=seed,
        max_value=float(_data["motivation_parameters"]["max_value"]),
        min_value=float(_data["motivation_parameters"]["min_value"]),
        nagents=number_agents,
        agent_ids=ped_ids,
        agent_positions=ped_positions,
        motivation_door_center=motivation_door_center,
        motivation_min=float(mapping_block["motivation_min"]),
        motivation_mode=motivation_mode,
        payoff_k=float(payoff_params["k"]),
        payoff_q0=float(payoff_params["q0"]),
        rank_tie_tolerance_m=float(payoff_params["rank_tie_tolerance_m"]),
        payoff_update_interval_s=payoff_update_interval_s,
    )
    motivation_strategy.configure_payoff_update_interval(time_step=time_step)

    parameter_mapper = None
    if motivation_mode != "BASE_MODEL":
        _, d_ped, _, _ = parse_velocity_init_parameters(_data)
        parameter_mapper = mmap.MotivationParameterMapper(
            mapping_block=mapping_block,
            normal_v_0=normal_v_0,
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
        logging.info("Init geometry from WKT")

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
                # (-2, -3),
                # (-2, -2.8),
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
                # (2, -3),
                # (2, -2.8),
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
            # [(3.55, -3), (2, -3), (2, -3.2), (3.55, -3.2), (3.55, -3)],
            # [(-3.55, -3), (-2, -3), (-2, -3.2), (-3.55, -3.2), (-3.55, -3)],
        ]

        # Door coordinates
        door = [(-0.41, 20), (0.37, 20), (0.37, 21), (-0.41, 21), (-0.41, 20)]
        #  door2 = [
        #     (-0.41, 19.9),
        #     (0.37, 19.9),
        #     (0.37, 19.8),
        #     (-0.41, 19.8),
        #     (-0.41, 19.9),
        # ]

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
        model=jps.CollisionFreeSpeedModelV3(),
        geometry=geometry_closed,
        dt=_time_step,
        trajectory_writer=jps.SqliteTrajectoryWriter(
            output_file=pathlib.Path(_trajectory_path), every_nth_frame=_fps
        ),
    )
    logging.info("Init simulation done.")
    return simulation, geometry_open


def process_agent(
    agent: jps.Agent,
    door: Point,
    simulation: jps.Simulation,
    motivation_model: mm.MotivationModel,
    file_handle: _io.TextIOWrapper,
    frame_to_write: int,
    rank_update_flag: int,
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
    rank_abs, rank_q, payoff_p = motivation_model.motivation_strategy.get_rank_payoff(
        agent.id
    )
    # if motivation_i > 1:
    #     logging.error(
    #         f"Motivation too high. Count: {simulation.iteration_count()}. Agent: {agent.id}. Motivation: {motivation_i = }"
    #     )

    if motivation_model.parameter_mapper is not None:
        v_0, time_gap = motivation_model.calculate_motivation_state(
            motivation_i, agent.id
        )
        agent.model.strength_neighbor_repulsion = (
            motivation_model.parameter_mapper.strength_neighbor_repulsion(motivation_i)
        )
        agent.model.range_neighbor_repulsion = (
            motivation_model.parameter_mapper.range_neighbor_repulsion(motivation_i)
        )

    if "do_adjust_buffer" in _data["motivation_parameters"]:
        do_adjust_buffer = _data["motivation_parameters"]["do_adjust_buffer"]

    if do_adjust_buffer and motivation_model.parameter_mapper is not None:
        agent.model.agent_buffer = motivation_model.parameter_mapper.buffer(
            motivation_i
        )

    agent.model.radius = _data["velocity_init_parameters"]["radius"]
    if motivation_model.parameter_mapper is not None:
        agent.model.desired_speed = v_0
        agent.model.time_gap = time_gap
    return (
        f"{frame_to_write}, {agent.id}, {simulation.elapsed_time():.2f}, "
        f"{motivation_i:.4f}, {position[0]:.4f}, {position[1]:.4f}, "
        f"{agent_value:.4f}, {rank_abs}, {rank_q:.6f}, {payoff_p:.6f}, {rank_update_flag}"
    )


def run_simulation_loop(
    simulation: jps.Simulation,
    geometry_open: Any,
    door: Point,
    motivation_model: mm.MotivationModel,
    simulation_time: float,
    open_door_time: float,
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
        every_nth_frame (int): Write to file every nth frame.
        motivation_file (pathlib.Path): Path to the motivation file to write.

    Returns:
        None
    """
    buffer = []

    with open(motivation_file, "w", encoding="utf-8") as file_handle:
        file_handle.write(MOTIVATION_CSV_HEADER + "\n")
        frame_to_write = 0

        while (
            simulation.elapsed_time() < simulation_time and simulation.agent_count() > 0
        ):
            print(f"Elapsed time: {simulation.elapsed_time():.2f}", end="\r")
            # open the gate after some time
            if simulation.elapsed_time() == open_door_time:
                logging.info(f"Open Door at {open_door_time} s")
                simulation.switch_geometry(geometry_open)

            active_positions = {
                agent.id: cast(Point, agent.position) for agent in simulation.agents()
            }
            rank_updated = motivation_model.motivation_strategy.update_payoff_cache(
                iteration_count=simulation.iteration_count(),
                agent_positions=active_positions,
                number_agents_in_simulation=simulation.agent_count(),
            )
            rank_update_flag = 1 if rank_updated else 0

            if simulation.iteration_count() % every_nth_frame == 0:
                for agent in simulation.agents():
                    ret = process_agent(
                        agent,
                        door,
                        simulation,
                        motivation_model,
                        file_handle,
                        frame_to_write,
                        rank_update_flag,
                        data,
                    )
                    buffer.append(ret)
                frame_to_write += 1
            simulation.iterate()

        logging.info(f">>> Agents still in simulation: {simulation.agent_count()}")
        with profile_function("Writing motivation data to csv file"):
            for items in buffer:
                write_value_to_file(file_handle, items)


def create_agent_parameters(
    _data: Dict[str, Any], simulation: jps.Simulation
) -> Tuple[List[jps.CollisionFreeSpeedModelV3AgentParameters], List[List[Point]]]:
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
    theta_max_upper_bound = parse_theta_max_upper_bound(_data)
    agent_parameters_list = []
    a_ped, d_ped, a_wall, d_wall = parse_velocity_init_parameters(_data)

    if not wp_ids:
        stage_id = exit_ids[0]
    else:
        stage_id = wp_ids[0]

    for exit_id in exit_ids:
        agent_parameters = jps.CollisionFreeSpeedModelV3AgentParameters(
            journey_id=journey_id,
            stage_id=stage_id,
            radius=radius,
            desired_speed=normal_v_0,
            time_gap=normal_time_gap,
            strength_neighbor_repulsion=a_ped,
            range_neighbor_repulsion=d_ped,
            strength_geometry_repulsion=a_wall,
            range_geometry_repulsion=d_wall,
            agent_buffer=agent_buffer,
        )
        if hasattr(agent_parameters, "theta_max_upper_bound"):
            setattr(agent_parameters, "theta_max_upper_bound", theta_max_upper_bound)
        elif hasattr(agent_parameters, "thetaMaxUpperBound"):
            setattr(agent_parameters, "thetaMaxUpperBound", theta_max_upper_bound)
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
) -> Tuple[float, mm.MotivationModel]:
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


def start_simulation(
    config_path: str, output_path: str
) -> Tuple[float, mm.MotivationModel]:
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
        pathlib.Path("files/base.json"),
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
    vis: bool = typer.Option(
        False,
        "--vis",
        help="Launch jpsvis after a completed single simulation run.",
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
            var_evac_time: float | None = None
            try:
                var_evac_time, _ = start_simulation(
                    str(new_config_path), str(output_path)
                )
                status = "completed"
            except Exception as e:
                logging.error(f"Error in simulation: {e}.")
                status = "failed"

            # Store the result
            result = {
                "variation_name": var_name,
                "description": var_desc,
                "parameters": variation["parameters"],
                "evac_time": var_evac_time,
                "status": status,
                "config_file": str(new_config_path),
                "output_file": str(output_path),
            }
            results.append(result)

            logging.info(f"Status: {status}.")
            if var_evac_time is not None:
                logging.info(f"Evacuation time: {var_evac_time:.2f} [s].")

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
        motivation_mode = str(base_config["motivation_parameters"]["motivation_mode"])

        # Optionally, you can save a copy of the base configuration in the output directory.
        config_file = output_dir / f"{inifile.stem}_{motivation_mode}_{timestamp}.json"
        with open(config_file, "w") as f:
            json.dump(base_config, f, indent=4)

        output_path = (
            output_dir / f"{inifile.stem}_{motivation_mode}_{timestamp}.sqlite"
        )

        evac_time: float | None = None
        motivation_model_result: mm.MotivationModel | None = None
        try:
            evac_time, motivation_model_result = start_simulation(
                str(config_file), str(output_path)
            )
            status = "completed"
        except Exception as e:
            logging.error(f"Error in simulation: {e}.")
            status = "failed"

        # Save run info for the base configuration simulation
        base_run_info: Dict[str, Any] = {
            "timestamp": timestamp,
            "base_config": str(inifile),
            "config_file": str(config_file),
            "output_file": str(output_path),
            "status": status,
            "evac_time": evac_time,
        }
        run_info_file = output_dir / f"run_info_{timestamp}.json"
        with open(run_info_file, "w") as f:
            json.dump(base_run_info, f, indent=4)

        logging.info(f"Status: {status}.")
        if evac_time is not None:
            logging.info(f"Evacuation time: {evac_time:.2f} [s].")
        logging.info(
            f"\nSimulation completed. Run info saved to {run_info_file}.\n Timestamp:"
        )
        print(timestamp)

        if status == "completed":
            trajectory_data, walkable_area = read_sqlite_file(output_path)
            scenario_name = (
                output_path.parent.parent.name
                if output_path.parent.name == "base_runs"
                else output_path.parent.name
            )
            output_file = output_path.with_name(
                f"{scenario_name}_{output_path.stem}.txt"
            )
            geometry_file = "geometry.xml"
            motivation_csv = output_path.with_name(output_path.stem + "_motivation.csv")
            logging.info(f"Using:  {geometry_file} ")
            v0_mean = 1.2
            assert motivation_model_result is not None
            export_trajectory_to_txt(
                trajectory_data,
                motivation_model_result,
                output_file=str(output_file),
                geometry_file=geometry_file,
                motivation_csv=motivation_csv,
                df=10,
                v0=v0_mean,
                by_speed=False,
            )

            polygon_to_xml(walkable_area=walkable_area, output_file=geometry_file)
            print(">>> ", output_file)
            print(">>> ", geometry_file)
            if vis:
                logging.info("Launching JPSVIS")
                command = [
                    "/Applications/jpsvis.app/Contents/MacOS/jpsvis",
                    str(output_file),
                ]
                subprocess.run(command, capture_output=True, text=True)

        else:
            logging.warning(f"Status: {status}")


if __name__ == "__main__":
    typer.run(main)
