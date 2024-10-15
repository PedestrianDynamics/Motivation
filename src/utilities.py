"""Some functions to setup the simulation."""

import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, TypeAlias, Union
from types import SimpleNamespace
import logging
import jupedsim as jps

import streamlit as st
from shapely import GeometryCollection, Polygon
from shapely.ops import unary_union
from math import sqrt

Point: TypeAlias = Tuple[float, float]


def parse(data: Union[List, Dict, Any]) -> Union[List, SimpleNamespace, Any]:
    """
    Recursively converts a nested structure of lists and dictionaries.

    into a structure of lists and SimpleNamespace objects. Other data types are left unchanged.

    Parameters:
    - data (Union[List, Dict, Any]): The input data to parse. This can be a list,
      dictionary, or any other data type. If it's a list or dictionary, the function
      will recursively parse its content.

    Returns:
    - Union[List, SimpleNamespace, Any]: The parsed data where dictionaries are
      converted to SimpleNamespace objects, lists are recursively parsed, and
      other data types are returned unchanged.

    """
    if isinstance(data, list):
        return list(map(parse, data))
    elif isinstance(data, dict):
        sns = SimpleNamespace()
        for key, value in data.items():
            setattr(sns, key, parse(value))
        return sns
    else:
        return data


def delete_txt_files() -> None:
    """Delete all *.sqlite files in the current directory."""
    files = glob.glob("files/*.sqlite")
    if not files:
        st.toast("No trajectories to delete!", icon="💿")
    for file in files:
        st.toast(f"Delete {file}", icon="💿")
        try:
            os.remove(file)
        except Exception as e:
            logging.error(f"Error deleting {file}: {e}")


def build_geometry(
    accessible_areas: Dict[int, List[List[float]]],
) -> GeometryCollection:
    """Build geometry object.

    All points should be defined CCW
    :returns: a geometry builder

    """
    # log_info("Build geometry")
    polygons = []
    for accessible_area in accessible_areas.values():
        # log_info(f"> {accessible_area=}")
        polygons.append(Polygon(accessible_area))

    # Combine polygons into a single geometry
    combined_area = GeometryCollection(unary_union(polygons))
    return combined_area


def init_journey(
    simulation: jps.Simulation,
    way_points: List[Tuple[Point, float]],
    exits: List[List[Point]],
) -> Tuple[int, int]:
    """Init goals of agents to follow.

    Add waypoints and exits to journey. Then register journey in simultion

    :param simulation:
    :param way_points: defined as a list of (point, distance)
    :returns: journey id and stage id

    """
    # log_info("Init journey with: ")
    # log_info(f"{ way_points= }")
    # log_info(f"{ exits= }")
    exit_ids = []
    # wp_ids = []
    journey = jps.JourneyDescription()
    # distance = 1
    # for way_point in way_points:
    #     # log_info(f"add way_point: {way_point}")
    #     wp_id = simulation.add_waypoint_stage((way_point[0], way_point[1]), distance)
    #     wp_ids.append(wp_id)
    #     journey.add(wp_id)

    for e in exits:
        # log_info(f"add {e}")
        exit_id = simulation.add_exit_stage(e)
        exit_ids.append(exit_id)
        journey.add(exit_id)

    # chosen_id = random.choice(exit_ids)
    # logging.info(f"{chosen_id}, {exit_ids}")
    # stage_id = chosen_id
    # # todo these wp id are not needed and not properly initialized
    # for wp_id in wp_ids:
    #     journey.set_transition_for_stage(
    #         wp_id, jps.Transition.create_fixed_transition(stage_id)
    #     )

    journey_id = int(simulation.add_journey(journey))
    return journey_id, exit_ids


def calculate_distance(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between two points."""
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_centroid(points: List[Point]) -> Point:
    """Calculate the centroid of a polygon (list of points)."""
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    centroid_x = sum(x_coords) / len(points)
    centroid_y = sum(y_coords) / len(points)
    return (centroid_x, centroid_y)


def distribute_and_add_agents(
    simulation: jps.Simulation,
    agent_parameters_list: List[jps.CollisionFreeSpeedModelV2AgentParameters],
    positions: List[Point],
    exit_positions: List[List[Point]],
) -> List[int]:
    """Initialize positions of agents, assign each one to the nearest exit (based on centroid).

    and insert them into the simulation.

    :param simulation: The simulation object.
    :param agent_parameters_list: List of agent parameters to be used for each agent.
    :param positions: List of initial positions for agents.
    :param exit_positions: List of positions for each exit (as polygons).
    :returns: List of pedestrian IDs after being added to the simulation.
    """
    ped_ids = []
    exit_centroids = [calculate_centroid(exit_points) for exit_points in exit_positions]

    for i, (pos_x, pos_y) in enumerate(positions):
        agent_position = (pos_x, pos_y)
        nearest_exit_index = min(
            range(len(exit_centroids)),
            key=lambda j: calculate_distance(agent_position, exit_centroids[j]),
        )
        agent_parameters = agent_parameters_list[nearest_exit_index]
        agent_parameters.position = (pos_x, pos_y)
        ped_id = simulation.add_agent(agent_parameters)
        ped_ids.append(ped_id)

    return ped_ids


def load_json(filename: Path) -> Any:
    """Load json file."""
    try:
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except json.JSONDecodeError as error_1:
        st.error(f"Error loading JSON file: {error_1}")
        return {}


def save_json(output: Path, data: Dict[str, Any]) -> None:
    """Save data in json file."""
    with open(output, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)
