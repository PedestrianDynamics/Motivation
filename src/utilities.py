"""Some functions to setup the simulation."""

import glob
import os
from typing import Dict, List, Tuple, TypeAlias

import jupedsim as jps
from jupedsim.util import build_jps_geometry
from shapely import GeometryCollection, Polygon
from shapely.ops import unary_union

from .logger_config import log_info, log_error

Point: TypeAlias = Tuple[float, float]


def delete_txt_files():
    """Delete all *.txt files in the current directory."""
    files = glob.glob("*.txt")

    for file in files:
        try:
            os.remove(file)
        except Exception as e:
            log_error(f"Error deleting {file}: {e}")


def build_geometry(
    accessible_areas: Dict[int, List[List[float]]],
) -> jps.GeometryBuilder:
    """Build geometry object.

    All points should be defined CCW
    :returns: a geometry builder

    """
    log_info("Build geometry")
    polygons = []
    for accessible_area in accessible_areas.values():
        log_info(f"> {accessible_area=}")
        polygons.append(Polygon(accessible_area))

    # Combine polygons into a single geometry
    combined_area = GeometryCollection(unary_union(polygons))
    return build_jps_geometry(combined_area)


def build_velocity_model(
    init_parameters: Dict[str, float],
) -> jps.OperationalModel:
    """Initialize velocity model with parameter values.

    :param a_ped:
    :param d_ped:
    :param a_wall:
    :param d_wall:
    :returns: velocity model

    """
    a_ped = init_parameters["a_ped"]
    d_ped = init_parameters["d_ped"]
    a_wall = init_parameters["a_wall"]
    d_wall = init_parameters["d_wall"]
    model_builder = jps.VelocityModelBuilder(
        a_ped=a_ped, d_ped=d_ped, a_wall=a_wall, d_wall=d_wall
    )
    model = model_builder.build()
    return model


def init_journey(
    simulation: jps.Simulation,
    way_points: List[Tuple[Point, float]],
    exits: List[List[Point]],
) -> int:
    """Init goals of agents to follow.

    Add waypoints and exits to journey. Then register journey in simultion

    :param simulation:
    :param way_points: defined as a list of (point, distance)
    :returns:

    """
    log_info("Init journey with: ")
    log_info(f"{way_points=}")
    log_info(f"{exits=}")
    journey = jps.JourneyDescription()
    for way_point in way_points:
        log_info(f"add way_point: {way_point}")
        journey.add_waypoint(way_point[0], way_point[1])

    journey.add_exit(exits)

    journey_id = int(simulation.add_journey(journey))
    return journey_id


def init_velocity_agent_parameters(
    phi_x: float,
    phi_y: float,
    journey: int,
) -> jps.VelocityModelAgentParameters:
    """Init agent shape and parameters.

    :param radius: radius of the circle
    :param phi_x: direcion in x-axis
    :param phi_y: direction in y-axis
    :param journey: waypoints for agents to pass through
    :returns:

    """
    log_info("Create agents")
    agent_parameters = jps.VelocityModelAgentParameters()

    agent_parameters.journey_id = journey
    agent_parameters.orientation = (phi_x, phi_y)
    agent_parameters.position = (0.0, 0.0)
    return agent_parameters


def distribute_and_add_agents(
    simulation: jps.Simulation,
    agent_parameters: jps.VelocityModelAgentParameters,
    positions: List[Point],
) -> List[int]:
    """Initialize positions of agents and insert them into the simulation.

    :param simulation:
    :param agent_parameters:
    :param positions:
    :returns:

    """
    log_info("Distribute and Add Agent")
    ped_ids = []
    for pos_x, pos_y in positions:
        agent_parameters.position = (pos_x, pos_y)
        ped_id = simulation.add_agent(agent_parameters)
        ped_ids.append(ped_id)

    return ped_ids
