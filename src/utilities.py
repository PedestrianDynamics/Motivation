"""Some functions to setup the simulation."""

import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, TypeAlias

import jupedsim as jps
import numpy as np
import numpy.typing as npt
import pedpy
import plotly.graph_objects as go
import streamlit as st
from plotly.graph_objs import Figure
from scipy import stats
from shapely import GeometryCollection, Polygon
from shapely.ops import unary_union

from .logger_config import log_error, log_info

Point: TypeAlias = Tuple[float, float]


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
            log_error(f"Error deleting {file}: {e}")


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
    wp_ids = []
    journey = jps.JourneyDescription()
    distance = 1
    for way_point in way_points:
        # log_info(f"add way_point: {way_point}")
        wp_id = simulation.add_waypoint_stage((way_point[0], way_point[1]), distance)
        wp_ids.append(wp_id)
        journey.add(wp_id)

    exit_id = simulation.add_exit_stage(exits)
    exit_ids.append(exit_id)
    journey.add(exit_id)

    # todo: using only one exit here
    stage_id = exit_ids[0]
    for wp_id in wp_ids:
        journey.set_transition_for_stage(
            wp_id, jps.Transition.create_fixed_transition(stage_id)
        )

    journey_id = int(simulation.add_journey(journey))
    return journey_id, stage_id


def distribute_and_add_agents(
    simulation: jps.Simulation,
    agent_parameters: jps.CollisionFreeSpeedModelAgentParameters,
    positions: List[Point],
) -> List[int]:
    """Initialize positions of agents and insert them into the simulation.

    :param simulation:
    :param agent_parameters:
    :param positions:
    :returns:

    """
    # log_info("Distribute and Add Agent")
    ped_ids = []
    for pos_x, pos_y in positions:
        agent_parameters.position = (pos_x, pos_y)
        ped_id = simulation.add_agent(agent_parameters)
        ped_ids.append(ped_id)

    return ped_ids


def create_empty_figure() -> Figure:
    """
    Create an empty Plotly figure.

    Returns:
        go.Figure: An empty Plotly figure.
    """
    return go.Figure(go.Scatter(x=[], y=[], mode="markers", marker={"size": 0}))


def update_figure_layout(fig: Figure, walkable_area: pedpy.WalkableArea) -> None:
    """
    Update the layout of the Plotly figure based on polygon boundaries.

    Args:
        fig (go.Figure): The Plotly figure.
        polygons (dict): Dictionary of polygons representing accessible areas.
    """
    geo_min_x, geo_min_y, geo_max_x, geo_max_y = walkable_area.bounds
    eps = 0
    fig.update_xaxes(range=[geo_min_x - eps, geo_max_x + eps])
    fig.update_yaxes(range=[geo_min_y - eps, geo_max_y + eps])


def calculate_heatmap_values(
    position_x: npt.NDArray[Any],
    position_y: npt.NDArray[Any],
    value: npt.NDArray[Any],
    walkable_area: pedpy.WalkableArea,
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """
    Calculate heatmap values using statistical binning.

    Args:
        position_x: Array of X positions.
        position_y: Array of Y positions.
        value: Array of values associated with positions.
        walkable_area

    """
    geo_min_x, geo_max_x, geo_min_y, geo_max_y = walkable_area.bounds
    delta_x = st.slider(label="grid size", min_value=0.1, max_value=1.0, step=0.1)
    delta_y = delta_x
    xbins = np.arange(geo_min_x, geo_max_x + delta_x, delta_x)
    ybins = np.arange(geo_min_y, geo_max_y + delta_y, delta_y)
    area = delta_x * delta_y
    ret = stats.binned_statistic_2d(
        position_x,
        position_y,
        value,
        "mean",
        bins=[xbins, ybins],
    )
    heatmap_values = np.nan_to_num(ret.statistic.T) / area
    return heatmap_values, xbins, ybins


def add_heatmap_trace(
    fig: Figure,
    xbins: npt.NDArray[Any],
    ybins: npt.NDArray[Any],
    heatmap_values: npt.NDArray[Any],
) -> None:
    """
    Add a heatmap trace to the Plotly figure.

    Args:
        fig (go.Figure): The Plotly figure.
        xbins (np.ndarray): Binning edges for X coordinates.
        ybins (np.ndarray): Binning edges for Y coordinates.
        heatmap_values (np.ndarray): Calculated heatmap values.
    """
    fig.add_trace(
        go.Heatmap(
            x=xbins,
            y=ybins,
            z=heatmap_values,
            zmin=0,
            zmax=0.5,
            connectgaps=False,
            zsmooth="best",
            colorscale="Jet",
            colorbar={"title": "Motivation"},
        )
    )


def add_polygon_traces(fig: Figure, walkable_area: pedpy.WalkableArea) -> None:
    """
    Add polygon traces to the Plotly figure.

    Args:
        fig (go.Figure): The Plotly figure.
        polygons (dict): Dictionary of polygons representing accessible areas.
    """
    x, y = walkable_area.polygon.exterior.xy
    x = np.array(x)
    y = np.array(y)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line={"color": "white"},
        )
    )


def customize_fig_layout(fig: Figure) -> None:
    """
    Customize the layout of the Plotly figure.

    Args:
        fig (go.Figure): The Plotly figure.
    """
    fig.update_layout(title="Heatmap", showlegend=False)


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
