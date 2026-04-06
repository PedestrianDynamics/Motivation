"""Some functions to setup the simulation."""

import glob
import json
import logging
import os
import random
from math import sqrt
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, TypeAlias, Union

import jupedsim as jps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pedpy
import streamlit as st
from pedpy import Cutoff, compute_individual_voronoi_polygons
from shapely import GeometryCollection, Polygon
from shapely.ops import unary_union

Point: TypeAlias = Tuple[float, float]


def parse(
    data: Union[List[Any], Dict[str, Any], Any],
) -> Union[List[Any], SimpleNamespace, Any]:
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
) -> Tuple[int, List[int], List[int]]:
    """Init goals of agents to follow.

    Add waypoints and exits to journey. Then register journey in simulation

    :param simulation:
    :param way_points: defined as a list of (point, distance)
    :returns: journey id and stage id

    """
    # log_info("Init journey with: ")
    # log_info(f"{ way_points= }")
    # log_info(f"{ exits= }")
    exit_ids: List[int] = []
    wp_ids = []
    journey = jps.JourneyDescription()
    for way_point, distance in way_points:
        logging.info(f"add way_point: {way_point}, distance: {distance}")
        wp_id = simulation.add_waypoint_stage((way_point[0], way_point[1]), distance)
        wp_ids.append(wp_id)
        journey.add(wp_id)

    for e in exits:
        # log_info(f"add {e}")
        exit_id = simulation.add_exit_stage(e)
        exit_ids.append(exit_id)
        journey.add(exit_id)

    chosen_id = random.choice(exit_ids)
    logging.info(f"{chosen_id}, {exit_ids}")
    stage_id = chosen_id
    for wp_id in wp_ids:
        journey.set_transition_for_stage(
            wp_id, jps.Transition.create_fixed_transition(stage_id)
        )

    journey_id = int(simulation.add_journey(journey))
    return journey_id, exit_ids, wp_ids


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


def calculate_crossing_density(
    filename,
    walkable_area,
    fps=50,
    polygon_cutoff_radius=1,
    polygon_quad_segments=3,
    title=None,
    file_type="experiment",
):
    if file_type == "experiment":
        if not title:
            title = filename.split("_")[1].capitalize()
        df = pd.read_csv(
            filename, sep="\t", names=["id", "frame", "x", "y", "z", "m"], comment="#"
        )
        traj = pedpy.TrajectoryData(df, frame_rate=fps)
        print(filename, traj.frame_rate)
    elif file_type == "simulation":
        traj = pedpy.load_trajectory_from_jupedsim_sqlite(filename)
        walkable_area = pedpy.load_walkable_area_from_jupedsim_sqlite(filename)
        df = traj.data
        print(
            f"Calculate crossing density for simulation file: {filename = }, {traj.frame_rate = }"
        )

    print(f"Processing file: {title}")

    # --- Crossing Information Calculation ---
    # Filter rows where pedestrians have crossed (e.g., y >= 20)
    df_crossed = df[df["y"] >= 20].copy()

    # For each pedestrian (id), get the first frame where they cross.
    crossing_frames = df_crossed.groupby("id")["frame"].min().rename("crossing_frame")

    # Sort crossing frames to determine the order.
    crossing_frames_sorted = crossing_frames.sort_values()

    # Create a series that indicates crossing order (1 for first, 2 for second, etc.)
    crossing_order = pd.Series(
        range(1, len(crossing_frames_sorted) + 1),
        index=crossing_frames_sorted.index,
        name="crossing_order",
    )

    crossing_info = pd.concat([crossing_frames, crossing_order], axis=1)

    # --- Voronoi Polygon and Density Computation ---
    # Compute individual Voronoi polygons; ensure 'walkable_area' is defined properly.
    individual = compute_individual_voronoi_polygons(
        traj_data=traj,
        walkable_area=walkable_area,
        cut_off=Cutoff(
            radius=polygon_cutoff_radius, quad_segments=polygon_quad_segments
        ),
    )

    # Compute mean density over time (group by frame)
    density_over_time = (
        individual.groupby("frame")[
            "density"
        ].mean()  # .rename("mean_individual_density")
    )

    # Compute mean density per agent (group by id)
    density_per_agent = individual.groupby("id")["density"].mean()

    # Compute polygon area for each individual.
    individual["area"] = individual["polygon"].apply(lambda poly: poly.area)

    # Compute mean polygon area per agent.
    area_per_agent = individual.groupby("id")["area"].mean()

    # --- Merging the Data ---
    # Create DataFrames from Series
    df_order = crossing_order.to_frame(name="order")
    df_density = density_per_agent.to_frame(name="density")
    df_area = area_per_agent.to_frame()

    # Merge on the 'id' index (make sure all series are indexed by id)
    df_merged = df_order.join(df_density, how="inner").join(df_area, how="inner")

    return df_merged, density_over_time, crossing_info, individual, title


def calculate_final_rank_density(
    filename,
    walkable_area,
    door_center,
    fps=50,
    polygon_cutoff_radius=1,
    polygon_quad_segments=3,
    title=None,
    file_type="experiment",
):
    if file_type == "experiment":
        if not title:
            title = filename.split("_")[1].capitalize()
        df = pd.read_csv(
            filename, sep="\t", names=["id", "frame", "x", "y", "z", "m"], comment="#"
        )
        traj = pedpy.TrajectoryData(df, frame_rate=fps)
        print(filename, traj.frame_rate)
    elif file_type == "simulation":
        traj = pedpy.load_trajectory_from_jupedsim_sqlite(filename)
        walkable_area = pedpy.load_walkable_area_from_jupedsim_sqlite(filename)
        df = traj.data
        print(
            f"Calculate final rank density for simulation file: {filename = }, {traj.frame_rate = }"
        )

    print(f"Processing file: {title}")

    if door_center is None:
        raise ValueError("door_center is required for final rank analysis.")

    last_frame = int(df["frame"].max())
    df_last = df[df["frame"] == last_frame].copy()
    dx = df_last["x"] - float(door_center[0])
    dy = df_last["y"] - float(door_center[1])
    df_last["distance_to_door"] = np.sqrt(dx * dx + dy * dy)
    df_last = df_last.sort_values(["distance_to_door", "id"])
    final_rank = pd.Series(
        range(1, len(df_last) + 1),
        index=df_last["id"].to_numpy(),
        name="final_rank",
    )
    final_rank_info = (
        df_last.set_index("id")[["frame", "distance_to_door"]]
        .rename(columns={"frame": "final_frame"})
        .join(final_rank)
    )

    individual = compute_individual_voronoi_polygons(
        traj_data=traj,
        walkable_area=walkable_area,
        cut_off=Cutoff(
            radius=polygon_cutoff_radius, quad_segments=polygon_quad_segments
        ),
    )
    density_over_time = individual.groupby("frame")["density"].mean()
    density_per_agent = individual.groupby("id")["density"].mean()
    individual["area"] = individual["polygon"].apply(lambda poly: poly.area)
    area_per_agent = individual.groupby("id")["area"].mean()

    df_rank = final_rank.to_frame(name="final_rank")
    df_density = density_per_agent.to_frame(name="density")
    df_area = area_per_agent.to_frame()
    df_merged = df_rank.join(df_density, how="inner").join(df_area, how="inner")

    return df_merged, density_over_time, final_rank_info, individual, title


def plot_crossing_order_vs_area(
    df_merged,
    filename_stem,
    title=None,
    color="blue",
    output_dir="index_area_figs",
    figsize=(6, 4),
    marker="o",
):
    """
    Plot mean Voronoi polygon area per agent vs crossing order.

    Parameters:
    - df_merged: DataFrame with 'order' and 'area' columns (from process_trajectory_file)
    - filename_stem: str or Path.stem for output image filename
    - title: optional title for the plot
    - color: color of scatter points
    - output_dir: directory where the plot will be saved
    - figsize: figure size
    - marker: marker style for scatter plot
    """
    if df_merged.empty:
        print(f"Skipping crossing order plot for {filename_stem}: no crossings found.")
        return False

    plt.figure(figsize=figsize)
    plt.scatter(df_merged["order"], df_merged["area"], color=color, marker=marker)

    plt.xlabel("Crossing Order (1 = first to cross)", size=14)
    plt.ylabel(r"Mean Area per Agent / $m^2$", size=14)
    if title:
        plt.title(title)

    max_order = int(df_merged["order"].max())
    print(f"Max crossing order: {max_order}")
    plt.xticks(range(1, max_order + 1, 20))
    plt.ylim([0, 3])
    plt.grid(True, alpha=0.3)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = f"{output_dir}/{filename_stem}.pdf"
    plt.savefig(output_path, bbox_inches="tight")

    print(f"Plot: {output_path}")
    return True
    # plt.show()


def plot_final_rank_vs_area(
    df_merged,
    filename_stem,
    title=None,
    color="blue",
    output_dir="final_rank_figs",
    figsize=(6, 4),
    marker="o",
):
    """Plot mean Voronoi polygon area per agent vs final rank."""
    if df_merged.empty:
        print(f"Skipping final rank plot for {filename_stem}: no agents found.")
        return False

    plt.figure(figsize=figsize)
    plt.scatter(df_merged["final_rank"], df_merged["area"], color=color, marker=marker)

    plt.xlabel("Final Rank (1 = closest to door at last frame)", size=14)
    plt.ylabel(r"Mean Area per Agent / $m^2$", size=14)
    if title:
        plt.title(title)

    max_rank = int(df_merged["final_rank"].max())
    print(f"Max final rank: {max_rank}")
    plt.xticks(range(1, max_rank + 1, 20))
    plt.ylim([0, 3])
    plt.grid(True, alpha=0.3)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = f"{output_dir}/{filename_stem}.pdf"
    plt.savefig(output_path, bbox_inches="tight")

    print(f"Plot: {output_path}")
    return True
