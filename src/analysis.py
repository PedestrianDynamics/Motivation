"""Analysis module for motivation model."""

import glob
import json
from pathlib import Path
from typing import Any
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy.typing as npt
import pedpy
import streamlit as st
from jupedsim.internal.notebook_utils import read_sqlite_file
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from pedpy.column_identifier import FRAME_COL, ID_COL
from typing import Dict
import numpy as np
from .inifile_parser import parse_fps
from .plotting import (
    plot_density_time_series,
    plot_flow_time_series,
    plot_speed_time_series,
    plotly_nt_series,
)
from .ui import ui_measurement_parameters


def generate_heatmap(
    walkable_area: pedpy.WalkableArea,
    position_x: npt.NDArray[Any],
    position_y: npt.NDArray[Any],
    value: npt.NDArray[Any],
) -> None:
    """
    Generate and display a heatmap plot based on provided data.

    Args:
        config_file (str): Path to the configuration JSON file containing accessible areas.
        position_x: Array of X positions.
        position_y: Array of Y positions.
        value: Array of values associated with positions.
    """
    x, y = walkable_area.polygon.exterior.xy
    x = np.array(x)
    y = np.array(y)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y, color="black", label="Exterior Boundary")
    eps: float = 1.0
    geo_min_x = -3.54 - eps
    geo_max_x = 3.62 + eps
    geo_min_y = -1.13 - eps
    geo_max_y = 21.23 + eps
    for i, interior in enumerate(walkable_area.polygon.interiors):
        x_interior, y_interior = interior.xy
        ax.plot(
            x_interior,
            y_interior,
            color="white",
            linestyle="-",
            label=f"Interior {i+1}",
        )

    heatmap, xedges, yedges = np.histogram2d(
        position_x, position_y, bins=50, weights=value
    )
    heatmap = heatmap / np.max(heatmap)
    extent = [geo_min_x, geo_max_x, geo_min_y, geo_max_y]
    plt.imshow(
        heatmap.T,
        origin="lower",
        cmap="jet",
        extent=extent,
        vmin=0,
        vmax=1,
    )
    plt.colorbar(label="Motivation")
    plt.xlabel("X")
    plt.ylabel("Y")
    st.pyplot(fig)


def run() -> None:
    """Run the main logic of the analysis tab."""
    selected = st.sidebar.radio(
        "Choose option",
        [
            "Distance to entrance",
            "Speed",
            "Density",
            "Flow",
            "NT",
            "Voronoi polygons",
            "Heatmap",
        ],
    )
    SELECTED_OUTPUT_FILE = st.selectbox(
        "Select file", sorted(list(set(glob.glob("files/*.sqlite"))), reverse=True)
    )
    traj, walkable_area = read_sqlite_file(SELECTED_OUTPUT_FILE)
    print("HHHHHHHH", type(traj))
    json_data = load_json_data("files/inifile.json")

    if selected == "Heatmap":
        handle_heatmap(walkable_area)
    else:
        handle_analysis(selected, SELECTED_OUTPUT_FILE, traj, walkable_area, json_data)


def load_json_data(filepath: str) -> dict:
    """Load JSON data from a given file."""
    with open(filepath, "r", encoding="utf8") as f:
        return json.loads(f.read())


def handle_heatmap(walkable_area) -> None:
    """Handle heatmap selection and generation."""
    heatmap_files = glob.glob("files/*motivation.csv")
    selected_heatmap_file = st.selectbox(
        "Select motivation file", list(set(heatmap_files))
    )

    if selected_heatmap_file and Path(selected_heatmap_file).exists():
        df = pd.read_csv(
            selected_heatmap_file, names=["frame", "id", "time", "motivation", "x", "y"]
        )
        if not df.empty:
            x_values = df["x"].to_numpy()
            y_values = df["y"].to_numpy()
            motivation_values = df["motivation"].to_numpy()

            generate_heatmap(
                walkable_area,
                position_x=x_values,
                position_y=y_values,
                value=motivation_values,
            )


def handle_analysis(
    selected, SELECTED_OUTPUT_FILE, traj, walkable_area, json_data
) -> None:
    """Handle the analysis based on the selected option."""
    fps = parse_fps(json_data)

    if SELECTED_OUTPUT_FILE:
        output_path = Path(SELECTED_OUTPUT_FILE)
        motivation_file = output_path.with_name(output_path.stem + "_motivation.csv")
        print(f"{motivation_file = }")

        parsed_measurement_line = json_data["measurement_line"]["vertices"]
        measurement_area = pedpy.MeasurementArea(
            json_data["measurement_area"]["vertices"]
        )
        measurement_line = pedpy.MeasurementLine(parsed_measurement_line)
        plot_measurement_setup(walkable_area, traj, measurement_line, measurement_area)

        if selected == "NT":
            handle_nt(traj, measurement_line)
        elif selected == "Speed":
            handle_speed(traj, measurement_area, json_data)
        elif selected == "Flow":
            handle_flow(traj, measurement_line, fps, json_data)
        elif selected == "Density":
            handle_density(traj, walkable_area, measurement_area, json_data)
        elif selected == "Voronoi polygons":
            handle_voronoi(traj, walkable_area)
        elif selected == "Distance to entrance":
            original_positions_file = Path(json_data["init_trajectories_file"])
            original_traj = pedpy.load_trajectory_from_txt(
                trajectory_file=original_positions_file,
            )
            fig = handle_distance_to_entrance(
                traj, measurement_line, motivation_file, prefix="traj"
            )
            fig2 = handle_distance_to_entrance(
                original_traj, measurement_line, motivation_file, prefix="orig"
            )
            c1, c2 = st.columns(2)
            c1.pyplot(fig)
            c2.pyplot(fig2)


def plot_measurement_setup(
    walkable_area, traj, measurement_line, measurement_area
) -> None:
    """Plot the measurement setup."""
    pedpy.plot_measurement_setup(
        walkable_area=walkable_area,
        hole_color="lightgrey",
        traj=traj,
        traj_color="lightblue",
        traj_alpha=0.5,
        traj_width=1,
        measurement_lines=[measurement_line],
        measurement_areas=[measurement_area],
        ml_color="b",
        ma_color="r",
        ma_line_color="r",
        ma_line_width=1,
        ma_alpha=0.2,
    ).set_aspect("equal")
    fig = plt.gcf()
    st.sidebar.pyplot(fig)


def handle_nt(traj, measurement_line) -> None:
    """Handle NT (number of traversals) computation."""
    nt, crossing_frames = pedpy.compute_n_t(
        traj_data=traj, measurement_line=measurement_line
    )
    plotly_nt_series(nt)


def handle_speed(traj, measurement_area, json_data) -> None:
    """Handle speed computation and plotting."""
    ui_measurement_parameters(json_data)
    individual_speed = pedpy.compute_individual_speed(
        traj_data=traj,
        frame_step=5,
        speed_calculation=pedpy.SpeedCalculation.BORDER_SINGLE_SIDED,
    )
    mean_speed = pedpy.compute_mean_speed_per_frame(
        traj_data=traj,
        measurement_area=measurement_area,
        individual_speed=individual_speed,
    )
    plot_speed_time_series(mean_speed)


def handle_flow(traj, measurement_line, fps, json_data) -> None:
    """Handle flow computation and plotting."""
    ui_measurement_parameters(json_data)
    nt, crossing_frames = pedpy.compute_n_t(
        traj_data=traj, measurement_line=measurement_line
    )
    individual_speed = pedpy.compute_individual_speed(traj_data=traj, frame_step=5)

    flow_speed = pedpy.compute_flow(
        nt=nt,
        crossing_frames=crossing_frames,
        individual_speed=individual_speed,
        delta_frame=5,
        frame_rate=fps,
    )
    plot_flow_time_series(flow_speed)


def handle_density(traj, walkable_area, measurement_area, json_data) -> None:
    """Handle density computation and plotting."""
    ui_measurement_parameters(json_data)
    individual = pedpy.compute_individual_voronoi_polygons(
        traj_data=traj, walkable_area=walkable_area
    )
    density_voronoi, intersecting = pedpy.compute_voronoi_density(
        individual_voronoi_data=individual,
        measurement_area=measurement_area,
    )
    plot_density_time_series(density_voronoi)


def handle_voronoi(traj, walkable_area) -> None:
    """Handle Voronoi polygons plotting."""
    individual = pedpy.compute_individual_voronoi_polygons(
        traj_data=traj, walkable_area=walkable_area
    )
    data = individual.merge(traj.data, on=[ID_COL, FRAME_COL])

    frame_value = st.number_input(
        "Select Frame",
        min_value=min(data.frame),
        max_value=max(data.frame),
        value=10,
        step=5,
    )

    fig2 = plt.figure()
    pedpy.plot_voronoi_cells(
        voronoi_data=data[data.frame == frame_value],
        walkable_area=walkable_area,
        color_mode="density",
        frame=frame_value,
        vmin=0,
        vmax=10,
        show_ped_positions=True,
        ped_size=5,
    ).set_aspect("equal")
    st.pyplot(fig2)


# UI handling
def get_user_inputs(prefix=""):
    """Get inputs from the user for the plot configuration with unique keys."""
    c1, c2, c3 = st.columns(3)

    # Add unique keys using the prefix argument
    yaxis_max = c1.number_input(
        f"{prefix}Max y-Axis: ", value=200, step=5, key=f"{prefix}_yaxis_max"
    )
    if prefix != "orig":
        color_by_speed = c3.radio(
            f"{prefix}Select parameter to color by:",
            ("Motivation", "Speed"),
            key=f"{prefix}_color_by_speed",
        )
    else:
        color_by_speed = "Speed"
    colorbar_max = c2.number_input(
        f"{prefix}Max colorbar: ",
        value=(2.0 if color_by_speed == "Speed" else 1.0),
        step=0.1,
        key=f"{prefix}_colorbar_max",
    )
    unit_text = "Speed / m/s" if color_by_speed == "Speed" else "Motivation"

    return yaxis_max, color_by_speed == "Speed", colorbar_max, unit_text


# Logic handling
def compute_speed_or_motivation(traj, motivation_file, color_by_speed):
    """Compute either speed or motivation data based on user input."""
    if color_by_speed:
        # Compute speed
        speed = pedpy.compute_individual_speed(
            traj_data=traj,
            frame_step=5,
            speed_calculation=pedpy.SpeedCalculation.BORDER_SINGLE_SIDED,
        )
    else:
        # Read motivation from file
        speed = pd.read_csv(
            motivation_file,
            names=[FRAME_COL, ID_COL, "time", "speed", "x", "y"],
            dtype={ID_COL: "int64", FRAME_COL: "int64"},
        )
    return speed


def process_data(traj, measurement_line, motivation_file, color_by_speed):
    """Process and merge the data required for plotting."""
    # Compute time distance line
    df_time_distance = pedpy.compute_time_distance_line(
        traj_data=traj, measurement_line=measurement_line
    )
    # Compute speed or motivation
    speed = compute_speed_or_motivation(traj, motivation_file, color_by_speed)

    df_time_distance["time_seconds"] = (
        df_time_distance["time"] / 1.0
    )  # Assuming traj.frame_rate is 1
    # Merge speed/motivation with time distance data
    speed = speed.merge(df_time_distance, on=[ID_COL, FRAME_COL])

    first_frame_speed = speed.loc[
        speed[FRAME_COL] == speed[FRAME_COL].min(),
        ["speed", "time_seconds", "distance"],
    ]

    return speed, df_time_distance, first_frame_speed


# Plotting function
@st.cache_data
def plot_distance_to_entrance(
    df_time_distance,
    speed,
    first_frame_speed,
    color_by_speed,
    yaxis_max,
    colorbar_max,
    unit_text,
):
    """Plot the distance to entrance with speed or motivation coloring."""
    norm = Normalize(speed["speed"].min(), speed["speed"].max())
    cmap = cm.jet

    # Create the figure and axis
    fig, ax = plt.subplots()

    # Plot each trajectory
    trajectory_ids = df_time_distance[ID_COL].unique()
    for traj_id in trajectory_ids:
        traj_data = df_time_distance[df_time_distance[ID_COL] == traj_id]
        speed_id = speed[speed[ID_COL] == traj_id]["speed"].to_numpy()
        points = traj_data[["distance", "time_seconds"]].to_numpy()
        segments = [
            [(points[i, 0], points[i, 1]), (points[i + 1, 0], points[i + 1, 1])]
            for i in range(len(points) - 1)
        ]
        lc = LineCollection(segments, cmap=cmap, alpha=1, norm=norm)
        lc.set_array(speed_id)
        lc.set_linewidth(0.5)
        line = ax.add_collection(lc)

    # Scatter plot of first frame speeds
    ax.scatter(
        first_frame_speed["distance"],
        first_frame_speed["time_seconds"],
        c=first_frame_speed["speed"],
        cmap=cmap,
        norm=norm,
        s=10,
    )

    # Add colorbar and labels
    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label(unit_text)

    # Set plot properties
    ax.autoscale()
    ax.margins(0.1)
    ax.set_title("Distance to entrance/Time to entrance")
    plt.grid(alpha=0.3)
    ax.set_xlabel("Distance / m")
    ax.set_ylabel("Time / s")
    ax.set_ylim(top=yaxis_max)
    line.set_clim(vmax=colorbar_max)

    return fig


# Main handler function
def handle_distance_to_entrance(traj, measurement_line, motivation_file, prefix=""):
    """Handle distance to entrance plotting."""
    # Get user inputs
    yaxis_max, color_by_speed, colorbar_max, unit_text = get_user_inputs(prefix)

    # Process the data
    speed, df_time_distance, first_frame_speed = process_data(
        traj, measurement_line, motivation_file, color_by_speed
    )

    fig = plot_distance_to_entrance(
        df_time_distance,
        speed,
        first_frame_speed,
        color_by_speed,
        yaxis_max,
        colorbar_max,
        unit_text,
    )
    return fig
