"""Motivation model with jupedsim.

Description: This module contains functions for visualizing and simulating data.
Author: Mohcine Chraibi
Date: August 11, 2023
"""

import json
import pathlib as p
import subprocess
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pedpy.column_identifier import (
    CUMULATED_COL,
    DENSITY_COL,
    FRAME_COL,
    ID_COL,
    TIME_COL,
)
from plotly.graph_objs import Figure
from scipy import stats
from shapely import Polygon
from shapely.ops import unary_union

from src import inifile_parser as parser
from src.utilities import delete_txt_files


def read_data(output_file: str) -> pd.DataFrame:
    """Read data from csv file.

    Args:
        output_file : path to csv file

    Returns:
        _type_: dataframe containing trajectory data
    """
    data_df = pd.read_csv(
        output_file,
        sep=r"\s+",
        dtype=np.float64,
        comment="#",
        names=["ID", "FR", "X", "Y", "Z", "A", "B", "P", "PP"],
    )
    return data_df


def set_color_and_size(data_df: pd.DataFrame) -> Tuple[str, List[float]]:
    """Set color and size with help of trajectory data.

    Args:
        data_df (_type_): DataFrame containing trajectory data

    Returns:
        _type_: color and range_color
    """
    if "SPEED" in data_df.columns:
        color = "SPEED"
        range_color = [0, max(data_df["SPEED"])]
    elif "COLOR" in data_df.columns:
        color = "COLOR"
        range_color = [0, 255]
    else:
        data_df["COLOR"] = 125
        color = "COLOR"
        range_color = [0, 125]

    if "A" in data_df.columns:
        data_df["A"] /= 2
    else:
        data_df["A"] = 0.2

    return color, range_color


def update_fig_layout(
    fig: Figure, geo_min_x: float, geo_max_x: float, geo_min_y: float, geo_max_y: float
) -> None:
    """Update figure layout to adjust axes ranges.

    Args:
        fig (_type_): Plotly figure
        geo_min_x (_type_): Minimum x-coordinate
        geo_max_x (_type_): Maximum x-coordinate
        geo_min_y (_type_): Minimum y-coordinate
        geo_max_y (_type_): Maximum y-coordinate
    """
    fig.update_xaxes(range=[geo_min_x, geo_max_x])
    fig.update_yaxes(range=[geo_min_y, geo_max_y])


def add_polygons_to_fig(fig: Figure, polygons: Dict[int, List[List[float]]]) -> None:
    """Add polygons to figure.

    Args:
        fig (_type_): Plotly figure
        polygons (_type_): lists of points
    """
    for polygon in polygons.values():
        x_values = [point[0] for point in polygon] + [polygon[0][0]]
        y_values = [point[1] for point in polygon] + [polygon[0][1]]
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                line={"color": "grey"},
            )
        )


def customize_fig(fig: Figure) -> None:
    """Customize the appearance and layout of the Plotly figure.

    Args:
        fig (_type_): Plotly figure
    """
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 30
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 5
    fig.update_geos(projection_type="equirectangular", visible=True, resolution=110)
    fig.update_traces(marker={"line": {"width": 0.5, "color": "Gray"}})
    fig.update_geos(projection_type="equirectangular", visible=True, resolution=110)
    fig.update_layout(title="Visualisation", showlegend=False)


def moving_trajectories(config_file: str, output_file: str) -> None:
    """Generate moving trajectories based on simulation.

    Args:
        config_file (_type_): JSON file
        output_file (_type_): trajectory data file
    """
    data_df = read_data(output_file)
    color, range_color = set_color_and_size(data_df)
    fig = px.scatter(
        data_df,
        x="X",
        y="Y",
        animation_frame="FR",
        animation_group="ID",
        color=color,
        size="A",
        range_color=range_color,
        color_continuous_scale=px.colors.diverging.RdBu_r[::-1],
    )

    with open(config_file, "r", encoding="utf-8") as fig1:
        json_str = fig1.read()
        data = json.loads(json_str)
        polygons = parser.parse_accessible_areas(data)
        geo_min_x = min(point[0] for polygon in polygons.values() for point in polygon)
        geo_max_x = max(point[0] for polygon in polygons.values() for point in polygon)
        geo_min_y = min(point[1] for polygon in polygons.values() for point in polygon)
        geo_max_y = max(point[1] for polygon in polygons.values() for point in polygon)

    update_fig_layout(fig, geo_min_x, geo_max_x, geo_min_y, geo_max_y)
    add_polygons_to_fig(fig, polygons)
    customize_fig(fig)

    st.plotly_chart(fig)


def generate_heatmap(
    config_file: str,
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
    # Parse the geometry from the JSON file (Accessible areas).
    polygons = parse_geometry(config_file)

    fig = create_empty_figure()
    update_figure_layout(fig, polygons)

    heatmap_values, xbins, ybins = calculate_heatmap_values(
        position_x, position_y, value, polygons
    )

    add_heatmap_trace(fig, xbins, ybins, heatmap_values)

    add_polygon_traces(fig, polygons)

    customize_fig_layout(fig)

    st.plotly_chart(fig)


def parse_geometry(config_file: str) -> Dict[int, List[List[float]]]:
    """
    Parse accessible areas from a JSON configuration file.

    Args:
        config_file (str): Path to the JSON configuration file.

    Returns:
        dict: Parsed polygons representing accessible areas.
    """
    with open(config_file, "r", encoding="utf8") as fig2:
        json_str = fig2.read()
        data = json.loads(json_str)
        return parser.parse_accessible_areas(data)


def create_empty_figure() -> Figure:
    """
    Create an empty Plotly figure.

    Returns:
        go.Figure: An empty Plotly figure.
    """
    return go.Figure(go.Scatter(x=[], y=[], mode="markers", marker={"size": 0}))


def update_figure_layout(fig: Figure, polygons: Dict[int, List[List[float]]]) -> None:
    """
    Update the layout of the Plotly figure based on polygon boundaries.

    Args:
        fig (go.Figure): The Plotly figure.
        polygons (dict): Dictionary of polygons representing accessible areas.
    """
    geo_min_x = min(point[0] for polygon in polygons.values() for point in polygon)
    geo_max_x = max(point[0] for polygon in polygons.values() for point in polygon)
    geo_min_y = min(point[1] for polygon in polygons.values() for point in polygon)
    geo_max_y = max(point[1] for polygon in polygons.values() for point in polygon)

    fig.update_xaxes(range=[geo_min_x, geo_max_x])
    fig.update_yaxes(range=[geo_min_y, geo_max_y])


def calculate_heatmap_values(
    position_x: npt.NDArray[Any],
    position_y: npt.NDArray[Any],
    value: npt.NDArray[Any],
    polygons: Dict[int, List[List[float]]],
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """
    Calculate heatmap values using statistical binning.

    Args:
        position_x: Array of X positions.
        position_y: Array of Y positions.
        value: Array of values associated with positions.
        polygons (dict): polygons representing accessible areas.

    """
    geo_min_x = min(point[0] for polygon in polygons.values() for point in polygon)
    geo_max_x = max(point[0] for polygon in polygons.values() for point in polygon)
    geo_min_y = min(point[1] for polygon in polygons.values() for point in polygon)
    geo_max_y = max(point[1] for polygon in polygons.values() for point in polygon)
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


def add_polygon_traces(fig: Figure, polygons: Dict[int, List[List[float]]]) -> None:
    """
    Add polygon traces to the Plotly figure.

    Args:
        fig (go.Figure): The Plotly figure.
        polygons (dict): Dictionary of polygons representing accessible areas.
    """
    for polygon in polygons.values():
        x_values = [point[0] for point in polygon] + [polygon[0][0]]
        y_values = [point[1] for point in polygon] + [polygon[0][1]]
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
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


def load_json(filename: p.Path) -> Any:
    """Load json file."""

    try:
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except json.JSONDecodeError as error_1:
        st.error(f"Error loading JSON file: {error_1}")
        return {}


def save_json(output: p.Path, data: Dict[str, Any]) -> None:
    """Save data in json file."""
    with open(output, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def ui_measurement_parameters(data: Dict[str, Any]) -> None:
    """Measurement lines, polygons."""
    with st.expander("Measurement Parameters"):
        st.code("Measurement line:")
        column_1, column_2 = st.columns((1, 1))
        line = data["measurement_line"]["vertices"]
        for idx, vertex in enumerate(line):
            x_key = f"l_vertex_x_{idx}"
            y_key = f"l_vertex_y_{idx}"
            vertex[0] = column_1.number_input("Point X:", value=vertex[0], key=x_key)
            vertex[1] = column_2.number_input("Point Y:", value=vertex[1], key=y_key)

        st.code("Measurement area:")
        column_1, column_2 = st.columns((1, 1))
        area = data["measurement_area"]["vertices"]
        for idx, vertex in enumerate(area):
            x_key = f"a_vertex_x_{idx}"
            y_key = f"a_vertex_y_{idx}"
            vertex[0] = column_1.number_input("Point X:", value=vertex[0], key=x_key)
            vertex[1] = column_2.number_input("Point Y:", value=vertex[1], key=y_key)


def ui_velocity_model_parameters(data: Dict[str, Any]) -> None:
    """Set velocity Parameters Section."""
    with st.expander("Velocity model Parameters"):
        data["velocity_init_parameters"]["a_ped"] = st.slider(
            "a_ped:",
            min_value=0.0,
            max_value=10.0,
            value=data["velocity_init_parameters"]["a_ped"],
        )
        data["velocity_init_parameters"]["d_ped"] = st.slider(
            "d_ped:",
            min_value=0.01,
            max_value=1.0,
            value=data["velocity_init_parameters"]["d_ped"],
        )
        data["velocity_init_parameters"]["a_wall"] = st.slider(
            "a_wall:",
            min_value=0.0,
            max_value=10.0,
            value=data["velocity_init_parameters"]["a_wall"],
        )
        data["velocity_init_parameters"]["d_wall"] = st.slider(
            "d_wall:",
            min_value=0.01,
            max_value=1.0,
            value=data["velocity_init_parameters"]["d_wall"],
        )


def ui_simulation_parameters(data: Dict[str, Any]) -> None:
    """Set simulation Parameters Section."""
    with st.expander("Simulation Parameters"):
        data["simulation_parameters"]["fps"] = st.slider(
            "FPS:",
            min_value=1,
            max_value=60,
            value=data["simulation_parameters"]["fps"],
        )
        data["simulation_parameters"]["time_step"] = st.number_input(
            "Time Step:", value=data["simulation_parameters"]["time_step"]
        )
        data["simulation_parameters"]["number_agents"] = st.number_input(
            "Number of Agents:", value=data["simulation_parameters"]["number_agents"]
        )
        data["simulation_parameters"]["simulation_time"] = st.number_input(
            "Simulation Time:", value=data["simulation_parameters"]["simulation_time"]
        )


def ui_motivation_parameters(data: Dict[str, Any]) -> None:
    """Motivation Parameters Section."""
    with st.expander("Motivation Parameters"):
        motivation_activated = st.checkbox("Activate motivation")
        if motivation_activated:
            data["motivation_parameters"]["active"] = 1
        else:
            data["motivation_parameters"]["active"] = 0

        motivation_strategy = st.selectbox(
            "Select model",
            ["default", "EVC"],
            help="Model 2: M = M(dist). Model 3: M = V.E, Model4: M=V.E.C",
        )
        data["motivation_parameters"]["width"] = st.text_input(
            "Width",
            key="width",
            value=float(data["motivation_parameters"]["width"]),
            help="width of function defining distance dependency",
        )
        data["motivation_parameters"]["height"] = st.text_input(
            "Height",
            key="hight",
            value=float(data["motivation_parameters"]["height"]),
            help="Height of function defining distance dependency",
        )

        data["motivation_parameters"]["seed"] = st.text_input(
            "Seed",
            key="seed",
            value=float(data["motivation_parameters"]["seed"]),
            help="Seed for random generator for value",
        )

        data["motivation_parameters"]["max_value"] = st.text_input(
            "Max_value",
            key="max_value",
            value=float(data["motivation_parameters"]["max_value"]),
            help="Max Value",
        )

        data["motivation_parameters"]["min_value"] = st.text_input(
            "Min_value",
            key="min_value",
            value=float(data["motivation_parameters"]["min_value"]),
            help="Min Value",
        )

        data["motivation_parameters"]["motivation_strategy"] = motivation_strategy
        data["motivation_parameters"]["normal_v_0"] = st.slider(
            "Normal V0:",
            min_value=0.5,
            max_value=2.5,
            value=float(data["motivation_parameters"]["normal_v_0"]),
        )
        data["motivation_parameters"]["normal_time_gap"] = st.slider(
            "Normal Time Gap:",
            min_value=0.1,
            max_value=3.0,
            step=0.1,
            value=float(data["motivation_parameters"]["normal_time_gap"]),
        )
        column_1, column_2 = st.columns((1, 1))
        for door_idx, door in enumerate(
            data["motivation_parameters"]["motivation_doors"]
        ):
            column_2.text_input(
                "Door Label:", value=door["label"], key=f"door_label_{door_idx}"
            )
            door["id"] = column_1.number_input(
                "Door ID:", value=door["id"], key=f"door_id_{door_idx}"
            )
            for door_idx, door in enumerate(
                data["motivation_parameters"]["motivation_doors"]
            ):
                for vertex_idx, vertex in enumerate(door["vertices"]):
                    x_key = f"vertex_x_{door_idx}_{vertex_idx}"
                    y_key = f"vertex_y_{door_idx}_{vertex_idx}"
                    vertex[0] = column_1.number_input(
                        "Point X:", value=vertex[0], key=x_key
                    )
                    vertex[1] = column_2.number_input(
                        "Point Y:", value=vertex[1], key=y_key
                    )


def ui_grid_parameters(data: Dict[str, Any]) -> None:
    """Grid Parameters Section."""
    with st.expander("Grid Parameters"):
        column_1, column_2, column_3 = st.columns((1, 1, 1))
        data["grid_parameters"]["min_v_0"] = column_1.number_input(
            "Min V0:", value=data["grid_parameters"]["min_v_0"]
        )
        data["grid_parameters"]["max_v_0"] = column_2.number_input(
            "Max V0:", value=data["grid_parameters"]["max_v_0"]
        )
        data["grid_parameters"]["v_0_step"] = column_3.number_input(
            "V0 Step:", value=data["grid_parameters"]["v_0_step"]
        )
        data["grid_parameters"]["min_time_gap"] = column_1.number_input(
            "Min Time Gap:", value=data["grid_parameters"]["min_time_gap"]
        )
        data["grid_parameters"]["max_time_gap"] = column_2.number_input(
            "Max Time Gap:", value=data["grid_parameters"]["max_time_gap"]
        )
        data["grid_parameters"]["time_gap_step"] = column_3.number_input(
            "Time Gap Step:", value=data["grid_parameters"]["time_gap_step"]
        )


def plot_density_time_series(df_data: pd.DataFrame):
    """Figure for density."""
    fig_density = go.Figure(
        data=[go.Scatter(y=df_data[DENSITY_COL], mode="lines", name="Density")]
    )
    fig_density.update_layout(
        title="Density over Time Steps",
        xaxis_title="Time Steps",
        yaxis_title="Density (1/m/m)",
    )

    st.plotly_chart(fig_density)


def plotly_time_series(df_data: pd.DataFrame):
    """Figure for flow."""
    fig_flow = go.Figure(
        data=[go.Scatter(y=df_data["flow"], mode="lines", name="Flow")]
    )
    fig_flow.update_layout(
        title="Flow over Time Steps", xaxis_title="Time Steps", yaxis_title="Flow (1/s)"
    )

    st.plotly_chart(fig_flow)

    fig_speed = go.Figure(
        data=[go.Scatter(y=df_data["mean_speed"], mode="lines", name="Mean Speed")]
    )
    fig_speed.update_layout(
        title="Mean Speed over Time Steps",
        xaxis_title="Time Steps",
        yaxis_title="Mean Speed (m/s)",
    )

    st.plotly_chart(fig_speed)


def plotly_nt_series(df_data: pd.DataFrame):
    """Figure for flow."""
    fig_nt = go.Figure(
        data=[
            go.Scatter(
                x=df_data[TIME_COL], y=df_data[CUMULATED_COL], mode="lines", name="NT"
            )
        ]
    )
    fig_nt.update_layout(
        title="NT", xaxis_title="Time Steps", yaxis_title="# pedestrians"
    )

    st.plotly_chart(fig_nt)


if __name__ == "__main__":
    if "data" not in st.session_state:
        st.session_state.data = {}

    if "all_files" not in st.session_state:
        st.session_state.all_files = []
        # User will select from these files to do simulations

    tab1, tab2, tab3 = st.tabs(["Initialisation", "Simulation", "Analysis"])

    with tab1:
        column_1, column_2 = st.columns((1, 1))
        file_name = column_1.text_input(
            "Load config file: ", value="files/bottleneck.json"
        )
        json_file = p.Path(file_name)
        data = {}
        if not json_file.exists():
            st.error(f"file: {file_name} does not exist!")
            st.stop()

        with column_1:
            data = load_json(json_file)
            ui_velocity_model_parameters(data)
            ui_simulation_parameters(data)
            ui_motivation_parameters(data)
            ui_grid_parameters(data)
            st.session_state.data = data
            st.session_state.all_files.append(file_name)

        # Save Button (optional)
        new_json_name = column_2.text_input(
            "Save config file: ", value="files/bottleneck.json"
        )
        new_json_file = p.Path(new_json_name)
        if column_2.button(
            "Save config",
            help=f"After changing the values, you can save the configs in a separate file ({new_json_name})",
        ):
            save_json(new_json_file, data)
            column_1.info(f"Saved file as {new_json_name}")
            st.session_state.all_files.append(new_json_name)

        if column_2.button("Reset", help="Delete all trajectory files"):
            delete_txt_files()

    # Run Simulation
    with tab2:
        OUTPUT_FILE = st.text_input("Result: ", value="files/trajectory.txt")
        CONFIG_FILE = str(
            st.selectbox("Select config file", list(set(st.session_state.all_files)))
        )
        if st.button("Run Simulation"):
            # Modify the command as needed

            command = f"python simulation.py {CONFIG_FILE} {OUTPUT_FILE}"
            n_agents = st.session_state.data["simulation_parameters"]["number_agents"]
            with st.spinner(f"Simulating with {n_agents}"):
                with subprocess.Popen(
                    command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                ) as process:
                    stdout, stderr = process.communicate()
                INFO_OUTPUT = stdout.decode().replace("\n", "  \n")
                WARNINGS = stderr.decode().replace("\n", "  \n")
                st.info(INFO_OUTPUT)
                if WARNINGS:
                    st.error(WARNINGS)

        output_path = p.Path(OUTPUT_FILE)
        if output_path.exists():
            moving_trajectories(CONFIG_FILE, OUTPUT_FILE)

        if p.Path("values.txt").exists():
            print(output_path.name)
            p.Path("values.txt").rename(output_path.name + "_Heatmap.txt")

    with tab3:
        # measure flow
        import glob
        import pedpy

        heatmap_files = glob.glob("*Heatmap.*")
        selected_heatmap_file = st.selectbox(
            "Select heatmap file", list(set(heatmap_files))
        )
        if selected_heatmap_file and p.Path(selected_heatmap_file).exists():
            values = np.loadtxt(selected_heatmap_file)
            if values.any():
                generate_heatmap(CONFIG_FILE, values[:, 0], values[:, 1], values[:, 2])

        fps = parser.parse_fps(data)
        SLECTED_OUTPUT_FILE = st.selectbox(
            "Select file", list(set(glob.glob("files/*.txt")))
        )
        ui_measurement_parameters(data)
        if SLECTED_OUTPUT_FILE:
            traj = pedpy.load_trajectory(
                trajectory_file=p.Path(SLECTED_OUTPUT_FILE),
                default_frame_rate=fps,
                default_unit=pedpy.TrajectoryUnit.METER,
            )

            parsed_measurement_line = data["measurement_line"]["vertices"]
            measurement_area = pedpy.MeasurementArea(
                data["measurement_area"]["vertices"]
            )

            measurement_line = pedpy.MeasurementLine(parsed_measurement_line)
            accessible_areas = parser.parse_accessible_areas(data)
            polygons = [Polygon(value) for value in accessible_areas.values()]
            walkable_area = unary_union(polygons)
            walkable_area = pedpy.WalkableArea(walkable_area)

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
            st.pyplot(fig)
            nt, crossing_frames = pedpy.compute_n_t(
                traj_data=traj,
                measurement_line=measurement_line,
            )
            individual_speed = pedpy.compute_individual_speed(
                traj_data=traj, frame_step=10
            )
            flow_speed = pedpy.compute_flow(
                nt=nt,
                crossing_frames=crossing_frames,
                individual_speed=individual_speed,
                delta_t=10,
                frame_rate=fps,
            )

            individual = pedpy.compute_individual_voronoi_polygons(
                traj_data=traj, walkable_area=walkable_area
            )
            density_voronoi, intersecting = pedpy.compute_voronoi_density(
                individual_voronoi_data=individual, measurement_area=measurement_area
            )
            plot_density_time_series(density_voronoi)
            # pedpy.compute_voronoi_density()
            plotly_time_series(flow_speed)
            plotly_nt_series(nt)

            data = individual.merge(traj.data, on=[ID_COL, FRAME_COL])
            frame_value = st.slider(
                "Select Frame",
                min_value=min(data.frame),
                max_value=max(data.frame),
                value=10,
            )
            fig2 = plt.figure()
            ax2 = pedpy.plot_voronoi_cells(
                data=data[data.frame == frame_value],
                walkable_area=walkable_area,
                color_mode="density",
                vmin=0,
                vmax=10,
                show_ped_positions=True,
                ped_size=5,
            ).set_aspect("equal")
            fig2 = plt.gcf()
            st.pyplot(fig2)
