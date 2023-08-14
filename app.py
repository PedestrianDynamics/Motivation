"""
Module Name: Jupedsim 
Description: This module contains functions for visualizing and simulating data.
Author: Mohcine Chraibi
Date: August 11, 2023
"""

import json
import pathlib as p
import subprocess
from typing import Any, Dict

import numpy as np
import numpy.typing as npt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from scipy import stats
from src import inifile_parser as parser


# pd.DataFrame(data, columns=header)


def read_data(output_file):
    """reading data from csv file

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

def set_color_and_size(data_df):
    """setting color and size with help of trajectory data

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

def update_fig_layout(fig, geo_min_x, geo_max_x, geo_min_y, geo_max_y):
    """ Update figure layout to adjust axes ranges

    Args:
        fig (_type_): Plotly figure
        geo_min_x (_type_): Minimum x-coordinate
        geo_max_x (_type_): Maximum x-coordinate
        geo_min_y (_type_): Minimum y-coordinate
        geo_max_y (_type_): Maximum y-coordinate
    """
    fig.update_xaxes(range=[geo_min_x, geo_max_x])
    fig.update_yaxes(range=[geo_min_y, geo_max_y])

def add_polygons_to_fig(fig, polygons):
    """adding polygons to figure

    Args:
        fig (_type_): Plotly figure
        polygons (_type_): lists of points
    """
    for polygon in polygons.values():
        x_values = [point[0] for point in polygon] + [polygon[0][0]]
        y_values = [point[1] for point in polygon] + [polygon[0][1]]
        fig.add_trace(
            go.Scatter(
                x=x_values, y=y_values, mode="lines", line={"color": 'grey'},)
        )

def customize_fig(fig):
    """Customize the appearance and layout of the Plotly figure

    Args:
        fig (_type_): Plotly figure
    """
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 30
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 5
    fig.update_geos(projection_type="equirectangular", visible=True, resolution=110)
    fig.update_traces(marker={"line": {"width": 0.5, "color": "Gray"}})
    fig.update_geos(projection_type="equirectangular", visible=True, resolution=110)
    fig.update_layout(title="Visualisation", showlegend=False)

def moving_trajectories(config_file, output_file):
    """Generate moving trajectories based on simulation

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

    with open(config_file, "r", encoding="utf-8-sig") as fig1:
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


def generate_heatmap(config_file, position_x: npt.NDArray, position_y: npt.NDArray, value: npt.NDArray) -> None:
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

    heatmap_values, xbins, ybins = calculate_heatmap_values(position_x, position_y, value, polygons)

    add_heatmap_trace(fig, xbins, ybins, heatmap_values)

    add_polygon_traces(fig, polygons)

    customize_fig_layout(fig)

    st.plotly_chart(fig)


def parse_geometry(config_file):
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


def create_empty_figure():
    """
    Create an empty Plotly figure.

    Returns:
        go.Figure: An empty Plotly figure.
    """
    return go.Figure(go.Scatter(x=[], y=[], mode="markers", marker={"size": 0}))



def update_figure_layout(fig, polygons):
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

def calculate_heatmap_values( position_x, position_y, value, polygons,):
    """
    Calculate heatmap values using statistical binning

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


def add_heatmap_trace(fig, xbins, ybins, heatmap_values):
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
            colorbar={"title": 'Motivation'},
        )
    )


def add_polygon_traces(fig, polygons):
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
                line={"color": 'white'},
            )
        )


def customize_fig_layout(fig):
    """
    Customize the layout of the Plotly figure.

    Args:
        fig (go.Figure): The Plotly figure.
    """
    fig.update_layout(title="Heatmap", showlegend=False)


def load_json(filename: p.Path):
    """load json file"""

    try:
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except json.JSONDecodeError as error_1:
        st.error(f"Error loading JSON file: {error_1}")
        return {}


def save_json(output: p.Path, data: Dict[str, Any]):
    """save data in json file"""
    with open(output, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def ui_simulation_parameters(data: Dict[str, Any]) -> None:
    """ "Simulation Parameters Section"""
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
    """Motivation Parameters Section"""
    with st.expander("Motivation Parameters"):
        motivation_activated = st.checkbox("Activate motivation")

        if motivation_activated:
            data["motivation_parameters"]["active"] = 1
        else:
            data["motivation_parameters"]["active"] = 0

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
                    vertex[0] = column_1.number_input("Point X:", value=vertex[0], key=x_key)
                    vertex[1] = column_2.number_input("Point Y:", value=vertex[1], key=y_key)


def ui_grid_parameters(data: Dict[str, Any]) -> None:
    """Grid Parameters Section"""

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


if __name__ == "__main__":
    if "data" not in st.session_state:
        st.session_state.data = {}

    if "all_files" not in st.session_state:
        st.session_state.all_files = []
        # User will select from these files to do simulations

    tab1, tab2 = st.tabs(["Initialisation", "Simulation"])

    with tab1:
        column_1, column_2 = st.columns((1, 1))
        file_name = column_1.text_input("Load config file: ", value="files/bottleneck.json")
        json_file = p.Path(file_name)
        data = {}
        if not json_file.exists():
            st.error(f"file: {file_name} does not exist!")
            st.stop()

        with column_1:
            data = load_json(json_file)
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

    # Run Simulation
    with tab2:
        output_file = st.text_input("Result: ", value="files/trajectory.txt")
        config_file = st.selectbox("Select config file", st.session_state.all_files)
        if st.button("Run Simulation"):
            # Modify the command as needed

            command = f"python simulation.py {config_file} {output_file}"
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

        if p.Path(output_file).exists():
            moving_trajectories(config_file, output_file)

        if p.Path("values.txt").exists():
            values = np.loadtxt("values.txt")
            if values.any():
                generate_heatmap(config_file, values[:, 0], values[:, 1], values[:, 2])
