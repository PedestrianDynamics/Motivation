import json
from typing import Dict, List, Tuple

import pandas as pd
import pedpy
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pedpy.column_identifier import CUMULATED_COL, DENSITY_COL, TIME_COL
from plotly.graph_objs import Figure

from .inifile_parser import parse_accessible_areas


def plot_density_time_series(df_data: pd.DataFrame) -> None:
    """Figure for density."""
    fig_density = go.Figure(
        data=[go.Scatter(y=df_data[DENSITY_COL], mode="lines", name="Density")]
    )
    fig_density.update_layout(
        title="Density over Time Steps",
        xaxis_title="Time Steps",
        yaxis_title="Density (1/m/m)",
    )
    fig_density.update_yaxes(range=[0, 12])

    st.plotly_chart(fig_density)


def plot_flow_time_series(df_data: pd.DataFrame) -> None:
    """Figure for flow."""
    fig_flow = go.Figure(
        data=[go.Scatter(y=df_data["flow"], mode="lines", name="Flow")]
    )
    fig_flow.update_layout(
        title="Flow over Time Steps", xaxis_title="Time Steps", yaxis_title="Flow (1/s)"
    )
    st.plotly_chart(fig_flow)


def plot_speed_time_series(df_data: pd.DataFrame) -> None:
    fig_speed = go.Figure(data=[go.Scatter(y=df_data, mode="lines", name="Mean Speed")])
    fig_speed.update_layout(
        title="Mean Speed over Time Steps",
        xaxis_title="Time Steps",
        yaxis_title="Mean Speed (m/s)",
    )

    st.plotly_chart(fig_speed)


def plotly_nt_series(df_data: pd.DataFrame) -> None:
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


def set_color_and_size(data_df: pd.DataFrame) -> Tuple[str, List[float]]:
    """Set color and size with help of trajectory data.

    Args:
        data_df (_type_): DataFrame containing trajectory data

    Returns:
        _type_: color and range_color
    """
    if "speed" in data_df.columns:
        color = "speed"
        range_color = [0, max(data_df["speed"])]
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


def moving_trajectories(config_file: str, trajectory_data: pd.DataFrame) -> None:
    """Generate moving trajectories based on simulation.

    Args:
        config_file (_type_): JSON file
        output_file (_type_): trajectory data file
    """
    data_df = pedpy.compute_individual_speed(
        traj_data=trajectory_data,
        frame_step=5,
        speed_calculation=pedpy.SpeedCalculation.BORDER_SINGLE_SIDED,
    )
    data_df = data_df.merge(trajectory_data.data, on=["id", "frame"], how="left")
    data_df["radius"] = 0.2
    color, range_color = set_color_and_size(data_df)
    fig = px.scatter(
        data_df,
        x="x",
        y="y",
        animation_frame="frame",
        animation_group="id",
        color=color,
        size="radius",
        range_color=range_color,
        color_continuous_scale=px.colors.diverging.RdBu_r[::-1],
    )

    with open(config_file, "r", encoding="utf-8") as fig1:
        json_str = fig1.read()
        data = json.loads(json_str)
        polygons = parse_accessible_areas(data)
        geo_min_x = min(point[0] for polygon in polygons.values() for point in polygon)
        geo_max_x = max(point[0] for polygon in polygons.values() for point in polygon)
        geo_min_y = min(point[1] for polygon in polygons.values() for point in polygon)
        geo_max_y = max(point[1] for polygon in polygons.values() for point in polygon)

    update_fig_layout(fig, geo_min_x, geo_max_x, geo_min_y, geo_max_y)
    add_polygons_to_fig(fig, polygons)
    customize_fig(fig)

    st.plotly_chart(fig)
