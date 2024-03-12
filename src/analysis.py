import glob
import numpy as np
import numpy.typing as npt
from typing import Any
import pedpy

import streamlit as st
from .utilities import (
    create_empty_figure,
    update_figure_layout,
    calculate_heatmap_values,
    add_heatmap_trace,
    add_polygon_traces,
    customize_fig_layout,
)
import json
from pathlib import Path
from .inifile_parser import parse_fps, parse_accessible_areas, parse_geometry
from shapely import Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from pedpy.column_identifier import (
    CUMULATED_COL,
    DENSITY_COL,
    FRAME_COL,
    ID_COL,
)
import pedpy
from .plotting import (
    plot_density_time_series,
    plotly_nt_series,
    plot_flow_time_series,
    plot_speed_time_series,
)
from .ui import ui_measurement_parameters
from jupedsim.internal.notebook_utils import read_sqlite_file


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
    # Parse the geometry from the JSON file (Accessible areas).
    fig = create_empty_figure()
    update_figure_layout(fig, walkable_area.polygon)
    heatmap_values, xbins, ybins = calculate_heatmap_values(
        position_x, position_y, value, walkable_area.polygon
    )

    add_heatmap_trace(fig, xbins, ybins, heatmap_values)

    add_polygon_traces(fig, walkable_area)

    customize_fig_layout(fig)

    st.plotly_chart(fig)


def run():
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

    with open("files/bottleneck.json", "r", encoding="utf8") as f:
        json_str = f.read()
        json_data = json.loads(json_str)
    ui_measurement_parameters(json_data)

    if selected == "Heatmap":
        heatmap_files = glob.glob("*Heatmap.*")
        selected_heatmap_file = st.selectbox(
            "Select heatmap file", list(set(heatmap_files))
        )
        if selected_heatmap_file and Path(selected_heatmap_file).exists():
            values = np.loadtxt(selected_heatmap_file)
            if values.any():
                generate_heatmap(
                    walkable_area, values[:, 0], values[:, 1], values[:, 2]
                )
    else:
        fps = parse_fps(json_data)
        if SELECTED_OUTPUT_FILE:
            parsed_measurement_line = json_data["measurement_line"]["vertices"]
            measurement_area = pedpy.MeasurementArea(
                json_data["measurement_area"]["vertices"]
            )
            measurement_line = pedpy.MeasurementLine(parsed_measurement_line)
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
            if selected == "NT":
                nt, crossing_frames = pedpy.compute_n_t(
                    traj_data=traj,
                    measurement_line=measurement_line,
                )
                plotly_nt_series(nt)

            if selected == "Speed":
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

            if selected == "Flow":
                nt, crossing_frames = pedpy.compute_n_t(
                    traj_data=traj,
                    measurement_line=measurement_line,
                )
                individual_speed = pedpy.compute_individual_speed(
                    traj_data=traj, frame_step=5
                )

                flow_speed = pedpy.compute_flow(
                    nt=nt,
                    crossing_frames=crossing_frames,
                    individual_speed=individual_speed,
                    delta_frame=5,
                    frame_rate=fps,
                )
                plot_flow_time_series(flow_speed)

            if selected == "Density":
                individual = pedpy.compute_individual_voronoi_polygons(
                    traj_data=traj, walkable_area=walkable_area
                )
                density_voronoi, intersecting = pedpy.compute_voronoi_density(
                    individual_voronoi_data=individual,
                    measurement_area=measurement_area,
                )
                plot_density_time_series(density_voronoi)

            if selected == "Voronoi polygons":
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
                fig2 = plt.gcf()
                st.pyplot(fig2)
            if selected == "Distance to entrance":
                df_time_distance = pedpy.compute_time_distance_line(
                    traj_data=traj, measurement_line=measurement_line
                )
                fig = plt.figure()
                pedpy.plot_time_distance(
                    time_distance=df_time_distance,
                    title="Distance to entrance/Time to entrance",
                    frame_rate=traj.frame_rate,
                )
                st.pyplot(fig)
