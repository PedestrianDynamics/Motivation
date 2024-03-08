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
from .plotting import plot_density_time_series, plotly_nt_series, plotly_time_series
from .ui import ui_measurement_parameters
from jupedsim.internal.notebook_utils import read_sqlite_file


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


def run(data: pedpy.TrajectoryData, CONFIG_FILE: str):
    heatmap_files = glob.glob("*Heatmap.*")
    selected_heatmap_file = st.selectbox(
        "Select heatmap file", list(set(heatmap_files))
    )
    if selected_heatmap_file and Path(selected_heatmap_file).exists():
        values = np.loadtxt(selected_heatmap_file)
        if values.any():
            generate_heatmap(CONFIG_FILE, values[:, 0], values[:, 1], values[:, 2])

    fps = parse_fps(data)
    SELECTED_OUTPUT_FILE = st.selectbox(
        "Select file", list(set(glob.glob("files/*.sqlite")))
    )
    ui_measurement_parameters(data)
    if SELECTED_OUTPUT_FILE:
        traj, walkable_area = read_sqlite_file(SELECTED_OUTPUT_FILE)
        parsed_measurement_line = data["measurement_line"]["vertices"]
        measurement_area = pedpy.MeasurementArea(data["measurement_area"]["vertices"])
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
        st.pyplot(fig)
        nt, crossing_frames = pedpy.compute_n_t(
            traj_data=traj,
            measurement_line=measurement_line,
        )
        individual_speed = pedpy.compute_individual_speed(traj_data=traj, frame_step=10)
        flow_speed = pedpy.compute_flow(
            nt=nt,
            crossing_frames=crossing_frames,
            individual_speed=individual_speed,
            delta_frame=10,
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
        frame_value = st.number_input(
            "Select Frame",
            min_value=min(data.frame),
            max_value=max(data.frame),
            value=10,
            step=5,
        )
        fig2 = plt.figure()
        ax2 = pedpy.plot_voronoi_cells(
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
