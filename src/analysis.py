import glob
import json
from pathlib import Path
from typing import Any

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pedpy
import streamlit as st
from jupedsim.internal.notebook_utils import read_sqlite_file
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from pedpy.column_identifier import FRAME_COL, ID_COL

from .inifile_parser import parse_fps
from .plotting import (
    plot_density_time_series,
    plot_flow_time_series,
    plot_speed_time_series,
    plotly_nt_series,
)
from .ui import ui_measurement_parameters
from .utilities import (
    add_heatmap_trace,
    add_polygon_traces,
    calculate_heatmap_values,
    create_empty_figure,
    customize_fig_layout,
    update_figure_layout,
)

# from shapely import Polygon
# from shapely.ops import unary_union


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


def run() -> None:
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

            if selected == "Flow":
                ui_measurement_parameters(json_data)
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
                ui_measurement_parameters(json_data)
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
                speed = pedpy.compute_individual_speed(
                    traj_data=traj,
                    frame_step=5,
                    speed_calculation=pedpy.SpeedCalculation.BORDER_SINGLE_SIDED,
                )
                data = speed.merge(traj.data, on=[ID_COL, FRAME_COL])
                df_time_distance["time_seconds"] = (
                    df_time_distance["time"] / traj.frame_rate
                )
                speed = speed.merge(df_time_distance, on=[ID_COL, FRAME_COL])
                first_frame_speed = speed.loc[
                    speed[FRAME_COL] == speed[FRAME_COL].min(),
                    ["speed", "time_seconds", "distance"],
                ]
                norm = Normalize(speed.min().speed, speed.max().speed)
                cmap = cm.jet  # type: ignore
                # ---------------
                trajectory_ids = df_time_distance["id"].unique()
                fig, ax = plt.subplots()
                for traj_id in trajectory_ids:
                    traj_data = df_time_distance[df_time_distance[ID_COL] == traj_id]
                    speed_id = speed[speed[ID_COL] == traj_id].speed.to_numpy()
                    # Extract points and speeds for the current trajectory
                    points = traj_data[["distance", "time_seconds"]].to_numpy()
                    # st.dataframe(points)
                    # st.dataframe(points)
                    # Prepare segments for the current trajectory
                    segments = [
                        [
                            (points[i, 0], points[i, 1]),
                            (points[i + 1, 0], points[i + 1, 1]),
                        ]
                        for i in range(len(points) - 1)
                    ]
                    lc = LineCollection(segments, cmap="jet", alpha=0.7, norm=norm)
                    lc.set_array(speed_id)
                    lc.set_linewidth(0.5)
                    line = ax.add_collection(lc)

                ax.scatter(
                    first_frame_speed["distance"],
                    first_frame_speed["time_seconds"],
                    c=first_frame_speed["speed"],
                    cmap=cmap,
                    norm=norm,
                    s=10,
                )

                cbar = fig.colorbar(line, ax=ax)
                cbar.set_label("Speed / m/s")
                ax.autoscale()
                ax.margins(0.1)
                ax.set_title("Distance to entrance/Time to entrance")
                plt.grid(alpha=0.3)
                ax.set_xlabel("Distance / m")
                ax.set_ylabel("Time / s")
                st.pyplot(fig)
