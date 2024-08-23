"""Motivation model with jupedsim.

Description: This module contains functions for visualizing and simulating data.
Author: Mohcine Chraibi
Date: August 11, 2023
"""

from pathlib import Path

import streamlit as st
from jupedsim.internal.notebook_utils import read_sqlite_file
from anim import animate
import jupedsim as jps
from src import analysis, docs
import pedpy
from src.logger_config import init_logger
from src.simutilities import (
    call_simulation,
    extract_motivation_parameters,
    plot_motivation_model,
)
from src.ui import init_sidebar, simulation_tab, ui_simulation_controls

if __name__ == "__main__":
    init_logger()
    st.sidebar.code(f"jupedsim: {jps.__version__}")
    if "data" not in st.session_state:
        st.session_state.data = {}

    if "all_files" not in st.session_state:
        st.session_state.all_files = ["files/bottleneck.json"]

    tab = init_sidebar()

    match tab:
        case "Documentation":
            docs.main()
        case "Simulation":
            c1, c2, c3 = st.columns(3)
            data = simulation_tab()
            CONFIG_FILE, OUTPUT_FILE, fps = ui_simulation_controls(data)

            if c1.button("Run Simulation"):
                call_simulation(CONFIG_FILE, OUTPUT_FILE, data)

            if c2.button("Visualisation"):
                output_path = Path(OUTPUT_FILE)
                if output_path.exists():
                    trajectory_data, walkable_area = read_sqlite_file(OUTPUT_FILE)
                    data_with_speed = pedpy.compute_individual_speed(
                        traj_data=trajectory_data,
                        frame_step=5,
                        speed_calculation=pedpy.SpeedCalculation.BORDER_SINGLE_SIDED,
                    )
                    data_with_speed = data_with_speed.merge(
                        trajectory_data.data,
                        on=["id", "frame"],
                        how="left",
                    )
                    data_with_speed["gender"] = 1
                    width = data["motivation_parameters"]["width"]
                    vertices = data["motivation_parameters"]["motivation_doors"][0][
                        "vertices"
                    ]
                    x0 = 0.5 * (vertices[0][0] + vertices[1][0]) - width
                    y0 = 0.5 * (vertices[0][1] + vertices[1][1]) - width
                    x1 = 0.5 * (vertices[0][0] + vertices[1][0]) + width
                    y1 = 0.5 * (vertices[1][1] + vertices[1][1]) + width

                    st.info(f"{x0=}, {y0=}, {x1=}, {y1=}")
                    anm = animate(
                        data_with_speed,
                        walkable_area,
                        every_nth_frame=int(fps),
                        color_mode="Speed",
                        radius=0.1,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                    )
                    st.plotly_chart(anm)

            params = extract_motivation_parameters(data)
            plot_motivation_model(params)
        case "Analysis":
            analysis.run()
        case _:
            st.warning("Selected tab is not implemented.")
