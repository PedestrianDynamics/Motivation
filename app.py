"""Motivation model with jupedsim.

Description: This module contains functions for visualizing and simulating data.
Author: Mohcine Chraibi
Date: August 11, 2023
"""

from pathlib import Path

import streamlit as st
from jupedsim.internal.notebook_utils import animate, read_sqlite_file

from src import analysis, docs
from src.logger_config import init_logger
from src.simutilities import (call_simulation, extract_motivation_parameters,
                              plot_motivation_model)
from src.ui import init_sidebar, simulation_tab, ui_simulation_controls

if __name__ == "__main__":
    init_logger()
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
                    anm = animate(
                        trajectory_data, walkable_area, every_nth_frame=int(fps)
                    )
                    st.plotly_chart(anm)

            params = extract_motivation_parameters(data)
            plot_motivation_model(params)
        case "Analysis":
            analysis.run()
        case _:
            st.warning("Selected tab is not implemented.")
