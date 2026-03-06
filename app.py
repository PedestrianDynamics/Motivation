"""Motivation model with jupedsim.

Description: This module contains functions for visualizing and simulating data.
Author: Mohcine Chraibi
Date: August 11, 2023
"""

import jupedsim as jps
import streamlit as st
import pedpy
from src import analysis, docs
import glob
from src.logger_config import init_logger
from src.simutilities import (
    extract_motivation_parameters,
    plot_motivation_model,
)
from src.ui import init_sidebar, simulation_tab, ui_simulation_controls

if __name__ == "__main__":
    init_logger()
    st.sidebar.code(f"jupedsim: {jps.__version__}\npedpy: {pedpy.__version__}")
    if "data" not in st.session_state:
        st.session_state.data = {}

    if "all_files" not in st.session_state:
        st.session_state.all_files = glob.glob(
            "files/*.json"
        )  # ["files/inifile.json", "files/bottleneck.json"]

    tab = init_sidebar()

    match tab:
        case "Documentation":
            docs.main()
        case "Simulation":
            c1, c2, c3 = st.columns(3)
            data = simulation_tab()
            CONFIG_FILE, OUTPUT_FILE, fps = ui_simulation_controls(data)

            if c1.button("Run Simulation"):
                st.warning(
                    "Run parameter studies via simulation.py (CLI). "
                    "The app is kept for documentation and model-parameter inspection."
                )

            if c2.button("Visualization"):
                st.warning(
                    "Animation is disabled in the app for performance. "
                    "Use offline scripts and simulation.py outputs instead."
                )

            params = extract_motivation_parameters(data)
            mapping = params.get("mapping_block", {})
            if mapping:
                m_min = float(mapping.get("motivation_min", 0.1))
                m_max = 3.6 / float(params["normal_v_0"])
                st.caption(
                    "Motivation mapping: "
                    f"{mapping.get('mapping_function', 'logistic')} "
                    f"(clamp [{m_min:.2f}, {m_max:.2f}])"
                )

            # get_agents_positions
            plot_motivation_model(params)
        case "Analysis":
            analysis.run()
        case _:
            st.warning("Selected tab is not implemented.")
