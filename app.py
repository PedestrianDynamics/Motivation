"""Motivation model with jupedsim.

Description: This module contains functions for visualizing and simulating data.
Author: Mohcine Chraibi
Date: August 11, 2023
"""

from pathlib import Path

import jupedsim as jps
import pandas as pd
import pedpy
import streamlit as st
from anim import plot_frame_fast
from jupedsim.internal.notebook_utils import read_sqlite_file
from src import docs
import glob
from src.logger_config import init_logger
from src.simutilities import (
    call_simulation,
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
            c1, c2, _c3 = st.columns(3)
            data = simulation_tab()
            config_file, output_file, fps = ui_simulation_controls(data)
            if "show_frame_viewer" not in st.session_state:
                st.session_state.show_frame_viewer = False

            if c1.button("Run Simulation"):
                call_simulation(config_file, output_file, data)

            if c2.button("Visualization"):
                st.session_state.show_frame_viewer = True

            if st.session_state.show_frame_viewer:
                output_path = Path(output_file)
                if not output_path.exists():
                    st.warning(f"Trajectory file not found: {output_file}")
                else:
                    trajectory_data, walkable_area = read_sqlite_file(output_file)
                    motivation_path = output_path.with_name(
                        output_path.stem + "_motivation.csv"
                    )
                    if not motivation_path.exists():
                        st.warning(f"Motivation file not found: {motivation_path}")
                    else:
                        trajectory_df = trajectory_data.data.copy()
                        motivation_df = pd.read_csv(motivation_path)
                        data_with_motivation = trajectory_df.merge(
                            motivation_df[["frame", "id", "motivation"]],
                            on=["frame", "id"],
                            how="left",
                        )
                        data_with_motivation["motivation"] = data_with_motivation[
                            "motivation"
                        ].fillna(1.0)
                        frames = sorted(data_with_motivation["frame"].unique().tolist())
                        frame_step = max(1, int(fps))
                        selected_frame = st.slider(
                            "Frame",
                            min_value=int(frames[0]),
                            max_value=int(frames[-1]),
                            value=int(frames[0]),
                            step=frame_step,
                        )
                        fig = plot_frame_fast(
                            data_with_motivation,
                            walkable_area,
                            frame_num=selected_frame,
                            radius=0.1,
                        )
                        st.pyplot(fig)

            params = extract_motivation_parameters(data)
            mapping = params.get("mapping_block", {})
            if mapping:
                m_min = float(mapping.get("motivation_min", 0.1))
                m_max = 1.0
                st.caption(
                    "Motivation mapping: "
                    f"{mapping.get('mapping_function', 'logistic')} "
                    f"(clamp [{m_min:.2f}, {m_max:.2f}])"
                )

            plot_motivation_model(params)
        case _:
            st.warning("Selected tab is not implemented.")
