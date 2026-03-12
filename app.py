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
from anim import animate
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

            if c1.button("Run Simulation"):
                call_simulation(config_file, output_file, data)

            if c2.button("Visualization"):
                output_path = Path(output_file)
                if output_path.exists():
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
                        data_with_motivation["gender"] = 1
                        data_with_motivation["speed"] = 0.0
                        width = data["motivation_parameters"]["width"]
                        vertices = data["motivation_parameters"]["motivation_doors"][
                            0
                        ]["vertices"]
                        x0 = 0.5 * (vertices[0][0] + vertices[1][0]) - width
                        y0 = 0.5 * (vertices[0][1] + vertices[1][1]) - width
                        x1 = 0.5 * (vertices[0][0] + vertices[1][0]) + width
                        y1 = 0.5 * (vertices[1][1] + vertices[1][1]) + width

                        animation = animate(
                            data_with_motivation,
                            walkable_area,
                            every_nth_frame=int(fps),
                            color_mode="Motivation",
                            radius=0.1,
                            x0=x0,
                            y0=y0,
                            x1=x1,
                            y1=y1,
                        )
                        st.plotly_chart(animation)
                else:
                    st.warning(f"Trajectory file not found: {output_file}")

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
