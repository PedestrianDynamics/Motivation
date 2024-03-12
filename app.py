"""Motivation model with jupedsim.

Description: This module contains functions for visualizing and simulating data.
Author: Mohcine Chraibi
Date: August 11, 2023
"""

import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pedpy
import streamlit as st
from jupedsim.internal.notebook_utils import animate, read_sqlite_file
from src.logger_config import init_logger
import simulation
from src.inifile_parser import (
    parse_fps,
    parse_time_step,
    parse_number_agents,
    parse_simulation_time,
)
from src.ui import (
    ui_motivation_parameters,
    ui_simulation_parameters,
    ui_velocity_model_parameters,
    init_sidebar,
)
from src.utilities import delete_txt_files, load_json, save_json
from src.analysis import run
import jupedsim as jps


def read_data(output_file: str) -> pd.DataFrame:
    """Read data from csv file.

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


if __name__ == "__main__":
    init_logger()
    if "data" not in st.session_state:
        st.session_state.data = {}

    if "all_files" not in st.session_state:
        st.session_state.all_files = []
        # User will select from these files to do simulations

    tab = init_sidebar()

    # tab1, tab2, tab3 = st.tabs(["Initialisation", "Simulation", "Analysis"])
    # st.sidebar.info(f"{jps.__version__ = }")
    # st.sidebar.info(f"{pedpy.__version__ = }")

    if tab == "Simulation":
        with st.sidebar.expander("Save/load config"):
            column_1, column_2 = st.columns((1, 1))

        file_name = str(
            column_1.selectbox(
                "Load",
                sorted(list(set(st.session_state.all_files)), reverse=True),
            )
        )

        # file_name = column_1.text_input(
        #     "Load", value="files/bottleneck.json", help="Load config file"
        # )
        json_file = Path(file_name)
        data = {}
        if not json_file.exists():
            st.error(f"file: {file_name} does not exist!")
            st.stop()

        # with column_1:
        data = load_json(json_file)
        ui_velocity_model_parameters(data)
        ui_simulation_parameters(data)
        ui_motivation_parameters(data)
        st.session_state.data = data
        st.session_state.all_files.append(file_name)

        # Save Button (optional)
        new_json_name = column_2.text_input(
            "Save", help="Save config file: ", value="files/bottleneck2.json"
        )
        new_json_file = Path(new_json_name)
        save_json(new_json_file, data)
        # if column_1.button(
        #    "Save config",
        #    help=f"After changing the values, you can save the configs in a separate file ({new_json_name})",
        # ):
        #    save_json(new_json_file, data)
        #    st.sidebar.info(f"Saved file as {new_json_name}")
        st.session_state.all_files.append(new_json_name)

        if column_2.button("Delete files", help="Delete all trajectory files"):
            delete_txt_files()

    # Run Simulation
    if tab == "Simulation":
        c1, c2, c3 = st.columns(3)
        msg = st.empty()
        CONFIG_FILE = str(
            c2.selectbox(
                "Select config file",
                sorted(list(set(st.session_state.all_files)), reverse=True),
            )
        )
        strategy = data["motivation_parameters"]["motivation_strategy"]
        name, extension = CONFIG_FILE.rsplit(".", 1)
        sqlite_filename = f"{name}_{strategy}.{extension.replace('json', 'sqlite')}"
        OUTPUT_FILE = c1.text_input("Result: ", value=f"{sqlite_filename}")

        fps = c3.number_input(
            "fps", min_value=1, max_value=32, value=8, help="show every nth frame"
        )
        if c1.button("Run Simulation"):
            if Path(OUTPUT_FILE).exists():
                Path(OUTPUT_FILE).unlink()
            msg.empty()
            msg.code("Running simulation ...")
            with open(CONFIG_FILE, "r", encoding="utf8") as f:
                json_str = f.read()
                data = json.loads(json_str)
                fps = parse_fps(data)
                time_step = parse_time_step(data)
                number_agents = parse_number_agents(data)
                simulation_time = parse_simulation_time(data)

            with st.spinner("Simulating ..."):
                if fps and time_step:
                    simulation.main(
                        number_agents,
                        fps,
                        time_step,
                        simulation_time,
                        data,
                        Path(OUTPUT_FILE),
                    )
            msg.code("Finished simulation")
            st.empty()
        output_path = Path(OUTPUT_FILE)
        if Path("values.txt").exists():
            print(output_path.name)
            Path("values.txt").rename(output_path.name + "_Heatmap.txt")
        if c2.button("Visualisation"):
            if output_path.exists():
                trajectory_data, walkable_area = read_sqlite_file(OUTPUT_FILE)
                # moving_trajectories(CONFIG_FILE, trajectory_data)
                anm = animate(trajectory_data, walkable_area, every_nth_frame=int(fps))
                st.plotly_chart(anm)

    if tab == "Analysis":
        run()
