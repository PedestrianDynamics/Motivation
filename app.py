"""Motivation model with jupedsim.

Description: This module contains functions for visualizing and simulating data.
Author: Mohcine Chraibi
Date: August 11, 2023
"""

import glob
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pedpy
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from jupedsim.internal.notebook_utils import animate, read_sqlite_file
from pedpy.column_identifier import (
    CUMULATED_COL,
    DENSITY_COL,
    FRAME_COL,
    ID_COL,
    TIME_COL,
)
from scipy import stats
from shapely import Polygon
from shapely.ops import unary_union


from src.ui import (
    ui_motivation_parameters,
    ui_simulation_parameters,
    ui_velocity_model_parameters,
)
from src.utilities import delete_txt_files, load_json, save_json
from src.analysis import run
import jupedsim as jps
import pedpy


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
    if "data" not in st.session_state:
        st.session_state.data = {}

    if "all_files" not in st.session_state:
        st.session_state.all_files = []
        # User will select from these files to do simulations

    tab1, tab2, tab3 = st.tabs(["Initialisation", "Simulation", "Analysis"])
    st.sidebar.info(f"{jps.__version__ = }")
    st.sidebar.info(f"{pedpy.__version__ = }")
    with tab1:
        column_1, column_2 = st.columns((1, 1))
        file_name = column_1.text_input(
            "Load config file: ", value="files/bottleneck.json"
        )
        json_file = Path(file_name)
        data = {}
        if not json_file.exists():
            st.error(f"file: {file_name} does not exist!")
            st.stop()

        with column_1:
            data = load_json(json_file)
            ui_velocity_model_parameters(data)
            ui_simulation_parameters(data)
            ui_motivation_parameters(data)
            st.session_state.data = data
            st.session_state.all_files.append(file_name)

        # Save Button (optional)
        new_json_name = column_2.text_input(
            "Save config file: ", value="files/bottleneck.json"
        )
        new_json_file = Path(new_json_name)
        if column_2.button(
            "Save config",
            help=f"After changing the values, you can save the configs in a separate file ({new_json_name})",
        ):
            save_json(new_json_file, data)
            column_1.info(f"Saved file as {new_json_name}")
            st.session_state.all_files.append(new_json_name)

        if column_2.button("Reset", help="Delete all trajectory files"):
            delete_txt_files()

    # Run Simulation
    with tab2:
        msg = st.sidebar.empty()
        c1, c2, c3 = st.columns(3)
        OUTPUT_FILE = c1.text_input("Result: ", value="files/trajectory.sqlite")
        CONFIG_FILE = str(
            c2.selectbox("Select config file", list(set(st.session_state.all_files)))
        )
        fps = c3.number_input(
            "fps", min_value=1, max_value=32, value=1, help="show every nth frame"
        )
        if c1.button("Run Simulation"):
            if Path(OUTPUT_FILE).exists():
                Path(OUTPUT_FILE).unlink()
            msg.empty()
            msg.info("Running simulation ...")
            command = f"python simulation.py {CONFIG_FILE} {OUTPUT_FILE}"
            n_agents = st.session_state.data["simulation_parameters"]["number_agents"]
            with st.spinner(f"Simulating with {n_agents}"):
                with subprocess.Popen(
                    command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                ) as process:
                    stdout, stderr = process.communicate()
                INFO_OUTPUT = stdout.decode().replace("\n", "  \n")
                WARNINGS = stderr.decode().replace("\n", "  \n")
                msg.code(INFO_OUTPUT)
                if WARNINGS:
                    msg.error(WARNINGS)

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

    with tab3:
        # measure flow
        activate_tab3 = st.toggle("Activate", value=False)
        if activate_tab3 and output_path.exists():
            trajectory_data, walkable_area = read_sqlite_file(OUTPUT_FILE)
            run(data, CONFIG_FILE)
