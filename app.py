"""Motivation model with jupedsim.

Description: This module contains functions for visualizing and simulating data.
Author: Mohcine Chraibi
Date: August 11, 2023
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from jupedsim.internal.notebook_utils import animate, read_sqlite_file


from typing import Dict, Tuple, Any
import simulation
from src import docs
from src import motivation_model as mm
from src import analysis
from src.inifile_parser import (
    parse_fps,
    parse_motivation_doors,
    parse_motivation_strategy,
    parse_normal_time_gap,
    parse_normal_v_0,
    parse_number_agents,
    parse_simulation_time,
    parse_time_step,
)
from src.logger_config import init_logger
from src.ui import (
    init_sidebar,
    ui_motivation_parameters,
    ui_simulation_parameters,
    ui_velocity_model_parameters,
)
from src.utilities import delete_txt_files, load_json, save_json


tab_actions = {
    "Documentation": docs.main,
    # "Simulation":,
    "Analysis": analysis.run,
}


def ui_load_save_config() -> Tuple[str, str]:
    """Display UI elements for loading and saving configurations.

    Returns:
        A tuple containing the file names for loading and saving configurations.
    """
    with st.sidebar.expander("Save/load config"):
        column_1, column_2 = st.columns((1, 1))
        file_name = str(
            column_1.selectbox(
                "Load", sorted(list(set(st.session_state.all_files)), reverse=True)
            )
        )
        new_json_name = column_2.text_input(
            "Save", help="Save config file: ", value="files/bottleneck2.json"
        )

        if column_2.button("Delete files", help="Delete all trajectory files"):
            delete_txt_files()

        return file_name, new_json_name


def simulation_tab() -> pd.DataFrame:
    """Handle the Simulation tab."""
    file_name, new_json_name = ui_load_save_config()

    json_file = Path(file_name)
    if not json_file.exists():
        st.error(f"File: {file_name} does not exist!")
        st.stop()

    data = load_json(json_file)

    # Display UI for simulation parameters
    ui_velocity_model_parameters(data)
    ui_simulation_parameters(data)
    ui_motivation_parameters(data)

    save_json(Path(new_json_name), data)

    st.session_state.all_files.append(file_name)
    st.session_state.all_files.append(new_json_name)
    return data


def ui_simulation_controls(data: dict) -> Tuple[str, str, int]:
    """
    Display UI elements for controlling the simulation.

    Including a configuration file,
    specifying an output file, and setting the frames per second (fps).

    Args:
        data: The data dictionary containing simulation parameters, specifically used to determine
              the strategy for naming the output file.

    Returns:
        A tuple containing the path to the configuration file selected by the user, the output file
        name as specified by the user, and the fps setting as an integer.
    """
    # Setup UI elements for simulation control
    c1, c2, c3 = st.columns(3)
    CONFIG_FILE = str(
        c2.selectbox(
            "Select config file",
            sorted(list(set(st.session_state.all_files)), reverse=True),
        )
    )

    # Extract strategy from the loaded data for naming the output file
    strategy = data.get("motivation_parameters", {}).get(
        "motivation_strategy", "default"
    )
    name, extension = CONFIG_FILE.rsplit(".", 1)
    sqlite_filename = f"{name}_{strategy}.{extension.replace('json', 'sqlite')}"

    OUTPUT_FILE = c1.text_input("Result: ", value=sqlite_filename)

    fps = c3.number_input(
        "fps", min_value=1, max_value=32, value=8, help="Show every nth frame"
    )

    return CONFIG_FILE, OUTPUT_FILE, fps


def extract_motivation_parameters(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and convert motivation parameters from the data dictionary.

    Args:
        data: The data dictionary containing motivation parameters.

    Returns:
        A dictionary of extracted and converted motivation parameters.
    """
    params = data["motivation_parameters"]
    extracted_params = {
        "strategy": params["motivation_strategy"],
        "width": float(params["width"]),
        "height": float(params["height"]),
        "max_value_high": float(params["max_value_high"]),
        "min_value_high": float(params["min_value_high"]),
        "max_value_low": float(params["max_value_low"]),
        "min_value_low": float(params["min_value_low"]),
        "number_high_value": int(params["number_high_value"]),
        "seed": params["seed"],
        "competition_max": params["competition_max"],
        "competition_decay_reward": params["competition_decay_reward"],
        "percent": params["percent"],
        # Add more parameters as needed
    }
    extracted_params["number_agents"] = int(parse_number_agents(data))
    extracted_params["normal_v_0"] = parse_normal_v_0(data)
    extracted_params["normal_time_gap"] = parse_normal_time_gap(data)
    extracted_params["motivation_doors"] = parse_motivation_doors(data)
    return extracted_params


def create_motivation_strategy(params: Dict[str, Any]) -> mm.MotivationStrategy:
    """Create and return the appropriate motivation strategy based on the given parameters.

    Args:
        params: A dictionary of motivation parameters including the strategy type and other relevant data.

    Returns:
        An instance of a motivation strategy.
    """
    strategy = params["strategy"]
    if strategy == "default":
        return mm.DefaultMotivationStrategy(
            width=params["width"], height=params["height"]
        )
    elif strategy in ["EVC", "EC-V"]:
        return mm.EVCStrategy(
            width=params["width"],
            height=params["height"],
            max_reward=params["number_agents"],
            seed=params["seed"],
            max_value_high=params["max_value_high"],
            min_value_high=params["min_value_high"],
            max_value_low=params["max_value_low"],
            min_value_low=params["min_value_low"],
            number_high_value=params["number_high_value"],
            agent_ids=list(range(params["number_agents"])),
            competition_decay_reward=params["competition_decay_reward"],
            competition_max=params["competition_max"],
            percent=params["percent"],
            evc=strategy == "EVC",
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def plot_motivation_model(params: Dict[str, Any]) -> None:
    """Plot the motivation model based on the given parameters.

    Args:
        params: A dictionary of parameters required for creating and plotting the motivation model.
    """
    strategy = create_motivation_strategy(params)
    motivation_model = mm.MotivationModel(
        door_point1=(
            params["motivation_doors"][0][0][0],
            params["motivation_doors"][0][0][1],
        ),
        door_point2=(
            params["motivation_doors"][0][1][0],
            params["motivation_doors"][0][1][1],
        ),
        normal_v_0=params["normal_v_0"],
        normal_time_gap=params["normal_time_gap"],
        motivation_strategy=strategy,
    )
    figs = motivation_model.motivation_strategy.plot()
    if params["strategy"] != "default":
        fig1, fig2 = motivation_model.plot()
        figs.extend([fig1, fig2])
    with st.expander("Plot model", expanded=True):
        for fig in figs:
            st.pyplot(fig)


def call_simulation(config_file: str, output_file: str, data: dict) -> None:
    """Run the simulation based on the provided configuration and data.

    Args:
        config_file: Path to the configuration file.
        output_file: Desired path for the simulation output file.
        data: Data dictionary containing simulation parameters.
    """
    msg = st.empty()
    if Path(output_file).exists():
        Path(output_file).unlink()

    # Load configuration data again in case it's been updated
    with open(config_file, "r", encoding="utf8") as f:
        data = json.loads(f.read())

    fps = parse_fps(data)
    time_step = parse_time_step(data)
    number_agents = parse_number_agents(data)
    simulation_time = parse_simulation_time(data)
    strategy = parse_motivation_strategy(data)

    msg.code(f"Running simulation with {number_agents}. Strategy: <{strategy}>...")

    with st.spinner("Simulating..."):
        evac_time = simulation.main(
            number_agents, fps, time_step, simulation_time, data, Path(output_file), msg
        )

    msg.code(f"Finished simulation. Evac time {evac_time:.2f} s")


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
