"""Init ui."""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

from .utilities import delete_txt_files, load_json, save_json

TAB_INFO = [
    {"name": "Simulation", "icon": "info-square"},
    {"name": "Analysis", "icon": "bar-chart-line"},
    {"name": "Documentation", "icon": "book"},
]


def init_sidebar() -> Any:
    """Init sidebar and tabs using centralized tab info."""
    tab_names = [tab["name"] for tab in TAB_INFO]
    tab_icons = [tab["icon"] for tab in TAB_INFO]

    return option_menu(
        "",
        tab_names,
        icons=tab_icons,
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "gray", "font-size": "15px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
        },
    )


def ui_simulation_controls(data: Dict[str, Any]) -> Tuple[str, str, int]:
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

    fps = int(
        c3.number_input(
            "fps", min_value=1, max_value=32, value=4, help="Show every nth frame"
        )
    )

    return CONFIG_FILE, OUTPUT_FILE, fps


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


def ui_measurement_parameters(data: Dict[str, Any]) -> None:
    """Measurement lines, polygons."""
    with st.sidebar.expander("Measurement Parameters", expanded=True):
        st.code("Measurement line:")
        column_1, column_2 = st.columns((1, 1))
        line = data["measurement_line"]["vertices"]
        for idx, vertex in enumerate(line):
            x_key = f"l_vertex_x_{idx}"
            y_key = f"l_vertex_y_{idx}"
            vertex[0] = column_1.number_input("Point X:", value=vertex[0], key=x_key)
            vertex[1] = column_2.number_input("Point Y:", value=vertex[1], key=y_key)

        st.code("Measurement area:")
        column_1, column_2 = st.columns((1, 1))
        area = data["measurement_area"]["vertices"]
        for idx, vertex in enumerate(area):
            x_key = f"a_vertex_x_{idx}"
            y_key = f"a_vertex_y_{idx}"
            vertex[0] = column_1.number_input("Point X:", value=vertex[0], key=x_key)
            vertex[1] = column_2.number_input("Point Y:", value=vertex[1], key=y_key)


def ui_velocity_model_parameters(data: Dict[str, Any]) -> None:
    """Set velocity Parameters Section."""
    with st.sidebar.expander("Velocity model Parameters", expanded=False):
        c1, c2 = st.columns(2)
        data["velocity_init_parameters"]["a_ped"] = c1.number_input(
            "a_ped:",
            min_value=0.0,
            max_value=10.0,
            value=data["velocity_init_parameters"]["a_ped"],
        )
        data["velocity_init_parameters"]["d_ped"] = c2.number_input(
            "d_ped:",
            min_value=0.01,
            max_value=1.0,
            value=data["velocity_init_parameters"]["d_ped"],
        )
        data["velocity_init_parameters"]["a_wall"] = c1.number_input(
            "a_wall:",
            min_value=0.0,
            max_value=10.0,
            step=0.1,
            value=data["velocity_init_parameters"]["a_wall"],
        )
        data["velocity_init_parameters"]["d_wall"] = c2.number_input(
            "d_wall:",
            min_value=0.01,
            max_value=1.0,
            value=data["velocity_init_parameters"]["d_wall"],
        )


def ui_simulation_parameters(data: Dict[str, Any]) -> None:
    """Set simulation Parameters Section."""
    with st.sidebar.expander("Simulation Parameters"):
        data["simulation_parameters"]["fps"] = st.number_input(
            "FPS:",
            min_value=1,
            max_value=60,
            value=data["simulation_parameters"]["fps"],
        )
        data["simulation_parameters"]["time_step"] = st.number_input(
            "Time Step:",
            format="%f",
            value=data["simulation_parameters"]["time_step"],
            min_value=0.0001,
            max_value=0.5,
            step=0.0001,
        )
        data["simulation_parameters"]["number_agents"] = st.number_input(
            "Number of Agents:",
            value=data["simulation_parameters"]["number_agents"],
            step=10,
            min_value=2,
            max_value=150,
        )
        data["simulation_parameters"]["simulation_time"] = st.number_input(
            "Simulation Time:",
            value=data["simulation_parameters"]["simulation_time"],
            step=20,
            min_value=50,
            max_value=300,
        )


def ui_value_parameters(data: Dict[str, Any]) -> None:
    """Set Value function."""
    with st.sidebar.expander("Value Parameters", expanded=True):
        c1, c2 = st.columns(2)
        min_value_high, max_value_high = st.select_slider(
            "**Value high**",
            key="value_high",
            options=np.arange(
                1.0,  # float(data["motivation_parameters"]["min_value_high"]),
                5.0,  # float(data["motivation_parameters"]["max_value_high"]),
                0.1,
            ),
            value=[
                2.0,  # float(data["motivation_parameters"]["min_value_high"] + 0.2),
                4.0,  # float(data["motivation_parameters"]["max_value_high"] - 0.2),
            ],
            format_func=lambda x: f"{x:.2f}",
            help="Upper/Lower limit of high Value people.",
        )
        data["motivation_parameters"]["min_value_high"] = min_value_high
        data["motivation_parameters"]["max_value_high"] = max_value_high
        min_value_low, max_value_low = st.select_slider(
            "**Value low**",
            key="value_low",
            options=np.arange(
                0.1,  # float(data["motivation_parameters"]["min_value_low"]),
                1.0,  # float(data["motivation_parameters"]["max_value_low"]),
                0.1,
            ),
            value=[
                0.2,  # float(data["motivation_parameters"]["min_value_low"] + 0.2),
                0.8,  # float(data["motivation_parameters"]["max_value_low"] - 0.2),
            ],
            help="Upper/Lower limit of low Value people.",
            format_func=lambda x: f"{x:.2f}",
        )
        # st.info((min_value_low, max_value_low))
        data["motivation_parameters"]["min_value_low"] = min_value_low
        data["motivation_parameters"]["max_value_low"] = max_value_low

        data["motivation_parameters"]["number_high_value"] = st.slider(
            "Number of high **Value** people",
            key="num_high_value",
            step=1,
            min_value=0,
            max_value=int(data["simulation_parameters"]["number_agents"]),
            value=int(data["motivation_parameters"]["number_high_value"]),
            help="Number of high Value people.",
        )


def ui_competition_parameters(data: Dict[str, Any]) -> None:
    """Set competition function"""
    with st.sidebar.expander("Competition Parameters", expanded=True):
        c1, c2 = st.columns(2)
        data["motivation_parameters"]["competition_max"] = c1.number_input(
            "Competition max",
            key="comp_max",
            step=1.0,
            min_value=0.5,
            max_value=5.0,
            value=float(data["motivation_parameters"]["competition_max"]),
            help="Maximum of competition",
        )
        decay = int(data["motivation_parameters"]["competition_decay_reward"])
        if decay >= data["simulation_parameters"]["number_agents"]:
            decay = data["simulation_parameters"]["number_agents"] - 1
        data["motivation_parameters"]["competition_decay_reward"] = c2.number_input(
            "Competition decay",
            key="comp_dec",
            step=10,
            min_value=1,
            max_value=data["simulation_parameters"]["number_agents"],
            value=decay,
            help="Start of decay of competition",
        )
        data["motivation_parameters"]["percent"] = c1.number_input(
            "Competition percent",
            key="comp_perc",
            step=0.1,
            min_value=0.1,
            max_value=1.0,
            value=float(data["motivation_parameters"]["percent"]),
            help="Percent of competition max",
        )


def ui_motivation_parameters(data: Dict[str, Any]) -> None:
    """Motivation Parameters Section."""
    c1, c2 = st.sidebar.columns(2)
    motivation_strategy = st.sidebar.selectbox(
        "Select model",
        ["EVC", "default", "EC-V"],
        help="Default: M = M(dist). EVC: M = EVC, EC-V: M=(E.C).V",
    )
    data["motivation_parameters"]["normal_v_0"] = c1.number_input(
        "Normal V0:",
        min_value=0.1,
        max_value=2.5,
        value=float(data["motivation_parameters"]["normal_v_0"]),
    )
    data["motivation_parameters"]["normal_time_gap"] = c2.number_input(
        "Normal Time Gap:",
        min_value=0.1,
        max_value=3.0,
        step=0.1,
        value=float(data["motivation_parameters"]["normal_time_gap"]),
    )
    data["motivation_parameters"]["seed"] = c1.number_input(
        "Seed",
        key="seed",
        step=1.0,
        value=float(data["motivation_parameters"]["seed"]),
        help="Seed for random generator for value",
    )

    data["motivation_parameters"]["motivation_strategy"] = motivation_strategy
    if motivation_strategy == "default":
        title = "Motivation Parameters"
    else:
        title = "Expectancy Parameters"
    with st.sidebar.expander(title, expanded=True):
        c1, c2 = st.columns(2)
        data["motivation_parameters"]["width"] = c1.number_input(
            "Width",
            key="width",
            step=0.5,
            value=float(data["motivation_parameters"]["width"]),
            help="width of function defining distance dependency",
        )
        data["motivation_parameters"]["height"] = c2.number_input(
            "Height",
            key="hight",
            step=0.5,
            value=float(data["motivation_parameters"]["height"]),
            help="Height of function defining distance dependency",
        )
    if motivation_strategy != "default":
        ui_value_parameters(data)
        ui_competition_parameters(data)

    st.sidebar.write("**At this line the motivation is maximal**")
    with st.sidebar.expander(label="Motivation line"):
        column_1, column_2 = st.columns((1, 1))
        for door_idx, door in enumerate(
            data["motivation_parameters"]["motivation_doors"]
        ):
            column_2.text_input(
                "Door Label:", value=door["label"], key=f"door_label_{door_idx}"
            )
            door["id"] = column_1.number_input(
                "Door ID:", value=door["id"], key=f"door_id_{door_idx}"
            )
            for door_idx, door in enumerate(
                data["motivation_parameters"]["motivation_doors"]
            ):
                for vertex_idx, vertex in enumerate(door["vertices"]):
                    x_key = f"vertex_x_{door_idx}_{vertex_idx}"
                    y_key = f"vertex_y_{door_idx}_{vertex_idx}"
                    vertex[0] = column_1.number_input(
                        "Point X:",
                        value=float(vertex[0]),
                        key=x_key,
                        step=1.0,
                        min_value=47.0,
                        max_value=65.0,
                    )
                    vertex[1] = column_2.number_input(
                        "Point Y:",
                        value=float(vertex[1]),
                        key=y_key,
                        step=1.0,
                        min_value=100.0,
                        max_value=104.0,
                    )
