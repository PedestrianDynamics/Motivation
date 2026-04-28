"""Init ui."""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

from . import motivation_mapping as mmap
from .utilities import delete_txt_files, load_json, save_json

TAB_INFO = [
    {"name": "Simulation", "icon": "info-square"},
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
    config_files = sorted(list(set(st.session_state.all_files)), reverse=True)
    default_config = "files/base.json"
    default_index = (
        config_files.index(default_config) if default_config in config_files else 0
    )
    CONFIG_FILE = str(
        c2.selectbox(
            "Select config file",
            config_files,
            index=default_index,
        )
    )

    # Extract strategy from the loaded data for naming the output file
    strategy = data.get("motivation_parameters", {}).get("motivation_mode", "PVE")
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
        file_name = str(column_1.selectbox("Load", "files/base.json"))

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
        # data["velocity_init_parameters"]["a_ped"] = c1.number_input(
        #     "a_ped:",
        #     min_value=0.0,
        #     max_value=10.0,
        #     value=data["velocity_init_parameters"]["a_ped"],
        # )
        # data["velocity_init_parameters"]["d_ped"] = c2.number_input(
        #     "d_ped:",
        #     min_value=0.01,
        #     max_value=1.0,
        #     value=data["velocity_init_parameters"]["d_ped"],
        # )
        data["velocity_init_parameters"]["a_ped"] = c1.number_input(
            "a_ped (normal):",
            min_value=0.0,
            max_value=10.0,
            value=data["velocity_init_parameters"]["a_ped"],
        )
        data["velocity_init_parameters"]["d_ped"] = c2.number_input(
            "d_ped (constant range):",
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
        data["velocity_init_parameters"]["theta_max_upper_bound"] = c1.number_input(
            "theta_max_upper_bound [rad]:",
            min_value=0.0,
            max_value=3.141592653589793,
            value=float(
                data["velocity_init_parameters"].get("theta_max_upper_bound", 1.0)
            ),
            step=0.01,
        )


def ui_simulation_parameters(data: Dict[str, Any]) -> None:
    """Set simulation Parameters Section."""
    with st.sidebar.expander("Simulation Parameters"):
        data["simulation_parameters"]["fps"] = st.number_input(
            "FPS:",
            min_value=1,
            max_value=100,
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
            max_value=3000,
        )


def ui_value_parameters(data: Dict[str, Any]) -> None:
    """Set Value function."""
    with st.sidebar.expander("Value Parameters", expanded=True):
        value_options = [round(float(v), 2) for v in np.arange(1.0, 7.1, 0.1)]

        def _closest_option(options: list[float], value: float) -> float:
            return min(options, key=lambda opt: abs(opt - value))

        v_min = _closest_option(
            value_options, float(data["motivation_parameters"]["min_value"])
        )
        v_max = _closest_option(
            value_options, float(data["motivation_parameters"]["max_value"])
        )
        if v_min > v_max:
            v_min, v_max = v_max, v_min

        min_value, max_value = st.select_slider(
            "**Value range**",
            key="value_range",
            options=value_options,
            value=[v_min, v_max],
            help="Lower/Upper limit of the uniform value distribution.",
            format_func=lambda x: f"{x:.2f}",
        )
        data["motivation_parameters"]["min_value"] = min_value
        data["motivation_parameters"]["max_value"] = max_value


def ui_payoff_parameters(data: Dict[str, Any]) -> None:
    """Set payoff function parameters."""
    with st.sidebar.expander("Payoff Parameters", expanded=True):
        payoff = data["motivation_parameters"].setdefault("payoff", {})
        c1, c2 = st.columns(2)
        payoff["k"] = c1.number_input(
            "Payoff k",
            step=0.5,
            min_value=0.1,
            max_value=50.0,
            value=float(payoff.get("k", 8.0)),
            help="Steepness of payoff logistic over normalized rank q.",
        )
        payoff["q0"] = c2.number_input(
            "Payoff q0",
            step=0.05,
            min_value=0.0,
            max_value=1.0,
            value=float(payoff.get("q0", 0.5)),
            help="Inflection point of payoff logistic.",
        )
        payoff["rank_tie_tolerance_m"] = c1.number_input(
            "Rank tie tol [m]",
            step=0.0005,
            min_value=0.0001,
            max_value=0.1,
            value=float(payoff.get("rank_tie_tolerance_m", 0.001)),
            format="%.4f",
            help="Distance tolerance for equal rank assignment.",
        )
        payoff["update_interval_s"] = c2.number_input(
            "Rank update [s]",
            step=0.1,
            min_value=0.01,
            max_value=10.0,
            value=float(payoff.get("update_interval_s", 1.0)),
            help="Recompute ranks/payoff every N seconds.",
        )


def ui_mapping_parameters(data: Dict[str, Any]) -> None:
    """Set motivation-to-parameter mapping options."""
    params = data["motivation_parameters"]
    mapping = mmap.ensure_mapping_block(params)

    with st.sidebar.expander("Motivation Mapping", expanded=True):
        mapping["mapping_function"] = st.selectbox(
            "Mapping function",
            ["logistic"],
            index=0,
            help="Function used to map motivation to model parameters.",
        )
        mapping["motivation_min"] = st.number_input(
            "Motivation min",
            min_value=0.0,
            max_value=1.0,
            value=float(mapping["motivation_min"]),
            step=0.01,
            help="Lower clamp for motivation.",
        )
        mapping["inflection_target"] = st.number_input(
            "Inflection target (m)",
            min_value=0.0,
            max_value=1.0,
            value=float(mapping.get("inflection_target", 0.5)),
            step=0.1,
            help="Logistic midpoint parameter m0.",
        )

        st.markdown("**Desired speed anchors**")
        c1, c2, c3 = st.columns(3)
        mapping["desired_speed_anchors"]["low"] = c1.number_input(
            "v0 low",
            min_value=0.0,
            max_value=5.0,
            value=float(mapping["desired_speed_anchors"]["low"]),
            step=0.05,
        )
        mapping["desired_speed_anchors"]["normal"] = c2.number_input(
            "v0 normal",
            min_value=0.0,
            max_value=5.0,
            value=float(mapping["desired_speed_anchors"]["normal"]),
            step=0.05,
        )
        mapping["desired_speed_anchors"]["high"] = c3.number_input(
            "v0 high",
            min_value=0.0,
            max_value=6.0,
            value=float(mapping["desired_speed_anchors"]["high"]),
            step=0.05,
        )

        st.markdown("**Time gap anchors**")
        c1, c2, c3 = st.columns(3)
        mapping["time_gap_anchors"]["low"] = c1.number_input(
            "T low",
            min_value=0.0,
            max_value=10.0,
            value=float(mapping["time_gap_anchors"]["low"]),
            step=0.01,
        )
        mapping["time_gap_anchors"]["normal"] = c2.number_input(
            "T normal",
            min_value=0.0,
            max_value=10.0,
            value=float(mapping["time_gap_anchors"]["normal"]),
            step=0.01,
        )
        mapping["time_gap_anchors"]["high"] = c3.number_input(
            "T high",
            min_value=0.0,
            max_value=10.0,
            value=float(mapping["time_gap_anchors"]["high"]),
            step=0.01,
        )

        st.markdown("**Buffer anchors**")
        c1, c2, c3 = st.columns(3)
        mapping["buffer_anchors"]["low"] = c1.number_input(
            "b low",
            min_value=0.0,
            max_value=3.0,
            value=float(mapping["buffer_anchors"]["low"]),
            step=0.01,
        )
        mapping["buffer_anchors"]["normal"] = c2.number_input(
            "b normal",
            min_value=0.0,
            max_value=3.0,
            value=float(mapping["buffer_anchors"]["normal"]),
            step=0.01,
        )
        mapping["buffer_anchors"]["high"] = c3.number_input(
            "b high",
            min_value=0.0,
            max_value=3.0,
            value=float(mapping["buffer_anchors"]["high"]),
            step=0.01,
        )

        st.markdown("**Strength neighbor repulsion anchors**")
        c1, c2, c3 = st.columns(3)
        mapping["strength_neighbor_repulsion_anchors"]["low"] = c1.number_input(
            "A low",
            min_value=0.0,
            max_value=10.0,
            value=float(mapping["strength_neighbor_repulsion_anchors"]["low"]),
            step=0.01,
        )
        mapping["strength_neighbor_repulsion_anchors"]["normal"] = c2.number_input(
            "A normal",
            min_value=0.0,
            max_value=10.0,
            value=float(mapping["strength_neighbor_repulsion_anchors"]["normal"]),
            step=0.01,
        )
        mapping["strength_neighbor_repulsion_anchors"]["high"] = c3.number_input(
            "A high",
            min_value=0.0,
            max_value=10.0,
            value=float(mapping["strength_neighbor_repulsion_anchors"]["high"]),
            step=0.01,
        )
        mapping["use_manual_logistic_k"] = st.checkbox(
            "Use manual logistic k",
            value=bool(mapping.get("use_manual_logistic_k", False)),
            help="Override fitted k values per parameter while keeping low/high anchors.",
        )
        mapping.setdefault("logistic_k", {})
        if mapping["use_manual_logistic_k"]:
            st.markdown("**Manual k values**")
            c1, c2 = st.columns(2)
            mapping["logistic_k"]["desired_speed"] = c1.number_input(
                "k(v0)",
                value=float(mapping["logistic_k"].get("desired_speed", 10.0)),
                step=0.1,
                format="%.3f",
            )
            mapping["logistic_k"]["time_gap"] = c2.number_input(
                "k(T)",
                value=float(mapping["logistic_k"].get("time_gap", 10.0)),
                step=0.1,
                format="%.3f",
            )
            c1, c2 = st.columns(2)
            mapping["logistic_k"]["buffer"] = c1.number_input(
                "k(buffer)",
                value=float(mapping["logistic_k"].get("buffer", 10.0)),
                step=0.1,
                format="%.3f",
            )
            mapping["logistic_k"]["strength_neighbor_repulsion"] = c2.number_input(
                "k(A)",
                value=float(
                    mapping["logistic_k"].get("strength_neighbor_repulsion", 10.0)
                ),
                step=0.1,
                format="%.3f",
            )

        vparams = data["velocity_init_parameters"]
        try:
            mapper = mmap.MotivationParameterMapper(
                mapping_block=mapping,
                normal_v_0=float(params["normal_v_0"]),
                range_default=float(vparams["d_ped"]),
            )
            st.markdown("**Fitted logistic k**")
            for key, label in (
                ("desired_speed", "k(v0)"),
                ("time_gap", "k(T)"),
                ("buffer", "k(buffer)"),
                ("strength_neighbor_repulsion", "k(A)"),
            ):
                st.caption(f"{label} = {mapper.logistic_parameters[key].k:.2f}")
        except ValueError as exc:
            st.error(f"Logistic config error: {exc}")

    params.update(mapping)


def ui_motivation_parameters(data: Dict[str, Any]) -> None:
    """Motivation Parameters Section."""
    params = data["motivation_parameters"]
    c1, c2 = st.sidebar.columns(2)
    motivation_mode = st.sidebar.selectbox(
        "Select mode",
        ["PVE", "SE", "V", "P", "BASE_MODEL"],
        help="PVE: (V/alpha) * (ES + P); SE: spatial expectancy only; V: value only, P: payoff only, BASE_MODEL: keep base CFSM parameters.",
    )
    data["motivation_parameters"]["normal_v_0"] = c1.number_input(
        "Normal V0:",
        min_value=0.1,
        max_value=2.5,
        value=float(params["normal_v_0"]),
    )
    data["motivation_parameters"]["normal_time_gap"] = c2.number_input(
        "Normal Time Gap:",
        min_value=0.1,
        max_value=3.0,
        step=0.1,
        value=float(params["normal_time_gap"]),
    )
    data["motivation_parameters"]["seed"] = c1.number_input(
        "Seed",
        key="seed",
        step=1.0,
        value=float(params["seed"]),
        help="Seed for random generator for value",
    )

    data["motivation_parameters"]["motivation_mode"] = motivation_mode
    title = "Spatial Expectancy Parameters"
    with st.sidebar.expander(title, expanded=True):
        c1, c2 = st.columns(2)
        data["motivation_parameters"]["width"] = c1.number_input(
            "Width",
            key="width",
            step=0.5,
            value=float(params["width"]),
            help="width of function defining distance dependency",
        )
        data["motivation_parameters"]["height"] = c2.number_input(
            "Height",
            key="hight",
            step=0.5,
            value=float(params["height"]),
            help="Height of the ES function; ES is floored at 0.1 to avoid zero motivation.",
        )
    ui_value_parameters(data)
    ui_payoff_parameters(data)
    ui_mapping_parameters(data)

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
                        min_value=-3.0,
                        max_value=3.0,
                    )
                    vertex[1] = column_2.number_input(
                        "Point Y:",
                        value=float(vertex[1]),
                        key=y_key,
                        step=1.0,
                    )
