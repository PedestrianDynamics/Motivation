"""Init ui."""

from typing import Any, Dict

import streamlit as st
from streamlit_option_menu import option_menu


def init_sidebar() -> Any:
    """Init sidebar and 4 tabs."""
    return option_menu(
        "",
        ["Simulation", "Analysis"],
        icons=[
            "info-square",
            "pin-map",
            "bar-chart-line",
            "exclamation-triangle",
        ],
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
            "Time Step:", value=data["simulation_parameters"]["time_step"]
        )
        data["simulation_parameters"]["number_agents"] = st.number_input(
            "Number of Agents:", value=data["simulation_parameters"]["number_agents"]
        )
        data["simulation_parameters"]["simulation_time"] = st.number_input(
            "Simulation Time:", value=data["simulation_parameters"]["simulation_time"]
        )
        data["motivation_parameters"]["seed"] = st.text_input(
            "Seed",
            key="seed",
            value=float(data["motivation_parameters"]["seed"]),
            help="Seed for random generator for value",
        )


def ui_motivation_parameters(data: Dict[str, Any]) -> None:
    """Motivation Parameters Section."""
    with st.sidebar.expander("Motivation Parameters", expanded=True):
        act = st.empty()
        model = st.empty()
        c1, c2 = st.columns(2)
        motivation_activated = act.checkbox("Activate motivation", value=True)
        if motivation_activated:
            data["motivation_parameters"]["active"] = 1
        else:
            data["motivation_parameters"]["active"] = 0

        motivation_strategy = model.selectbox(
            "Select model",
            ["default", "EVC"],
            help="Model 2: M = M(dist). Model 3: M = V.E, Model4: M=V.E.C",
        )
        data["motivation_parameters"]["width"] = c1.text_input(
            "Width",
            key="width",
            value=float(data["motivation_parameters"]["width"]),
            help="width of function defining distance dependency",
        )
        data["motivation_parameters"]["height"] = c2.text_input(
            "Height",
            key="hight",
            value=float(data["motivation_parameters"]["height"]),
            help="Height of function defining distance dependency",
        )

        data["motivation_parameters"]["max_value"] = c1.number_input(
            "Max_value",
            key="max_value",
            value=float(data["motivation_parameters"]["max_value"]),
            help="Max Value",
        )

        data["motivation_parameters"]["min_value"] = c2.number_input(
            "Min_value",
            key="min_value",
            value=float(data["motivation_parameters"]["min_value"]),
            help="Min Value",
        )

        data["motivation_parameters"]["motivation_strategy"] = motivation_strategy
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

    with st.sidebar.expander(label="Door"):
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
                        "Point X:", value=vertex[0], key=x_key
                    )
                    vertex[1] = column_2.number_input(
                        "Point Y:", value=vertex[1], key=y_key
                    )


def ui_grid_parameters(data: Dict[str, Any]) -> None:
    """Grid Parameters Section."""
    with st.expander("Grid Parameters"):
        column_1, column_2, column_3 = st.columns((1, 1, 1))
        data["grid_parameters"]["min_v_0"] = column_1.number_input(
            "Min V0:", value=data["grid_parameters"]["min_v_0"]
        )
        data["grid_parameters"]["max_v_0"] = column_2.number_input(
            "Max V0:", value=data["grid_parameters"]["max_v_0"]
        )
        data["grid_parameters"]["v_0_step"] = column_3.number_input(
            "V0 Step:", value=data["grid_parameters"]["v_0_step"]
        )
        data["grid_parameters"]["min_time_gap"] = column_1.number_input(
            "Min Time Gap:", value=data["grid_parameters"]["min_time_gap"]
        )
        data["grid_parameters"]["max_time_gap"] = column_2.number_input(
            "Max Time Gap:", value=data["grid_parameters"]["max_time_gap"]
        )
        data["grid_parameters"]["time_gap_step"] = column_3.number_input(
            "Time Gap Step:", value=data["grid_parameters"]["time_gap_step"]
        )