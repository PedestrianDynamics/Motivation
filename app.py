import streamlit as st
import json
import subprocess
from typing import Dict, List, Any
import pathlib as p


def load_json(filename: p.Path):
    """load json file"""

    with open(filename, "r") as file:
        data = json.load(file)
    return data


def save_json(output: p.Path, data: Dict[str, Any]):
    """save data in json file"""
    with open(output, "w") as file:
        json.dump(data, file, indent=4)


def ui_simulation_parameters(data: Dict[str, Any]) -> None:
    """ "Simulation Parameters Section"""
    with st.expander("Simulation Parameters"):
        data["simulation_parameters"]["fps"] = st.slider(
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


def ui_motivation_parameters(data: Dict[str, Any]) -> None:
    """Motivation Parameters Section"""
    with st.expander("Motivation Parameters"):
        motivation_activated = False
        if data["motivation_parameters"]["active"]:
            motivation_activated = st.checkbox("Activate motivation")

        if motivation_activated:
            data["motivation_parameters"]["active"] = 1
        else:
            data["motivation_parameters"]["active"] = 0

        data["motivation_parameters"]["normal_v_0"] = st.slider(
            "Normal V0:",
            min_value=0.5,
            max_value=2.5,
            value=float(data["motivation_parameters"]["normal_v_0"]),
        )
        data["motivation_parameters"]["normal_time_gap"] = st.slider(
            "Normal Time Gap:",
            min_value=0.1,
            max_value=3.0,
            step=0.1,
            value=float(data["motivation_parameters"]["normal_time_gap"]),
        )
        c1, c2 = st.columns((1, 1))
        for door_idx, door in enumerate(
            data["motivation_parameters"]["motivation_doors"]
        ):
            c2.text_input(
                "Door Label:", value=door["label"], key=f"door_label_{door_idx}"
            )
            door["id"] = c1.number_input(
                "Door ID:", value=door["id"], key=f"door_id_{door_idx}"
            )
            for door_idx, door in enumerate(
                data["motivation_parameters"]["motivation_doors"]
            ):
                for vertex_idx, vertex in enumerate(door["vertices"]):
                    x_key = f"vertex_x_{door_idx}_{vertex_idx}"
                    y_key = f"vertex_y_{door_idx}_{vertex_idx}"
                    vertex[0] = c1.number_input("Point X:", value=vertex[0], key=x_key)
                    vertex[1] = c2.number_input("Point Y:", value=vertex[1], key=y_key)


def ui_grid_parameters(data: Dict[str, Any]) -> None:
    """Grid Parameters Section"""

    with st.expander("Grid Parameters"):
        c1, c2, c3 = st.columns((1, 1, 1))
        data["grid_parameters"]["min_v_0"] = c1.number_input(
            "Min V0:", value=data["grid_parameters"]["min_v_0"]
        )
        data["grid_parameters"]["max_v_0"] = c2.number_input(
            "Max V0:", value=data["grid_parameters"]["max_v_0"]
        )
        data["grid_parameters"]["v_0_step"] = c3.number_input(
            "V0 Step:", value=data["grid_parameters"]["v_0_step"]
        )
        data["grid_parameters"]["min_time_gap"] = c1.number_input(
            "Min Time Gap:", value=data["grid_parameters"]["min_time_gap"]
        )
        data["grid_parameters"]["max_time_gap"] = c2.number_input(
            "Max Time Gap:", value=data["grid_parameters"]["max_time_gap"]
        )
        data["grid_parameters"]["time_gap_step"] = c3.number_input(
            "Time Gap Step:", value=data["grid_parameters"]["time_gap_step"]
        )


if __name__ == "__main__":
    if "data" not in st.session_state:
        st.session_state.data = {}

    if "all_files" not in st.session_state:
        st.session_state.all_files = []
        # User will select from these files to do simulations

    tab1, tab2 = st.tabs(["Initialisation", "Simulation"])

    with tab1:
        c1, c2 = st.columns((1, 1))
        file_name = c1.text_input("Load config file: ", value="files/bottleneck.json")
        json_file = p.Path(file_name)
        data = {}
        if not json_file.exists():
            st.error(f"file: {file_name} does not exist!")
            st.stop()

        if c1.button(
            "Load config",
            help=f"Load config file ({file_name})",
        ):
            data = load_json(json_file)
            ui_simulation_parameters(data)
            ui_motivation_parameters(data)
            ui_grid_parameters(data)
            st.session_state.data = data
            st.session_state.all_files.append(file_name)

        # Save Button (optional)
        new_json_name = c2.text_input(
            "Save config file: ", value="files/bottleneck.json"
        )
        new_json_file = p.Path(new_json_name)

        if c2.button(
            "Save config",
            help=f"After changing the values, you can save the configs in a separate file ({new_json_name})",
        ):
            save_json(new_json_file, data)
            c1.info(f"Saved file as {new_json_name}")
            st.session_state.all_files.append(new_json_name)

    # Run Simulation
    with tab2:
        output_file = st.text_input("Result: ", value="files/trajectory.txt")
        config_file = st.selectbox("Select config file", st.session_state.all_files)
        if st.button("Run Simulation"):
            # Modify the command as needed

            command = f"python simulation.py {config_file} {output_file}"
            process = subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            info_output = stdout.decode().replace("\n", "  \n")

            warnings = stderr.decode().replace("\n", "  \n")
            if warnings:
                st.warning(warnings)
