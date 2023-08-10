import json
import pathlib as p
import subprocess
from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt
import plotly.express as px
import streamlit as st
from scipy.interpolate import griddata
import plotly.graph_objects as go
from src import inifile_parser as parser
from scipy import spatial, stats
import pandas as pd


# pd.DataFrame(data, columns=header)


def moving_trajectories(config_file, output_file):
    data_df = pd.read_csv(
        output_file,
        sep=r"\s+",
        dtype=np.float64,
        comment="#",
        names=["ID", "FR", "X", "Y", "Z", "A", "B", "P", "PP"],
    )
    st.dataframe(data_df)

    if "SPEED" in data_df.columns:
        color = "SPEED"
        range_color = [0, max(data_df["SPEED"])]
    elif "COLOR" in data_df.columns:
        color = "COLOR"
        range_color = [0, 255]
    else:
        data_df["COLOR"] = 125
        color = "COLOR"
        range_color = [0, 125]

    if "A" in data_df.columns:
        data_df["A"] /= 2
    else:
        data_df["A"] = 0.2

    fig = px.scatter(
        data_df,
        x="X",
        y="Y",
        animation_frame="FR",
        animation_group="ID",
        color=color,
        size="A",
        range_color=range_color,
        color_continuous_scale=px.colors.diverging.RdBu_r[::-1],
    )

    with open(config_file, "r", encoding="utf8") as f:
        json_str = f.read()
        data = json.loads(json_str)
        polygons = parser.parse_accessible_areas(data)
        geominX = min([point[0] for polygon in polygons.values() for point in polygon])
        geomaxX = max([point[0] for polygon in polygons.values() for point in polygon])
        geominY = min([point[1] for polygon in polygons.values() for point in polygon])
        geomaxY = max([point[1] for polygon in polygons.values() for point in polygon])

    fig.update_xaxes(
        range=[
            geominX,
            geomaxX,
        ]
    )
    fig.update_yaxes(
        range=[
            geominY,
            geomaxY,
        ]
    )
    for polygon in polygons.values():
        x_values = [point[0] for point in polygon] + [
            polygon[0][0]
        ]  # Add the first point again to close the polygon
        y_values = [point[1] for point in polygon] + [polygon[0][1]]
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                # fill="toself",
                line=dict(color="grey"),
            )
        )

    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 30
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 5
    fig.update_geos(projection_type="equirectangular", visible=True, resolution=110)
    fig.update_traces(marker=dict(line=dict(width=0.5, color="Gray")))
    fig.update_geos(projection_type="equirectangular", visible=True, resolution=110)
    fig.update_layout(title="Visualisation", showlegend=False)
    st.plotly_chart(fig)


def heatmap(
    config_file, position_x: npt.NDArray, position_y: npt.NDArray, value: npt.NDArray
) -> None:
    # Create grid
    grid_x, grid_y = np.mgrid[
        min(position_x) : max(position_x) : 100j,
        min(position_y) : max(position_y) : 100j,
    ]
    grid_z = griddata((position_x, position_y), value, (grid_x, grid_y), method="cubic")

    # # Create Plotly figure
    # fig = px.imshow(
    #     grid_z.T,
    #     color_continuous_scale="viridis",
    #     labels={"x": "Position X", "y": "Position Y", "color": "Value"},
    # )
    # TODO: parse the geometry from json file: Accessible areas.
    with open(config_file, "r", encoding="utf8") as f:
        json_str = f.read()
        data = json.loads(json_str)
        polygons = parser.parse_accessible_areas(data)
        fig = go.Figure(go.Scatter(x=[], y=[], mode="markers", marker=dict(size=0)))
        geominX = min([point[0] for polygon in polygons.values() for point in polygon])
        geomaxX = max([point[0] for polygon in polygons.values() for point in polygon])
        geominY = min([point[1] for polygon in polygons.values() for point in polygon])
        geomaxY = max([point[1] for polygon in polygons.values() for point in polygon])

        fig.update_xaxes(
            range=[
                geominX,
                geomaxX,
            ]
        )
        fig.update_yaxes(
            range=[
                geominY,
                geomaxY,
            ]
        )

        dx = st.slider(label="grid size", min_value=0.1, max_value=1.0, step=0.1)
        dy = dx
        xbins = np.arange(geominX, geomaxX + dx, dx)
        ybins = np.arange(geominY, geomaxY + dy, dy)
        area = dx * dy
        ret = stats.binned_statistic_2d(
            position_x,
            position_y,
            value,
            "mean",
            bins=[xbins, ybins],
        )
        zz = np.array(np.nan_to_num(ret.statistic.T)) / area

        fig.add_trace(
            go.Heatmap(
                x=xbins,
                y=ybins,
                z=zz,
                zmin=0,
                zmax=0.5,
                connectgaps=False,
                zsmooth="best",
                colorscale="Jet",
                colorbar=dict(title="Motivation"),
            )
        )

    # Add polygons to the plot
    for polygon in polygons.values():
        x_values = [point[0] for point in polygon] + [
            polygon[0][0]
        ]  # Add the first point again to close the polygon
        y_values = [point[1] for point in polygon] + [polygon[0][1]]
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                # fill="toself",
                line=dict(color="white"),
            )
        )

    fig.update_layout(title="Heatmap", showlegend=False)

    # Display in Streamlit
    st.plotly_chart(fig)


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

        with c1:
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
            n_agents = st.session_state.data["simulation_parameters"]["number_agents"]
            with st.spinner(f"Simulating with {n_agents}"):
                process = subprocess.Popen(
                    command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                stdout, stderr = process.communicate()
                info_output = stdout.decode().replace("\n", "  \n")
                warnings = stderr.decode().replace("\n", "  \n")
                st.info(info_output)
                if warnings:
                    st.error(warnings)

            moving_trajectories(config_file, output_file)

        if p.Path("values.txt").exists():
            values = np.loadtxt("values.txt")
            if values.any():
                heatmap(config_file, values[:, 0], values[:, 1], values[:, 2])
