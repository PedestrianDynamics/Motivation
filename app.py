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

from simulation import main
from src import motivation_model as mm
from src.analysis import run
from src.inifile_parser import (
    parse_fps,
    parse_motivation_strategy,
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
        st.session_state.all_files = ["files/bottleneck.json"]

    tab = init_sidebar()
    if tab == "Documentation":
        st.markdown("""## Default Strategy""")
        st.markdown(r"""
        $
        \textbf{motivation}(distance) = 
        \begin{cases}
        0 & \text{if\;} distance  \geq \text{width}, \\
        e \cdot \text{height}\cdot\exp\left(\frac{1}{\left(\frac{distance}{\text{width}}\right)^2 - 1}\right) & \text{otherwise}.
        \end{cases}
        $
        
        ---
        ---
        """)
        st.markdown(r"""
        ## EVC
        $\textbf{motivation} = E\cdot V\cdot C,$ where
        - $E$: expectancy
        - $V$: value
        - $C$: competition
        
        ---
        """)

        st.markdown(
            r"""
        $
        \textbf{expectancy}(distance) = 
        \begin{cases}
        0 & \text{if\;} distance  \geq \text{width}, \\
        e \cdot \text{height}\cdot\exp\left(\frac{1}{\left(\frac{distance}{\text{width}}\right)^2 - 1}\right) & \text{otherwise}.
        \end{cases}\\
        $
        
        **Note:** this is the same function like default strategy
        
        ---
        $$
        \textbf{competition} = 
        \begin{cases} 
        c_0 & \text{if } N \leq N_0 \\
        c_0 - \left(\frac{c_0}{\text{percent} \cdot N_{\text{max}} - N_0}\right) \cdot (N - N_0) & \text{if } N_0 < N < \text{percent} \cdot N_{\text{max}} \\
        0 & \text{if } N \geq \text{percent} \cdot N_{\text{max}},
        \end{cases}
        $$
        with:

        | Parameter    | Meaning|
        |--------------|:-----:|
        |$N$ | Agents still in the simulation|
        |$N_0$ | Number of agents at which the decay of the function starts.|
        |$N_{\max}$ | The initial number of agents in the simulation|
        |$c_0$ | Maximal competition|
        |$p$ | is a percentage number $\in [0, 1]$.|
        
        ---
        $\textbf{value} = random\_number \in [v_{\min}, v_{\max}].$
        
        ## Update agents
        For an agent $i$ we calculate $m_i$ by one of the methods above and update its parameters as follows:
        
        $\tilde v_i^0 =  v_i^0(1 + m_i)\cdot V_i$ and $\tilde T_i = \frac{T_i}{1 + m_i}$

        The first part of the equation is equalivalent to

        $\tilde v_i^0 =  v_i^0(1 + E_i\cdot V_i\cdot C_i)\cdot V_i$.

        Here we see that the influence of $V_i$ is squared.
        Therefore, the second variation of the model reads
            
        ## EC-V
       $$     
       \begin{cases}     
        \tilde v_i^0 =  v_i^0(1 + E_i\cdot C_i)\cdot V_i, \\
        \tilde T_i = \frac{T_i}{1 + E_i\cdot C_i}.
       \end{cases}
        $$
        """,
            unsafe_allow_html=True,
        )

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
        st.session_state.all_files.append(new_json_name)

        if column_2.button("Delete files", help="Delete all trajectory files"):
            delete_txt_files()

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
            with open(CONFIG_FILE, "r", encoding="utf8") as f:
                json_str = f.read()
                data = json.loads(json_str)
                fps = parse_fps(data)
                time_step = parse_time_step(data)
                number_agents = parse_number_agents(data)
                simulation_time = parse_simulation_time(data)
                strategy = parse_motivation_strategy(data)

            msg.code(
                f"Running simulation with {number_agents}. Strategy: <{strategy}>..."
            )

            with st.spinner("Simulating..."):
                if fps and time_step:
                    evac_time = main(
                        number_agents,
                        fps,
                        time_step,
                        simulation_time,
                        data,
                        Path(OUTPUT_FILE),
                        msg,
                    )
            msg.code(f"Finished simulation. Evac time {evac_time:.2f} s")
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

        if True or c3.button("Plot"):
            strategy = data["motivation_parameters"]["motivation_strategy"]
            width = float(data["motivation_parameters"]["width"])
            height = float(data["motivation_parameters"]["height"])
            max_value = float(data["motivation_parameters"]["max_value"])
            min_value = float(data["motivation_parameters"]["min_value"])
            seed = data["motivation_parameters"]["seed"]
            competition_max = data["motivation_parameters"]["competition_max"]
            competition_decay_reward = data["motivation_parameters"][
                "competition_decay_reward"
            ]
            percent = data["motivation_parameters"]["percent"]
            number_agents = int(parse_number_agents(data))
            st.info(number_agents)
            motivation_strategy: mm.MotivationStrategy
            if strategy == "default":
                motivation_strategy = mm.DefaultMotivationStrategy(
                    width=width, height=height
                )
            if strategy == "EVC":
                motivation_strategy = mm.EVCStrategy(
                    width=width,
                    height=height,
                    max_reward=number_agents,
                    seed=seed,
                    max_value=max_value,
                    min_value=min_value,
                    agent_ids=range(number_agents),
                    competition_decay_reward=competition_decay_reward,
                    competition_max=competition_max,
                    percent=percent,
                    evc=True,
                )
            if strategy == "EC-V":
                motivation_strategy = mm.EVCStrategy(
                    width=width,
                    height=height,
                    max_reward=number_agents,
                    seed=seed,
                    max_value=max_value,
                    min_value=min_value,
                    competition_decay_reward=competition_decay_reward,
                    competition_max=competition_max,
                    percent=percent,
                    agent_ids=range(number_agents),
                    evc=False,
                )

            figs = motivation_strategy.plot()
            with st.expander("Plot model", expanded=True):
                for fig in figs:
                    st.pyplot(fig)

    if tab == "Analysis":
        run()
