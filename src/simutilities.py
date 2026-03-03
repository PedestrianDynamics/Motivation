"""Utilities for simulation."""

import json
from pathlib import Path
from typing import Any, Dict

import streamlit as st
import logging
from simulation import get_agent_positions, init_and_run_simulation
from src import motivation_model as mm
from src import motivation_mapping as mmap
from src.inifile_parser import (
    parse_fps,
    parse_motivation_doors,
    parse_motivation_strategy,
    parse_normal_time_gap,
    parse_normal_v_0,
    parse_number_agents,
    parse_simulation_time,
    parse_time_step,
    parse_velocity_init_parameters,
)


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
    }
    mapping_block = mmap.ensure_mapping_block(params)
    extracted_params["mapping_block"] = mapping_block
    extracted_params["normal_v_0"] = parse_normal_v_0(data)
    extracted_params["normal_time_gap"] = parse_normal_time_gap(data)
    extracted_params["motivation_doors"] = parse_motivation_doors(data)
    (
        a_ped,
        d_ped,
        _a_wall,
        _d_wall,
        a_ped_min,
        a_ped_max,
        _d_ped_min,
        _d_ped_max,
    ) = parse_velocity_init_parameters(data)
    extracted_params["a_ped"] = a_ped
    extracted_params["d_ped"] = d_ped
    extracted_params["a_ped_min"] = a_ped_min
    extracted_params["a_ped_max"] = a_ped_max
    # calculate positions
    positions, num_agents = get_agent_positions(data)
    extracted_params["number_agents"] = num_agents
    extracted_params["positions"] = positions
    return extracted_params


def call_simulation(config_file: str, output_file: str, data: Dict[str, Any]) -> None:
    """Run the simulation based on the provided configuration and data.

    Args:
        config_file: Path to the configuration file.
        output_file: Desired path for the simulation output file.
        data: Data dictionary containing simulation parameters.
    """
    msg = st.empty()
    if Path(output_file).exists():
        Path(output_file).unlink()
    logging.info(f"in call simulation {data['simulation_parameters']['number_agents']}")
    logging.info(f"in call simulation {data['simulation_parameters']['number_agents']}")
    number_agents = parse_number_agents(data)
    logging.info(f"Call simulation {number_agents = }")
    fps = parse_fps(data)
    time_step = parse_time_step(data)
    simulation_time = parse_simulation_time(data)
    open_door_time = float(data["simulation_parameters"].get("open_door_time", 0.0))
    strategy = parse_motivation_strategy(data)

    msg.code(f"Running simulation with {number_agents}. Strategy: <{strategy}>...")

    with st.spinner("Simulating..."):
        try:
            evac_time, _ = init_and_run_simulation(
                fps,
                time_step,
                simulation_time,
                open_door_time,
                data,
                Path(output_file),
                msg,
            )
        except ValueError as exc:
            st.error(f"Gompertz configuration error: {exc}")
            return

    msg.code(f"Finished simulation. Evac time {evac_time:.2f} s")


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
        door_point1 = (
            params["motivation_doors"][0][0][0],
            params["motivation_doors"][0][0][1],
        )
        door_point2 = (
            params["motivation_doors"][0][1][0],
            params["motivation_doors"][0][1][1],
        )
        x_door = 0.5 * (door_point1[0] + door_point2[0])
        y_door = 0.5 * (door_point1[1] + door_point2[1])
        motivation_door_center = (x_door, y_door)
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
            nagents=params["number_agents"],
            agent_positions=params["positions"],
            motivation_door_center=motivation_door_center,
            competition_decay_reward=params["competition_decay_reward"],
            competition_max=params["competition_max"],
            percent=params["percent"],
            evc=strategy == "EVC",
            motivation_min=float(params["mapping_block"]["motivation_min"]),
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def plot_motivation_model(params: Dict[str, Any]) -> None:
    """Plot the motivation model based on the given parameters.

    Args:
        params: A dictionary of parameters required for creating and plotting the motivation model.
    """
    strategy = create_motivation_strategy(params)
    try:
        mapper = mmap.MotivationParameterMapper(
            mapping_block=params["mapping_block"],
            normal_v_0=params["normal_v_0"],
            strength_default=params["a_ped"],
            strength_min=params["a_ped_min"],
            strength_max=params["a_ped_max"],
            range_default=params["d_ped"],
        )
    except ValueError as exc:
        st.error(f"Gompertz configuration error: {exc}")
        return

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
        parameter_mapper=mapper,
    )
    figs = motivation_model.motivation_strategy.plot()
    if params["strategy"] != "default":
        fig1, fig2 = motivation_model.plot()
        figs.extend([fig1, fig2])
    if motivation_model.parameter_mapper is not None:
        figs.append(
            mmap.plot_parameter_mappings(
                mapper=motivation_model.parameter_mapper,
                normal_time_gap=params["normal_time_gap"],
            )
        )
    with st.expander("Plot model", expanded=False):
        for fig in figs:
            st.pyplot(fig)
