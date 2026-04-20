"""Utilities for simulation."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st

from src import motivation_mapping as mmap
from src import motivation_model as mm
from src.inifile_parser import (
    parse_fps,
    parse_motivation_doors,
    parse_motivation_strategy,
    parse_normal_time_gap,
    parse_normal_v_0,
    parse_number_agents,
    parse_number_high_value,
    parse_simulation_time,
    parse_time_step,
    parse_velocity_init_parameters,
)

Point = Tuple[float, float]


def _preview_agent_positions(data: Dict[str, Any]) -> Tuple[List[Point], int]:
    """Load deterministic preview positions for app plots without simulation runtime."""
    n_agents = parse_number_agents(data)
    init_file = data.get("init_trajectories_file")
    if init_file:
        candidate_paths = [
            Path(str(init_file)),
            Path(__file__).resolve().parents[1] / str(init_file),
        ]
        path = next((p for p in candidate_paths if p.exists()), None)
        if path is not None:
            with path.open("r", encoding="utf8") as f:
                rows: List[Tuple[int, int, float, float]] = []
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    try:
                        pid = int(float(parts[0]))
                        frame = int(float(parts[1]))
                        x = float(parts[2])
                        y = float(parts[3])
                    except ValueError:
                        continue
                    rows.append((pid, frame, x, y))
            if rows:
                first_frame = min(r[1] for r in rows)
                seen: set[int] = set()
                positions: List[Point] = []
                for pid, frame, x, y in sorted(rows, key=lambda r: (r[1], r[0])):
                    if frame != first_frame or pid in seen:
                        continue
                    seen.add(pid)
                    positions.append((x, y))
                    if len(positions) >= n_agents:
                        break
                if positions:
                    if len(positions) < n_agents:
                        positions.extend(
                            [
                                (float(i % 10), float(i // 10))
                                for i in range(len(positions), n_agents)
                            ]
                        )
                    return positions[:n_agents], n_agents

    # Fallback deterministic synthetic positions for plotting.
    positions = [(float(i % 10), float(i // 10)) for i in range(n_agents)]
    return positions, n_agents


def extract_motivation_parameters(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and convert motivation parameters from the data dictionary.

    Args:
        data: The data dictionary containing motivation parameters.

    Returns:
        A dictionary of extracted and converted motivation parameters.
    """
    params = data["motivation_parameters"]
    payoff = params["payoff"]
    extracted_params = {
        "strategy": params["motivation_mode"],
        "width": float(params["width"]),
        "height": float(params["height"]),
        "max_value_high": float(params["max_value_high"]),
        "min_value_high": float(params["min_value_high"]),
        "max_value_low": float(params["max_value_low"]),
        "min_value_low": float(params["min_value_low"]),
        "number_high_value": float(params["number_high_value"]),
        "seed": params["seed"],
        "payoff_k": float(payoff["k"]),
        "payoff_q0": float(payoff["q0"]),
        "rank_tie_tolerance_m": float(payoff["rank_tie_tolerance_m"]),
        "payoff_update_interval_s": float(payoff["update_interval_s"]),
    }
    mapping_block = mmap.ensure_mapping_block(params)
    extracted_params["mapping_block"] = mapping_block
    extracted_params["normal_v_0"] = parse_normal_v_0(data)
    extracted_params["normal_time_gap"] = parse_normal_time_gap(data)
    extracted_params["number_high_value_agents"] = parse_number_high_value(data)
    extracted_params["time_step"] = parse_time_step(data)
    extracted_params["motivation_doors"] = parse_motivation_doors(data)
    a_ped, d_ped, _a_wall, _d_wall = parse_velocity_init_parameters(data)
    extracted_params["a_ped"] = a_ped
    extracted_params["d_ped"] = d_ped
    # preview positions for plotting only
    positions, num_agents = _preview_agent_positions(data)
    extracted_params["number_agents"] = num_agents
    extracted_params["positions"] = positions
    return extracted_params


def create_motivation_strategy(params: Dict[str, Any]) -> mm.MotivationStrategy:
    """Create and return the appropriate motivation strategy based on the given parameters.

    Args:
        params: A dictionary of motivation parameters including the strategy type and other relevant data.

    Returns:
        An instance of a motivation strategy.
    """
    strategy = params["strategy"]
    if strategy in ["E", "SE", "V", "P", "PVE", "BASE_MODEL"]:
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
        strategy_obj = mm.EVPStrategy(
            width=params["width"],
            height=params["height"],
            max_reward=params["number_agents"],
            seed=params["seed"],
            max_value_high=params["max_value_high"],
            min_value_high=params["min_value_high"],
            max_value_low=params["max_value_low"],
            min_value_low=params["min_value_low"],
            number_high_value=params["number_high_value_agents"],
            agent_ids=list(range(params["number_agents"])),
            nagents=params["number_agents"],
            agent_positions=params["positions"],
            motivation_door_center=motivation_door_center,
            motivation_mode=strategy,
            payoff_k=params["payoff_k"],
            payoff_q0=params["payoff_q0"],
            rank_tie_tolerance_m=params["rank_tie_tolerance_m"],
            payoff_update_interval_s=params["payoff_update_interval_s"],
            motivation_min=float(params["mapping_block"]["motivation_min"]),
        )
        strategy_obj.configure_payoff_update_interval(
            time_step=float(params["time_step"])
        )
        strategy_obj.update_payoff_cache(
            iteration_count=0,
            agent_positions={
                agent_id: pos
                for agent_id, pos in zip(
                    list(range(params["number_agents"])), params["positions"]
                )
            },
            number_agents_in_simulation=int(params["number_agents"]),
        )
        return strategy_obj
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def call_simulation(config_file: str, output_file: str, data: Dict[str, Any]) -> None:
    """Run the simulation based on the provided configuration and data."""
    from simulation import init_and_run_simulation

    msg = st.empty()
    output_path = Path(output_file)
    if output_path.exists():
        output_path.unlink()

    number_agents = parse_number_agents(data)
    fps = parse_fps(data)
    time_step = parse_time_step(data)
    simulation_time = parse_simulation_time(data)
    open_door_time = float(data["simulation_parameters"].get("open_door_time", 0.0))
    strategy = parse_motivation_strategy(data)

    logging.info(f"Call simulation {number_agents = }")
    msg.code(f"Running simulation with {number_agents}. Strategy: <{strategy}>...")

    with st.spinner("Simulating..."):
        try:
            evac_time, _ = init_and_run_simulation(
                fps,
                time_step,
                simulation_time,
                open_door_time,
                data,
                output_path,
                msg,
            )
        except ValueError as exc:
            st.error(f"Logistic configuration error: {exc}")
            return

    msg.code(f"Finished simulation. Evac time {evac_time:.2f} s")


def plot_motivation_model(params: Dict[str, Any]) -> None:
    """Plot the motivation model based on the given parameters.

    Args:
        params: A dictionary of parameters required for creating and plotting the motivation model.
    """
    strategy = create_motivation_strategy(params)
    mapper = None
    if params["strategy"] != "BASE_MODEL":
        try:
            mapper = mmap.MotivationParameterMapper(
                mapping_block=params["mapping_block"],
                normal_v_0=params["normal_v_0"],
                range_default=params["d_ped"],
            )
        except ValueError as exc:
            st.error(f"Logistic configuration error: {exc}")
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
    with st.expander("Plot model", expanded=True):
        for fig in figs:
            st.pyplot(fig)
