"""Run simulations and all analysis scripts for every scenario."""

from __future__ import annotations

import subprocess
from pathlib import Path

from run_coordination_scenarios import (
    BASE_MODELS,
    NUMBER_AGENTS_VALUES,
    OPEN_DOOR_TIMES,
    ROOT,
    build_scenario_configs,
)


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT)


def run_simulations(configs_dir: Path, base_runs_dir: Path) -> None:
    base_runs_dir.mkdir(parents=True, exist_ok=True)
    for model in BASE_MODELS:
        config_path = configs_dir / f"base_{model}.json"
        run_command(
            [
                "python3",
                "simulation.py",
                "--inifile",
                str(config_path),
                "--output-dir",
                str(base_runs_dir),
            ]
        )


def main() -> None:
    for number_agents in NUMBER_AGENTS_VALUES:
        for open_door_time in OPEN_DOOR_TIMES:
            scenario_dir = build_scenario_configs(number_agents, open_door_time)
            scenario_name = scenario_dir.name
            configs_dir = scenario_dir / "base_files"
            base_runs_dir = scenario_dir / "base_runs"

            print(f"Running full analysis for {scenario_name}")
            run_simulations(configs_dir, base_runs_dir)

            run_command(
                [
                    "python3",
                    "scripts/coordination_number_analysis.py",
                    "--models",
                    "PVE",
                    "BASE_MODEL",
                    "--search-dir",
                    str(base_runs_dir),
                    "--t-min",
                    "10",
                    "--t-max",
                    "300",
                    "--output-dir",
                    str(scenario_dir / "coordination_number_results"),
                    "--tag",
                    scenario_name,
                ]
            )

            run_command(
                [
                    "python3",
                    "scripts/motivation_heatmap_analysis.py",
                    "--models",
                    "P",
                    "V",
                    "SE",
                    "PVE",
                    "BASE_MODEL",
                    "--search-dir",
                    str(base_runs_dir),
                    "--t-min",
                    "10",
                    "--t-max",
                    "300",
                    "--output-dir",
                    str(scenario_dir / "motivation_heatmap_results"),
                    "--tag",
                    scenario_name,
                ]
            )

            run_command(
                [
                    "python3",
                    "scripts/crossing_density_analysis.py",
                    "--models",
                    "P",
                    "V",
                    "SE",
                    "PVE",
                    "BASE_MODEL",
                    "--search-dir",
                    str(base_runs_dir),
                    "--output-dir",
                    str(scenario_dir / "crossing_density_results"),
                    "--tag",
                    scenario_name,
                ]
            )

            run_command(
                [
                    "python3",
                    "scripts/final_rank_analysis.py",
                    "--models",
                    "P",
                    "V",
                    "SE",
                    "PVE",
                    "BASE_MODEL",
                    "--search-dir",
                    str(base_runs_dir),
                    "--output-dir",
                    str(scenario_dir / "final_rank_results"),
                    "--tag",
                    scenario_name,
                ]
            )

            run_command(
                [
                    "python3",
                    "scripts/voronoi_density_analysis.py",
                    "--models",
                    "PVE",
                    "BASE_MODEL",
                    "--search-dir",
                    str(base_runs_dir),
                    "--t-min",
                    "10",
                    "--t-max",
                    "300",
                    "--output-dir",
                    str(scenario_dir / "voronoi_density_results"),
                    "--tag",
                    scenario_name,
                ]
            )


if __name__ == "__main__":
    main()
