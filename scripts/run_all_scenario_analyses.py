"""Run simulations and all analysis scripts for every scenario, across seeds."""

from __future__ import annotations

import subprocess
from pathlib import Path

from run_coordination_scenarios import (
    BASE_MODELS,
    NUMBER_AGENTS_VALUES,
    OPEN_DOOR_TIMES,
    ROOT,
    SEEDS,
    build_scenario_configs,
)


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT)


def run_simulation_for_seed(
    configs_dir: Path, base_runs_dir: Path, model: str, seed: int
) -> None:
    config_path = configs_dir / f"base_{model}_seed{seed}.json"
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


def find_seed_sqlite(base_runs_dir: Path, model: str, seed: int) -> Path:
    """Locate the sqlite output produced by a single (model, seed) run.

    simulation.py writes outputs as
    ``{inifile.stem}_{motivation_mode}_{timestamp}.sqlite``, which here
    becomes ``base_{model}_seed{seed}_{motivation_mode}_*.sqlite``.
    """
    pattern = f"base_{model}_seed{seed}_*.sqlite"
    candidates = sorted(base_runs_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No sqlite output found for model={model} seed={seed} "
            f"in {base_runs_dir} (pattern={pattern})"
        )
    # If multiple runs exist for the same seed, pick the most recent.
    return max(candidates, key=lambda path: path.stat().st_mtime)


def find_seed_motivation_csv(base_runs_dir: Path, model: str, seed: int) -> Path:
    """Locate the motivation.csv output for a (model, seed) run."""
    pattern = f"base_{model}_seed{seed}_*_motivation.csv"
    candidates = sorted(base_runs_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No motivation csv found for model={model} seed={seed} "
            f"in {base_runs_dir} (pattern={pattern})"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_input_overrides(
    locator, base_runs_dir: Path, models: list[str], seed: int
) -> list[str]:
    overrides: list[str] = []
    for model in models:
        path = locator(base_runs_dir, model, seed)
        overrides.extend(["--input", f"{model}={path}"])
    return overrides


def run_seed_analyses(
    scenario_dir: Path,
    scenario_name: str,
    base_runs_dir: Path,
    seed: int,
) -> None:
    seed_tag = f"{scenario_name}_seed{seed}"

    # Coordination number: PVE + BASE_MODEL
    run_command(
        [
            "python3",
            "scripts/coordination_number_analysis.py",
            "--models",
            "PVE",
            "BASE_MODEL",
            *build_input_overrides(
                find_seed_motivation_csv, base_runs_dir, ["PVE", "BASE_MODEL"], seed
            ),
            "--t-min",
            "10",
            "--t-max",
            "300",
            "--output-dir",
            str(scenario_dir / "coordination_number_results"),
            "--tag",
            seed_tag,
        ]
    )

    # Motivation heatmaps: all five models
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
            *build_input_overrides(
                find_seed_motivation_csv,
                base_runs_dir,
                ["P", "V", "SE", "PVE", "BASE_MODEL"],
                seed,
            ),
            "--t-min",
            "10",
            "--t-max",
            "300",
            "--output-dir",
            str(scenario_dir / "motivation_heatmap_results"),
            "--tag",
            seed_tag,
        ]
    )

    # Final rank: all five models
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
            *build_input_overrides(
                find_seed_sqlite,
                base_runs_dir,
                ["P", "V", "SE", "PVE", "BASE_MODEL"],
                seed,
            ),
            "--output-dir",
            str(scenario_dir / "final_rank_results"),
            "--tag",
            seed_tag,
        ]
    )

    # Voronoi density: PVE + BASE_MODEL
    run_command(
        [
            "python3",
            "scripts/voronoi_density_analysis.py",
            "--models",
            "PVE",
            "BASE_MODEL",
            *build_input_overrides(
                find_seed_sqlite, base_runs_dir, ["PVE", "BASE_MODEL"], seed
            ),
            "--t-min",
            "0",
            "--t-max",
            "100",
            "--output-dir",
            str(scenario_dir / "voronoi_density_results"),
            "--tag",
            seed_tag,
        ]
    )


def main() -> None:
    for number_agents in NUMBER_AGENTS_VALUES:
        for open_door_time in OPEN_DOOR_TIMES:
            scenario_dir = build_scenario_configs(number_agents, open_door_time)
            scenario_name = scenario_dir.name
            configs_dir = scenario_dir / "base_files"
            base_runs_dir = scenario_dir / "base_runs"
            base_runs_dir.mkdir(parents=True, exist_ok=True)

            print(f"Running multi-seed analysis for {scenario_name}")

            for seed in SEEDS:
                print(f"  seed={seed}: running simulations")
                for model in BASE_MODELS:
                    run_simulation_for_seed(configs_dir, base_runs_dir, model, seed)
                print(f"  seed={seed}: running analyses")
                run_seed_analyses(scenario_dir, scenario_name, base_runs_dir, seed)


if __name__ == "__main__":
    main()
