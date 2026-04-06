"""Generate and run coordination-number scenarios across agent counts and door times."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
FILES_DIR = ROOT / "files"
SCENARIOS_DIR = FILES_DIR / "coordination_scenarios"
RUNNER = ROOT / "scripts" / "run_all_base_coordination.sh"
BASE_MODELS = ["P", "V", "SE", "PVE", "BASE_MODEL"]
NUMBER_AGENTS_VALUES = [40, 80]
OPEN_DOOR_TIMES = [100]
SEEDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]


def load_json(path: Path) -> dict:  # type: ignore[type-arg]
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict) -> None:  # type: ignore[type-arg]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)


def iter_base_files() -> Iterable[tuple[str, Path]]:
    for model in BASE_MODELS:
        yield model, FILES_DIR / f"base_{model}.json"


def build_scenario_configs(number_agents: int, open_door_time: int) -> Path:
    scenario_name = f"agents_{number_agents}_open_{open_door_time}"
    scenario_dir = SCENARIOS_DIR / scenario_name
    configs_dir = scenario_dir / "base_files"
    configs_dir.mkdir(parents=True, exist_ok=True)

    for model, base_path in iter_base_files():
        data = load_json(base_path)
        data["simulation_parameters"]["number_agents"] = number_agents
        data["simulation_parameters"]["open_door_time"] = open_door_time
        output_name = f"base_{model}_{number_agents}_{open_door_time}.json"
        write_json(configs_dir / output_name, data)
        # The batch runner expects canonical base file names in its input directory.
        write_json(configs_dir / f"base_{model}.json", data)
        # Emit one config per seed so the paired design is automatic:
        # the same seed in base_{model}_seed{seed}.json across models
        # yields identical initial positions and value assignments.
        for seed in SEEDS:
            data["motivation_parameters"]["seed"] = float(seed)
            write_json(configs_dir / f"base_{model}_seed{seed}.json", data)

    return scenario_dir


def run_scenario(scenario_dir: Path) -> None:
    configs_dir = scenario_dir / "base_files"
    base_runs_dir = scenario_dir / "base_runs"
    analysis_dir = scenario_dir / "coordination_number_results"
    command = [
        "bash",
        str(RUNNER),
        str(base_runs_dir),
        str(analysis_dir),
        str(configs_dir),
    ]
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> None:
    for number_agents in NUMBER_AGENTS_VALUES:
        for open_door_time in OPEN_DOOR_TIMES:
            scenario_dir = build_scenario_configs(number_agents, open_door_time)
            print(f"Running scenario {scenario_dir.name}")
            run_scenario(scenario_dir)


if __name__ == "__main__":
    main()
