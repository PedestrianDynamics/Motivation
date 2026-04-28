"""Render the EVP schematic figures (expectancy, value, payoff, motivation)
plus the parameter-mapping panel, from a chosen base config (default: base_PVE.json).

Outputs go to the paper figures directory by default.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.motivation_mapping as mmap  # noqa: E402
import src.motivation_model as mm  # noqa: E402
from src.simutilities import (  # noqa: E402
    create_motivation_strategy,
    extract_motivation_parameters,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "files" / "base_PVE.json"
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT.parent / "motivation-for-springer" / "figures",
    )
    p.add_argument("--n-agents", type=int, default=80)
    p.add_argument("--seed", type=int, default=101)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.config.read_text())
    data["simulation_parameters"]["number_agents"] = args.n_agents
    data["motivation_parameters"]["seed"] = float(args.seed)
    params = extract_motivation_parameters(data)

    strategy = create_motivation_strategy(params)
    mapper = mmap.MotivationParameterMapper(
        mapping_block=params["mapping_block"],
        normal_v_0=params["normal_v_0"],
        range_default=params["d_ped"],
    )
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

    work = Path("/tmp/evp_schematics")
    work.mkdir(exist_ok=True)
    cwd = Path.cwd()
    try:
        os.chdir(work)
        motivation_model.motivation_strategy.plot()
        fig = mmap.plot_parameter_mappings(
            mapper=mapper, normal_time_gap=params["normal_time_gap"]
        )
        fig.savefig("parameter_mappings.pdf", bbox_inches="tight")
    finally:
        os.chdir(cwd)

    for name in (
        "expectancy.pdf",
        "value.pdf",
        "payoff.pdf",
        "motivation.pdf",
        "parameter_mappings.pdf",
    ):
        src = work / name
        if src.exists():
            shutil.copy(src, args.output_dir / name)
            print(f"Wrote {args.output_dir / name}")


if __name__ == "__main__":
    main()
