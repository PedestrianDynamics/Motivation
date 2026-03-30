#!/bin/bash

set -euo pipefail

SCENARIO_ROOT="${1:-files/coordination_scenarios}"
OUTPUT_ROOT="${2:-voronoi_density_results}"
TMIN="${3:-10}"
TMAX="${4:-300}"

for scenario_dir in "${SCENARIO_ROOT}"/agents_*_open_*; do
    if [[ ! -d "${scenario_dir}" ]]; then
        continue
    fi

    scenario_name="$(basename "${scenario_dir}")"
    base_runs_dir="${scenario_dir}/base_runs"
    output_dir="${OUTPUT_ROOT}/${scenario_name}"

    if [[ ! -d "${base_runs_dir}" ]]; then
        echo "Skipping ${scenario_name}: missing ${base_runs_dir}" >&2
        continue
    fi

    echo "Running Voronoi density analysis for ${scenario_name}"
    python3 scripts/voronoi_density_analysis.py \
        --models PVE NO_MOTIVATION \
        --search-dir "${base_runs_dir}" \
        --t-min "${TMIN}" \
        --t-max "${TMAX}" \
        --output-dir "${output_dir}"
done
