#!/bin/bash

set -euo pipefail

SCENARIO_ROOT="${1:-files/coordination_scenarios}"
OUTPUT_SUBDIR="${2:-overtaking_results}"
TMIN="${3:-10}"
TMAX="${4:-300}"

for scenario_dir in "${SCENARIO_ROOT}"/agents_*_open_*; do
    if [[ ! -d "${scenario_dir}" ]]; then
        continue
    fi

    scenario_name="$(basename "${scenario_dir}")"
    base_runs_dir="${scenario_dir}/base_runs"
    output_dir="${scenario_dir}/${OUTPUT_SUBDIR}"

    if [[ ! -d "${base_runs_dir}" ]]; then
        echo "Skipping ${scenario_name}: missing ${base_runs_dir}" >&2
        continue
    fi

    echo "Running overtaking analysis for ${scenario_name}"
    python3 scripts/overtaking_analysis.py \
        --models P V E PVE NO_MOTIVATION \
        --search-dir "${base_runs_dir}" \
        --t-min "${TMIN}" \
        --t-max "${TMAX}" \
        --output-dir "${output_dir}" \
        --tag "${scenario_name}"
done
