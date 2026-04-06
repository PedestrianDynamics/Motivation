#!/bin/bash

set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-overtaking}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache-overtaking}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

SCENARIO_ROOT="${1:-files/coordination_scenarios}"
SIM_OUTPUT_SUBDIR="${2:-overtaking_results}"
TMIN="${3:-10}"
TMAX="${4:-300}"
EXPERIMENT_DIR="${5:-/Users/chraibi/workspace/Writing/Motivation/CroMa videos}"
EXPERIMENT_OUTPUT_DIR="${6:-overtaking_results_experimental}"

for scenario_dir in "${SCENARIO_ROOT}"/agents_*_open_*; do
    if [[ ! -d "${scenario_dir}" ]]; then
        continue
    fi

    scenario_name="$(basename "${scenario_dir}")"
    base_runs_dir="${scenario_dir}/base_runs"
    output_dir="${scenario_dir}/${SIM_OUTPUT_SUBDIR}"

    if [[ ! -d "${base_runs_dir}" ]]; then
        echo "Skipping ${scenario_name}: missing ${base_runs_dir}" >&2
        continue
    fi

    echo "Running overtaking analysis for ${scenario_name}"
    python3 scripts/overtaking_analysis.py \
        --models P V SE PVE BASE_MODEL \
        --search-dir "${base_runs_dir}" \
        --t-min "${TMIN}" \
        --t-max "${TMAX}" \
        --output-dir "${output_dir}" \
        --tag "${scenario_name}"
done

if [[ -d "${EXPERIMENT_DIR}" ]]; then
    echo "Running overtaking analysis for experimental trajectories"
    python3 scripts/overtaking_experimental.py \
        --experiment-dir "${EXPERIMENT_DIR}" \
        --t-min "${TMIN}" \
        --t-max "${TMAX}" \
        --output-dir "${EXPERIMENT_OUTPUT_DIR}"
else
    echo "Skipping experiments: missing ${EXPERIMENT_DIR}" >&2
fi
