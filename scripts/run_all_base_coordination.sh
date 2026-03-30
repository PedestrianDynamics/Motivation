#!/bin/bash

set -euo pipefail

OUTPUT_DIR="${1:-files/base_runs}"
ANALYSIS_DIR="${2:-coordination_number_results}"
BASE_DIR="${3:-files}"

BASE_FILES=(
    "${BASE_DIR}/base_P.json"
    "${BASE_DIR}/base_V.json"
    "${BASE_DIR}/base_E.json"
    "${BASE_DIR}/base_PVE.json"
    "${BASE_DIR}/base_NO_MOTIVATION.json"
)

for base_file in "${BASE_FILES[@]}"; do
    if [[ ! -f "${base_file}" ]]; then
        echo "Missing base file: ${base_file}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${ANALYSIS_DIR}"

for base_file in "${BASE_FILES[@]}"; do
    echo "Running simulation for ${base_file}"
    python simulation.py --inifile "${base_file}" --output-dir "${OUTPUT_DIR}"
done

python scripts/coordination_number_analysis.py \
    --models PVE NO_MOTIVATION \
    --search-dir "${OUTPUT_DIR}" \
    --output-dir "${ANALYSIS_DIR}"
