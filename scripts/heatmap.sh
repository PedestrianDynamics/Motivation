for scenario_dir in files/coordination_scenarios/agents_*_open_*; do
  scenario_name="$(basename "$scenario_dir")"
  python3 scripts/motivation_heatmap_analysis.py \
    --models P V E PVE NO_MOTIVATION \
    --search-dir "$scenario_dir/base_runs" \
    --t-min 10 \
    --t-max 300 \
    --bins 60 \
    --sigma 1.5 \
    --output-dir "motivation_heatmap_results/$scenario_name"
done
