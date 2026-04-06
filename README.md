[![Streamlit](https://github.com/PedestrianDynamics/Motivation/actions/workflows/streamlit-actions.yml/badge.svg)](https://github.com/PedestrianDynamics/Motivation/actions/workflows/streamlit-actions.yml)
[![Ruff](https://github.com/PedestrianDynamics/Motivation/actions/workflows/ruff.yml/badge.svg)](https://github.com/PedestrianDynamics/Motivation/actions/workflows/ruff.yml)
[![Mypy](https://github.com/PedestrianDynamics/Motivation/actions/workflows/mypy.yml/badge.svg)](https://github.com/PedestrianDynamics/Motivation/actions/workflows/mypy.yml)
[![Tests](https://github.com/PedestrianDynamics/Motivation/actions/workflows/tests.yml/badge.svg)](https://github.com/PedestrianDynamics/Motivation/actions/workflows/tests.yml)

# Motivation

An expectancy-value-payoff (EVP) motivation model for pedestrian dynamics,
implemented on top of [JuPedSim](https://github.com/PedestrianDynamics/jupedsim).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the paper results

The full pipeline runs in three stages. All commands are run from
the repository root (`simulation/`).

### Stage 1: Multi-seed simulation + per-seed analysis

```bash
python scripts/run_all_scenario_analyses.py
```

**What it does** (≈ 40 min):

1. Reads `scripts/run_coordination_scenarios.py` which defines:
   - `BASE_MODELS = ["P", "V", "SE", "PVE", "BASE_MODEL"]`
   - `NUMBER_AGENTS_VALUES = [40, 80]`
   - `SEEDS = [101, 102, ..., 110]`
2. For each `(N, seed)` combination, generates a per-seed config by
   copying the base JSON (e.g. `files/base_PVE.json`) and overwriting
   `motivation_parameters.seed`. The same seed is used across all
   models so that initial positions and value assignments are
   identical — this is the **paired design**.
3. Runs `simulation.py` once per `(model, seed)`, producing:
   - `base_{MODEL}_seed{SEED}_{MODE}_{TIMESTAMP}.sqlite` — trajectory
   - `..._motivation.csv` — per-agent per-frame motivation values
4. For each seed, calls the four analysis scripts with explicit
   `--input MODEL=<file>` overrides and a `--tag` suffix:

| Analysis script | What it computes | Output |
|---|---|---|
| `scripts/final_rank_analysis.py` | Distance-to-door rank at last frame + mean Voronoi area per agent | `final_rank_details_*_seed{N}.csv` + scatter plots |
| `scripts/coordination_number_analysis.py` | Delaunay neighbor count per agent per frame | `coordination_number_details_*_seed{N}.csv` + histograms |
| `scripts/motivation_heatmap_analysis.py` | Spatial mean-motivation grid | heatmap PNGs |
| `scripts/voronoi_density_analysis.py` | Voronoi density in measurement area over time | `voronoi_density_details_*_seed{N}.csv` + time series |

**Outputs** go to `files/coordination_scenarios/agents_{N}_open_100/`.

### Stage 2: Aggregate across seeds

```bash
python scripts/aggregate_seeds.py
```

**What it does** (seconds):

1. Walks `files/coordination_scenarios/*/` and reads all seed-tagged
   detail CSVs from Stage 1.
2. Computes **per-seed summary scalars**:
   - Rank–area: Spearman ρ, OLS slope, tail ratio
   - Voronoi density: mean, std, fraction of time above threshold
   - Coordination number: mean, std, mode
3. Runs **paired Wilcoxon signed-rank tests** with **Cliff's δ**
   effect sizes comparing PVE vs BASE_MODEL.
4. Produces **band plots** (median + IQR across 10 seeds).

**Outputs** go to `files/coordination_scenarios/_aggregated/`:

| File | Contents |
|---|---|
| `seed_summary.csv` | One row per (seed, model, observable, scalar) |
| `tests.csv` | Paired Wilcoxon p-values and Cliff's δ per comparison |
| `rank_area_band_*.png` | Rank–area median + IQR bands |
| `voronoi_density_band_*.png` | Density time-series bands |
| `coordination_number_band_*.png` | Coordination number histogram bands |

### Stage 3: Experimental (CROMA) rank–area analysis

```bash
python scripts/experimental_rank_area.py
```

**What it does** (≈ 10 min):

1. Reads the CROMA `.txt` trajectory files from `../trajectories_croma/`.
2. For each experimental run, computes **crossing order** at the door
   (first frame where y ≥ 20) as the experimental rank.
3. Computes per-agent mean Voronoi area using PedPy with the CROMA
   corridor geometry.
4. Writes per-scenario CSVs + summary scalars (Spearman ρ, slope,
   tail ratio) and comparison plots.

**Outputs** go to `files/experimental_rank_area_results/`.

### Generating the EVP component figures

The expectancy, value, payoff, and motivation plots shown in the paper
are produced by the `EVCStrategy.plot()` method in
`src/motivation_model.py`. They can be regenerated via the Streamlit app
or by calling the plot method directly from a script (see
`src/simutilities.py:plot_motivation_model`).

The parameter-mapping figure (logistic curves for desired speed, time
gap, buffer, repulsion strength) is produced by
`motivation_mapping.plot_parameter_mappings()`.

## Configuration

All simulation parameters are in `files/base_{MODEL}.json`. Key settings:

| Parameter | Location in JSON | Paper value |
|---|---|---|
| Motivation mode | `motivation_parameters.motivation_mode` | PVE / BASE_MODEL / P / V / SE |
| Auto-k logistic | `motivation_parameters.use_manual_logistic_k` | `false` |
| Seeds | `motivation_parameters.seed` | 101–110 (set by pipeline) |
| Number of agents | `simulation_parameters.number_agents` | 40 or 80 |
| Door open time | `simulation_parameters.open_door_time` | 100 (> sim time → closed) |
| Simulation time | `simulation_parameters.simulation_time` | 90 s |

## Project structure

```
simulation/
├── simulation.py                  Main simulation runner (JuPedSim)
├── app.py                         Streamlit interactive UI
├── files/
│   ├── base.json                  Template configuration
│   ├── base_{MODEL}.json          Per-model configs (P, V, SE, PVE, BASE_MODEL)
│   ├── coordination_scenarios/    Simulation outputs + analysis results
│   └── experimental_rank_area_results/
├── scripts/
│   ├── run_coordination_scenarios.py   Config generation + seed definitions
│   ├── run_all_scenario_analyses.py    Full pipeline runner (Stage 1)
│   ├── aggregate_seeds.py              Cross-seed aggregation (Stage 2)
│   ├── experimental_rank_area.py       CROMA analysis (Stage 3)
│   ├── final_rank_analysis.py          Rank–area per-seed analysis
│   ├── coordination_number_analysis.py Coordination number per-seed analysis
│   ├── motivation_heatmap_analysis.py  Spatial motivation heatmaps
│   └── voronoi_density_analysis.py     Voronoi density time series
├── src/
│   ├── motivation_model.py        EVCStrategy: expectancy-value model
│   ├── motivation_mapping.py      Logistic motivation-to-parameter mapping
│   ├── coordination_number.py     Delaunay coordination numbers
│   ├── utilities.py               Voronoi area, rank, crossing-order utilities
│   └── ...
└── tests/
```

## Interactive exploration

```bash
streamlit run app.py
```

## Running a single simulation

```bash
python simulation.py --inifile files/base_PVE.json --output-dir output/
```

## License

MIT
