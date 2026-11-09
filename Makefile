# Regenerate every figure the paper includes from committed code and configs.
#
#   make paper-figures          simulations + analyses + figures into $(OUT)
#   make compare                compare $(FIG) against the paper's figures
#   make install-figures        copy $(FIG) into the paper (after compare)
#
# Set OUT to a fresh directory: the runners refuse to pick among several
# runs of the same (model, seed).

PY       ?= .venv/bin/python
OUT      ?= build/paper
FIG      ?= $(OUT)/figures
CROMA    ?= ../trajectories_croma
PAPER    ?= ../motivation-for-springer
SCEN     := $(OUT)/coordination_scenarios
TESTING  := 1C060 2C020 2C070 2C120
MODELS   := se v p pve base_model

export MOTIVATION_SCENARIOS_DIR := $(abspath $(SCEN))

.PHONY: paper-figures simulations aggregate schematics trajectories croma testing replica seed-figures check manifest compare install-figures

paper-figures: simulations aggregate schematics trajectories croma testing seed-figures check manifest

# Stage 1: 5 models x N in {40, 80} x seeds 101-110, plus per-seed analyses.
simulations:
	$(PY) scripts/run_all_scenario_analyses.py

# Stage 2: cross-seed bands (rank_area_band_*.png).
aggregate: | $(FIG)
	$(PY) scripts/aggregate_seeds.py --scenarios-dir $(SCEN) --output-dir $(SCEN)/_aggregated
	cp $(SCEN)/_aggregated/rank_area_band_agents_40_open_100.png $(SCEN)/_aggregated/rank_area_band_agents_80_open_100.png $(FIG)/

schematics: | $(FIG)
	$(PY) scripts/plot_evp_schematics.py --output-dir $(FIG)

trajectories: | $(FIG)
	$(PY) scripts/plot_simulation_trajectories.py --base-runs-dir $(SCEN)/agents_80_open_100/base_runs --seed 101 --output-dir $(FIG)
	$(PY) scripts/plot_experimental_trajectories.py --traj-dir $(CROMA) --output $(FIG)/croma_trajectories_4panel.png

croma: | $(FIG)
	$(PY) scripts/experimental_rank_area.py --input-dir $(CROMA) --geometry $(CROMA)/delme/geometry.xml --output-dir $(OUT)/croma_rank_area
	cp $(OUT)/croma_rank_area/croma_rank_area_nM.png $(OUT)/croma_rank_area/croma_rank_area_hM.png $(FIG)/

# Fig. 10: EVP runs matched to the four CROMA runs.
testing: | $(FIG)
	rm -rf $(OUT)/testing_runs
	for run in $(TESTING); do \
		mkdir -p $(OUT)/testing_runs/test_$$run && \
		$(PY) simulation.py --inifile files/base_testing_$$run.json --output-dir $(OUT)/testing_runs/test_$$run || exit 1; \
	done
	$(PY) scripts/plot_sim_testing_trajectories.py --runs-dir $(OUT)/testing_runs --output $(FIG)/sim_testing_trajectories_4panel.png
	$(PY) scripts/crowd_shape.py --traj-dir $(CROMA) --runs-dir $(OUT)/testing_runs --output $(OUT)/crowd_shape.csv

# CROMA replica: fit of the three models, held-out table and waiting-phase
# time series. Needs a jupedsim release with the V3 wall fix (PR #1721);
# not yet part of paper-figures. REPLICA_PY may point to such a build.
REPLICA_PY ?= $(PY)
replica: | $(FIG)
	$(REPLICA_PY) scripts/croma_waiting.py --traj-dir $(CROMA) --output $(OUT)/croma_waiting.csv
	for mode in BASE_MODEL V PVE; do \
		$(REPLICA_PY) scripts/croma_anchor_fit.py --mode $$mode --experiment $(OUT)/croma_waiting.csv --output-dir $(OUT)/croma_anchor_fit || exit 1; \
	done
	$(PY) scripts/croma_results_table.py --fit-dir $(OUT)/croma_anchor_fit --experiment $(OUT)/croma_waiting.csv --output $(FIG)/croma_heldout_table.tex
	$(PY) scripts/croma_timeseries.py --traj-dir $(CROMA) --replica-dir $(OUT)/croma_anchor_fit --output-dir $(OUT)/croma_timeseries
	$(PY) scripts/plot_croma_timeseries.py --input-dir $(OUT)/croma_timeseries --output-dir $(FIG)

# The paper shows seed 101 as the representative single-seed figure.
seed-figures: | $(FIG)
	cp $(SCEN)/agents_80_open_100/motivation_heatmap_results/motivation_heatmap_models_agents_80_open_100_seed101.png $(FIG)/motivation_heatmap_models_agents_80_open_100.png
	for m in $(MODELS); do \
		cp $(SCEN)/agents_80_open_100/final_rank_results/final_rank_vs_area_$${m}_agents_80_open_100_seed101.pdf $(FIG)/final_rank_vs_area_$${m}_agents_80_open_100.pdf || exit 1; \
	done

check:
	$(PY) scripts/paper_manifest.py check --figures $(FIG)

manifest:
	$(PY) scripts/paper_manifest.py manifest --figures $(FIG)

compare:
	$(PY) scripts/paper_manifest.py compare --figures $(FIG) --reference $(PAPER)/figures

install-figures: check
	cp $(FIG)/*.png $(FIG)/*.pdf $(PAPER)/figures/

$(FIG):
	mkdir -p $@
