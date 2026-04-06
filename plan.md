# Seed-Replication Analysis Plan

## Goal

Turn the current qualitative single-seed scenario analysis into a
defensible multi-seed comparison that supports the existing claims
(rank-dependent structure in PVE, flat base, density as secondary,
spatial localization of motivation) without rewriting the paper's
interpretation.

## Guiding rule

**Unit of inference = one run → one summary scalar.** Compare
distributions of per-seed scalars across models. Never pool
agent-frames across seeds and treat them as independent samples — that
inflates the effective N and gives spurious confidence intervals.

---

## 1. Rank–area (primary observable)

### Visualize
- For each rank `r ∈ {1..N}`, aggregate area values `{a_r^(s)}` across
  seeds `s`.
- Plot across-seed **median area vs. rank** with an **IQR band**
  (10–90 percentile band is acceptable too).
- One curve + band per model, overlaid.

### Per-run scalars (compute one value per seed)
- **Spearman ρ(rank, area)** — direct measure of monotonicity.
  Flat base → ρ ≈ 0; structured PVE → ρ bounded away from 0.
- **Tail ratio** = mean area in top quartile (worst-ranked agents)
  ÷ mean area in bottom quartile. Captures the "upward tail" claim.
- **Slope** of OLS fit `a = α + β·r`.

### Test
- With K seeds per model, run **Mann–Whitney U** (or permutation test)
  on the per-seed ρ values, base vs. PVE.
- Report **Cliff's δ** as effect size.
- Repeat for tail ratio and slope.

### Paired design (recommended)
- Run each seed through **both** base and PVE using the same initial
  agent placement.
- Use **Wilcoxon signed-rank** test on within-seed differences
  `ρ_PVE^(s) − ρ_base^(s)`.
- Paired designs give much tighter inference with few seeds.

---

## 2. Voronoi density (secondary observable)

### Visualize
- Align time axes, compute across-seed **median and IQR band** of
  `ρ(t)` per model.
- Overlay both bands. Avoid line-plots of raw runs stacked on top
  of each other.

### Per-run scalars
- **Time-averaged density** over a fixed window `[t1, t2]`.
- **Fraction of time above a density threshold** (e.g. 3 m⁻²).
- **Density variability** (per-run std) — directly tests the
  heterogeneous-buffers prediction: PVE should show higher
  within-run variability, not necessarily higher mean.

### Test
- Same per-seed Mann–Whitney / Wilcoxon with Cliff's δ.

---

## 3. Motivation heatmaps (spatial channel)

Averaging raw heatmaps across seeds is fine but not very informative
by itself. Reduce each heatmap to **one or two scalars per seed**.

### Per-run scalars
- **Spatial center of mass of motivation**, especially the
  y-coordinate. Low `y_COM` ⇒ motivation localized near the door.
- **Front-box fraction**: share of total motivation mass inside the
  Voronoi measurement box (or any chosen box near the door). Tests
  spatial localization directly.
- **Spatial entropy / Gini** of the bin-level motivation distribution.
  Tests "uniform urgency vs. structured localization" in one number.

### Visualize (support only)
- Across-seed **mean heatmap** per model + a **per-bin std map**.
- The scalars carry the statistical weight; the heatmap carries
  the visual intuition.

---

## 4. Coordination number (structural control)

### Visualize
- Per seed, histogram of N_n.
- Across seeds, plot **mean probability per bin ± seed-std**
  (or bootstrap band). Overlay base vs. PVE.

### Per-run scalars
- Mean N_n
- Variance of N_n
- Mode of N_n

The structural-control framing means weak differences are *expected*.
Bands that overlap are supporting evidence that this observable is
not the discriminator.

---

## How many seeds?

| Setting         | Seeds per (model × N) cell |
|-----------------|-----------------------------|
| Minimum defensible | 10                       |
| Recommended        | 20–30                    |
| Paired (same seed drives base & PVE) | 10 paired ≈ 20 unpaired |

Current status: 5 seeds at N=40, 2 seeds at N=80 — too few for
inference but sufficient to pilot the pipeline and select which
scalars discriminate before committing to a larger sweep.

---

## How the seed works in this codebase

The seed lives in the base JSON files at
`motivation_parameters.seed` (e.g. `files/base_PVE.json:67`,
`files/base_BASE_MODEL.json:67`). It is consumed at two points in
`simulation.py`:

- **Initial agent positions** — `init_positions` (line 671,680)
  passes `seed` to `distribute_by_number(...)`.
- **Motivation strategy RNG** — `init_motivation_model` (line 294,320)
  passes `seed` to `EVCStrategy(...)`, which drives high/low-value
  assignment and related draws.

**Consequence:** if the same seed value is written into
`base_PVE.json` and `base_BASE_MODEL.json` before running, both
models start from **identical initial positions and identical
value assignments**. That is the paired design — no extra bookkeeping
required.

## Implementation plan

### Step 1 — seed sweep runner

Extend `scripts/run_coordination_scenarios.py` so that, for each
`(N, open_door_time)` scenario and each base model, it writes
**one config per seed** with `motivation_parameters.seed` overwritten
to the chosen value.

Concretely, add:

```python
SEEDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]  # 10 paired seeds
```

and in `build_scenario_configs`, loop over `SEEDS` and emit config
filenames that encode both model and seed, e.g.
`base_PVE_seed101.json`, `base_BASE_MODEL_seed101.json`, …

Then extend `scripts/run_all_scenario_analyses.py` so
`run_simulations` iterates over `(model, seed)` pairs and invokes
`simulation.py --inifile <config> --output-dir <runs_dir>` once per
pair. Each run produces its own sqlite and `_motivation.csv`, tagged
with the seed in the filename.

### Step 2 — per-seed analysis
The existing per-scenario scripts
(`coordination_number_analysis.py`, `final_rank_analysis.py`,
`motivation_heatmap_analysis.py`, `voronoi_density_analysis.py`)
currently pick the **most recent** sqlite/csv matching a model.
Change them so they either:

(a) accept an explicit `--seed` filter and a list of seeds, and
emit one set of details CSVs per seed, **or**

(b) be called in a loop from the runner with explicit `--input`
overrides pointing to the per-seed file.

Option (b) requires no changes to the existing analysis scripts —
just loop from `run_all_scenario_analyses.py`.

### Step 3 — second-level aggregation script
Write `scripts/aggregate_seeds.py` that ingests the per-seed CSVs
produced by Step 2 and emits:

1. **Band plots** (median + IQR) for
   - rank–area
   - Voronoi density time series
   - coordination-number histogram
2. **`seed_summary.csv`** — one row per (seed, model, observable, N),
   holding all per-run scalars defined above.
3. **`tests.csv`** — one row per comparison, containing:
   - observable, scalar, model_A, model_B, N
   - test name (Mann–Whitney U or Wilcoxon)
   - p-value
   - Cliff's δ (effect size)
   - paired flag

The paired convention is already guaranteed by Step 1: the same
`SEEDS` list is used for every model, so for each seed `s` there is
exactly one base run and one PVE run that share initial positions
and value assignments.

### Step 4 — pilot run
Run the pipeline with 5–10 paired seeds first. Inspect which scalars
actually discriminate base from PVE. Drop scalars that add noise;
keep the ones that carry the claim.

### Step 5 — production sweep
Once scalars are chosen, run 20 paired seeds per `(model, N)` cell
and regenerate all plots/tables for the paper.

---

## What this buys the paper

- The existing claims (rank structure in PVE, flat base, density as
  secondary, spatial localization) become claims about distributions
  of per-seed scalars with associated p-values and effect sizes.
- Band plots replace single-run overlays, removing the obvious
  reviewer objection.
- No rewrite of the Methods/Discussion/Conclusion narrative is needed:
  the scalars are aligned with the expectancy-value framing
  (rank = collective channel, heatmap = spatial channel).
