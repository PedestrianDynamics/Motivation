# CROMA replica: design and decision criteria

Fixed on 2026-11-07, before any replica run, so that results are not
interpreted after the fact.

## Question

Does the rank–area gradient in the CROMA runs (ρ 0.39–0.68, see
`scripts/croma_ordering.py`) require motivation, or does a static model
started from the measured entries reproduce it? And can any model
reproduce the motivation-dependent overtaking (entry order vs crossing
order: blank runs nM 0.83 and 1.00, hM 0.66 and 0.56)?

## Design

- Geometry: the CROMA corridor and closed door as in `simulation.py`
  (identical to `trajectories_croma/delme/geometry.xml`).
- Population: every participant enters at the position and time of
  their first observation; the door stays closed until the measured
  opening frame, then opens.
- Measures: exactly the experimental definitions (crossing at y ≥ 20 m;
  rank–area over all frames and over the waiting phase;
  entry–crossing ρ; crowd depth and front fraction from
  `scripts/crowd_shape.py`).
- Base model: CollisionFreeSpeedModelV3, the model EVP runs on.
- Arms, each with the same fitting budget:
  1. static: one constant time gap per condition, fitted only to crowd
     depth and front fraction at door opening;
  2. V-only: static individual values;
  3. EVP.
- Several seeds per run; report the spread.

## Decision criteria

1. A static arm with realistic crowd shape (depth and front fraction
   within the experimental range of its condition) gives rank–area ρ
   within 0.39–0.59: the rank–area gradient is withdrawn as evidence
   for motivation.
2. A static arm gives entry–crossing ρ close to 1 in both conditions
   while the experiment shows hM < nM: overtaking becomes the test, and
   EVP and V-only must reproduce the drop from a change in payoff or V
   alone, with all other parameters fixed.
3. A static arm that cannot reach realistic crowd depth on V3 is not
   interpreted; the check is repeated with CollisionFreeSpeedModelV2
   and the calibrated set C of jupedsim-dakota-calibration.

## Amendment (2026-11-07, before any arm was run)

Smoke tests of the static arm (2C070, seed 101, time gap 0.3–1.5 s,
desired speed 1.2–1.5 m/s) deadlock at the 0.8 m door after it opens:
11–84 of 89 agents are still waiting 120 s later, while the experiment
discharges everyone at 0.95 /s. The paper's simulations never opened the
door. The measures therefore move to the moment the door opens, which
the experiment justifies: queue position at opening predicts crossing
order with ρ 0.96–1.00 in all eight runs (`croma_ordering.py`).

- rank: distance rank to the door at the opening time,
- area: mean Voronoi area over the waiting phase (before opening),
- overtaking: Spearman(entry order, rank at opening),
- crowd shape at opening: depth and front fraction (`crowd_shape.py`).

The experimental values are computed with the same code. Criteria 1–3
apply unchanged to these measures. The discharge deadlock is itself a
finding to report: the model has not been shown to reproduce the
opening phase.

## Second amendment (2026-11-07, before the full grid was run)

The door deadlock of the first amendment was two artefacts, not a
model property:

- The replica placed nudged entrants outside the corridor walls, from
  where they walked around the closed door. Entrants are now placed
  inside the corridor only.
- CollisionFreeSpeedModelV3 with agent buffer 0 stops two agents that
  touch head-on at exactly zero speed, so the reverse floor never
  frees them. Any buffer above 0 dissolves the jam. The static arm
  uses buffer 0.1 m, the "normal" anchor of the motivation mapping;
  V-only and EVP take their buffer from the mapping as before.

The grid runs on jupedsim 1.4.2 with the V3 fix of PR #1721 (agents no
longer back through walls); results are regenerated on the release
that contains it.

The measures at opening stay as amended. Added as a held-out measure,
not used for selection: discharge, the number of people crossing the
door (y ≥ 20 m) in the first 60 s after opening
(`croma_waiting.discharge`). Each replica runs to 60 s after opening.

## Caveats known in advance

Two blank runs per condition; long waits (2C130, 2C150, about 200 s)
also lower entry–crossing ρ, so waiting time confounds overtaking.

## Third amendment (2026-11-08, before the fitting grid was run)

Result of criterion 3 (`--static-params`, CFSM V2, runs in
`build/croma_replica_fixed/grid_V2_*.csv`): set C and the CrowdQueue
condition fits (h0: T 0.79 s, v0 1.17 m/s; h-: T 1.73 s, v0 1.50 m/s)
all stand 1.0–1.3 m deep with front fraction 1.0 (experiment 3.2 m and
0.44–0.49 in hM, 9.8–13.9 m and 0.08–0.16 in nM) and discharge 64–89
(experiment 44–63). The static arm on V2 is therefore not interpreted
either. Its rank–area ρ of 0.45–0.75 is reported, not used: a compact
static crowd gives a gradient of the paper's size. V2 runs are
deterministic; the seed only moves entrants that have to be nudged.

Time gap and desired speed do not set the standing depth: at rest CFSM
V2 keeps agents 2r apart and V3 keeps them 2r + buffer apart, and
neither the V3 grid (T 0.3–2.0 s) nor the three V2 sets moved the depth.
The fitted parameter is therefore the standing distance; T stays at the
configured 1.0 s, so discharge stays held out.

- Static arm (V3): agent buffer ∈ {0.1, 0.2, 0.3, 0.5, 0.7, 1.0} m,
  selected separately per condition. This gives the static arm one more
  fitted value than the motivation arms and makes it a descriptive
  upper bound, not a predictive competitor.
- V-only and EVP: the low buffer anchor of the mapping
  ∈ {0.4, 0.6, 0.8, 1.0, 1.3, 1.6} m (normal 0.2, high 0.1 unchanged),
  selected on nM only; hM is then a prediction from V or payoff alone.
- Selection score per condition, mean over its two blank runs and
  three seeds: |ln(depth_sim / depth_exp)| + |front_sim − front_exp|.
- Held out from selection: entry–crossing ρ, rank–area ρ, discharge,
  and every hM measure of V-only and EVP.

## Result of the standing-distance fit (2026-11-08)

`scripts/croma_fit_select.py` on `build/croma_fit/`. Deviation, made
after seeing the first grid: the static nM optimum sat at the edge
(1.0 m) while the motivation arms could reach 1.6 m, so the static
buffer grid was extended to 1.3, 1.6 and 2.0 m. Its optimum is still at
the edge (2.0 m); the selected values are static nM 2.0 / hM 0.1,
V low anchor 1.3, EVP low anchor 1.6 (edge; scores flat from 1.0).

Shape (fitted): every arm reaches hM shape; nM scores static 0.43,
V 0.35, EVP 0.31. Held out (experiment in brackets):

- nM entry–crossing ρ: static 0.89–0.90 (0.81–0.99), V 0.43–0.44,
  EVP 0.28–0.40. Both motivation arms reorder nM far too much.
- hM entry–crossing ρ: static 0.91 (0.46–0.66), V 0.46–0.50, EVP
  0.00–0.03. Only V-only reproduces hM overtaking.
- hM rank–area ρ: static −0.12 to −0.15 (0.42–0.51), V 0.34–0.35,
  EVP 0.06–0.16.
- Discharge in 60 s: nM static 27, V 13, EVP 13–22 (44–46);
  hM static 48–49, V 74–75, EVP 59–75 (63).

V-only predicts the hM measures best from the nM fit; the EVP dynamics
make them worse. No arm reproduces nM ordering together with nM
discharge, and all nM discharges are too slow.

## Why the motivation arms reorder nM (2026-11-08)

`scripts/croma_value_rank.py`, `scripts/croma_nm_factorial.py`.

- The mapping is saturated in nM. Values 1–3 give motivation 0.21–0.64,
  where every agent gets buffer 1.3 m (the low anchor), v0 0.50–0.52 m/s,
  T 1.5–1.9 s and neighbour strength 0.10–0.13. The nM agents are
  effectively identical, and value does not predict rank at opening
  (ρ −0.04 to 0.01 in V, −0.06 to −0.02 in EVP).
- A static replica with these values reproduces the V arm in nM:
  entry–crossing ρ 0.35–0.37 (V 0.43–0.44), discharge 12 (V 13).
  One at a time from the static arm at buffer 1.3 m (ρ 0.84–0.95,
  discharge 30): v0 0.5 gives ρ 0.64–0.68 and discharge 15; neighbour
  strength 0.1 gives ρ 0.65–0.67 and a crowd 10–15 m deep; T 1.9 s
  changes neither.
- The nM failure is therefore a property of the low anchors of the
  mapping (v0 0.5 m/s, strength 0.1), not of motivation or its dynamics.
- In hM the V arm's overtaking does come from heterogeneity: values
  4–7 straddle the steep step of the mapping (buffer 1.3 → 0.1 m and v0
  0.7 → 2.6 m/s between V 4 and 5.5), and value predicts advance over
  entry order with ρ 0.68. EVP weakens this link (ρ 0.11–0.20).

## Fourth amendment (2026-11-08, before the low-anchor grid was run)

The nM failure lies in the low anchors (previous section), so the fit
is widened to the three parameters that set nM behaviour, with the same
grid for every arm, and nM discharge becomes a fit target.

- Grid: standing distance ∈ {1.0, 1.3, 1.6, 2.0} m × desired speed
  ∈ {0.6, 0.8, 1.0, 1.2} m/s × neighbour strength ∈ {0.1, 0.3, 0.5}.
  V-only and EVP vary the low anchors (buffer, desired speed, strength;
  normal and high unchanged), fitted on nM only, and are run on hM once
  at the selected point. The static arm varies agent buffer, normal_v_0
  and a_ped: on nM with this grid, on hM with buffer ∈ {0.1, 0.2, 0.3,
  0.5} and the same speed and strength values.
- Score per condition, mean over its two blank runs and seeds:
  |ln(depth_sim/depth_exp)| + |front_sim − front_exp|
  + |ln(discharge_sim/discharge_exp)|.
- The static arm's hM fit now also uses hM discharge. For V-only and
  EVP every hM measure stays held out. Entry–crossing ρ and rank–area ρ
  stay held out in every arm and condition.
- Changing a low anchor reshapes the whole logistic curve, so it also
  moves hM agents (at V = 5.5, v0 2.2 m/s with low anchor 0.6 against
  3.6 m/s with 1.0). That is part of the prediction, not corrected.
