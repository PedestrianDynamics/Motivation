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
