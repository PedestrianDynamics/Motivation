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

## Caveats known in advance

Two blank runs per condition; long waits (2C130, 2C150, about 200 s)
also lower entry–crossing ρ, so waiting time confounds overtaking.
