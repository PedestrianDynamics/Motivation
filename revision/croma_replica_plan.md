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

## Caveats known in advance

Two blank runs per condition; long waits (2C130, 2C150, about 200 s)
also lower entry–crossing ρ, so waiting time confounds overtaking.
