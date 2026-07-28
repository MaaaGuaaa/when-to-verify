# SOP05 Regime A: Unseen-History Scenario Synthesis

This package defines the late-SOP05 synthesis stage selected when the source
pedestrian is hidden at the first history frame H0. Frames H1..H7 do not decide
the A/B regime. An accepted target-present realization must still pass the
separate all-eight-frame hidden candidate gate. It is not a SOP06 renderer plan.

## Goal

For each eligible mother event, publish at most one finalized scenario.

- With probability `0.70`, remove the hidden pedestrian and publish the empty
  scenario.
- With probability `0.30`, retain the pedestrian and find one legal rotated
  Long40 trajectory.

`0.30` is an application prior chosen for this synthetic regime. It is not an
estimate of natural pedestrian frequency. SOP05 does not force collision,
near-collision, or safe proportions, and does not produce class weights.

## Input

The mother provides the scene, robot history, static map, occluder, represented
context objects, sensor configuration, target identity/footprint, and one
authenticated Long40 target motion:

```text
positions:  float32 [40, 2]
velocities: float32 [40, 2]
headings:   float32 [40]
history:    indices 0..7
current:    index 7
future:     indices 8..39
```

H0 is hidden. Visibility at H1..H7 does not change the source A/B identity; it
is re-evaluated after rotation by the candidate gate.

## Synthesis

Use a stable per-mother RNG derived from the configured seed, split, and mother
identity. Never use Python `hash()`.

1. Draw the presence branch once.
2. Empty branch: remove only the target and emit that scenario.
3. Present branch: draw `theta ~ Uniform[-pi, pi)` and rigidly rotate all 40
   target poses, velocities, and headings about target position at index 7.
4. Reject candidates that leave the scene, hit static/occluder/context geometry,
   hit the robot in history frames `0..7`, or become visible in any history
   frame.
5. Retry at most 32 angles. Publish the first legal candidate; otherwise record
   `no_legal_angle` and publish no scenario for that mother.

Do not reject a candidate for robot collision after index 7. That future is the
realized scenario; SOP7 determines its risk label later.

## Handoff

SOP05 publishes one finalized scene or one accounted deficit per mother. Its
provenance may retain the branch, attempt count, selected angle, and rejection
counts for audit only. These fields, target future, and oracle collision result
must not enter model-visible tensors or metadata.

SOP6 receives only the finalized scene's history-safe render input and produces
one history BEV. SOP7 receives the safe render plus the oracle world and
computes labels. There are no paired siblings.

## Non-goals

- No angle grid or exhaustive angle sweep.
- No per-class quotas, natural-frequency estimate, or training weight.
- No future robot-collision filter, future-visibility filter, or trajectory
  smoothing.
- No fallback from a failed present branch to an empty scenario.
