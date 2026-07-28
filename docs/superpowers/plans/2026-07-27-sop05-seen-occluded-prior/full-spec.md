# SOP05 Regime B: Seen-Then-Occluded Scenario Synthesis

This package defines the late-SOP05 synthesis stage for a pedestrian that is
visible at the first history frame H0. Frames H1..H7, including whether the
pedestrian is visible at H7, do not decide the A/B regime. It is not a SOP06
renderer or risk-label plan.

## Goal

For every eligible mother, select at most one plausible future trajectory. The
first eight history poses are fixed. The future is allowed to turn around the
current direction, with small turns much more likely than large turns.

## Input

The source must contain finite floating target poses:

```text
history poses: [8, 3]
current pose:  [3] and byte-equal to history[7]
future poses:  [32, 3]
history visibility: bool [8]
```

History frame H0 must be visible. Other history frames do not determine the A/B
identity; their actual visibility mask is preserved for the model, so an
H7-visible pedestrian remains visible at H7 in SOP6.
The source also supplies static/occluder occupancy, represented context sweeps,
and the target footprint for environmental legality.

## Synthesis

For a stable RNG stream derived from dataset seed, split, source collection
identity, and mother ID:

1. Draw `theta ~ Normal(0, (pi/12)^2)`.
2. Treat a draw outside `[-pi, pi)` as a consumed rejected attempt.
3. Rotate only the 32 future poses about the XY position at history index 7.
   Rotate future yaw by the same angle. Copy history and current unchanged.
4. Reject a candidate that is non-finite, out of bounds, intersects static or
   occluder occupancy, or intersects a represented context sweep.
5. Take the first legal candidate from at most 32 attempts. If none is legal,
   record `no_legal_future` and publish no scenario.

The seam may be sharp. Do not smooth it or add speed/acceleration constraints.

## Causal Boundary

SOP05 does not test robot collision, calculate clearance, assign collision/
near/safe class, or examine future visibility. These are not trajectory-selection
conditions. SOP7 later combines the selected future with the robot's oracle
future to calculate risk labels.

SOP6 receives only the safe history render input. It cannot receive the selected
angle, attempt count, rejection counts, selected target future, or future risk.

## Output

An accepted result contains only mother ID, copied history/current, selected
future, selected angle, and accepted attempt number. A failure contains mother
ID, `no_legal_future`, 32 attempts, and rejection counts. Each mother yields
zero or one scenario; no candidate lists or paired variants are published.
