# SOP06: Single-Scene History BEV Rendering

SOP6 renders one model-visible history BEV for each finalized SOP5 scenario. It
does not synthesize alternative scenes and it does not assign risk labels.

## Goal

Take one accepted SOP5 scenario and produce one `RenderedObservation` from
deployment-available history. A mother can therefore contribute at most one SOP6
BEV, because SOP5 already limits it to at most one finalized scenario.

## Input

The renderer receives a model-safe input containing:

```text
sample_id, mother_id, split
BaseState
observed static occupancy
dynamic-object histories and footprint specs
per-object history-observed masks
optional sensor/structural blind-spot configuration
```

For Regime A, the hidden target is absent from the history scene whether the
oracle future contains it or not. For Regime B, the fixed observed target history
is present only at frames marked observed by SOP5. Exact poses after occlusion do
not cross the renderer boundary; ignored slots use deterministic last-observation
carry-forward values. Static geometry and visible context come from the finalized
scene.

## Rendering

1. Validate sample/mother/split identity and the history/spec mapping.
2. Call the existing history-only observation renderer with static occupancy,
   dynamic histories/specs, sensor configuration, and base configuration.
3. Publish the resulting BEV/state observation under the same sample ID.

The core renderer must receive no target future, robot future, oracle world,
selected angle, attempt count, rejection count, collision result, or risk class.

## Downstream Boundary

SOP7 joins the rendered observation with the corresponding oracle world and
trajectory to calculate collision, clearance, near-miss, and any training label.
This join occurs after the model-safe SOP6 render boundary.

## Non-goals

- No paired variants or complete-pair requirement.
- No trajectory rotation, sampling, retry, or class balancing.
- No risk calculation, label generation, or training-weight calculation.
- No extra BEV for an empty/present sibling of the same mother.
