# SOP06 Single-Scene Contracts

## Input Contract

`Sop06SingleRendererInput` is the only object passed to the core renderer. It
contains `sample_id`, `mother_id`, `split`, `BaseState`, observed static
occupancy, dynamic histories, dynamic specs, and optional sensor configuration.
It also carries a boolean observed mask for every included dynamic history.

- `sample_id` is unique in the release.
- `(split, mother_id)` occurs at most once across accepted entries and failures.
- Static occupancy matches the finalized scene and has the renderer grid shape.
- Dynamic-history and dynamic-spec mappings have the same object IDs.
- All history arrays use the configured eight-frame layout and finite floats.
- A false observed-mask slot is ignored by footprint rasterization and ray
  casting. Its finite pose value is only a deterministic carry-forward
  placeholder and cannot encode the hidden ground-truth pose.
- Regime A does not include the target history. Regime B includes only its
  SOP5-authenticated observed frames and must be false at current index 7.

## Safe/Oracle Separation

An internal publication envelope may retain the selected oracle future so SOP7
can label the rendered scene. The envelope must split it before calling the
renderer. The renderer input and model-visible metadata must not expose keys or
values for future trajectories, oracle state, sampling angle, attempts,
rejections, collision, clearance, or risk.

## Output Contract

`render_sop06_single_input` and `render_sop06_single_publication` return one
`RenderedObservation`. Rendering does not change `sample_id`, synthesize a new
world, or assign a label. A failure at SOP5 produces no SOP6 observation and is
kept only in release accounting.

## Ownership Contract

- SOP5 owns A/B trajectory synthesis and the one-scenario-per-mother decision.
- SOP6 owns history-only BEV rendering.
- SOP7 owns oracle collision/clearance computation and labels.

Legacy paired rendering helpers may remain for archived artifacts, but they are
not inputs, outputs, or acceptance conditions for this release.
