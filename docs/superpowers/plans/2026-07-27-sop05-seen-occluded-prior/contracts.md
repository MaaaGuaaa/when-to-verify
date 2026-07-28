# SOP05 Regime B Contracts

## Fixed Configuration

`configs/sop05_seen_prior.yaml` must encode only:

```yaml
schema_version: "4.0.0"
required_history_regime: "seen_then_occluded"
trajectory:
  history_steps: 8
  current_index: 7
  future_steps: 32
  dt_s: 0.2
angle_prior:
  kind: "truncated_normal"
  mean_rad: 0.0
  sigma_rad: 0.2617993877991494 # pi / 12
  min_rad_inclusive: -3.141592653589793
  max_rad_exclusive: 3.141592653589793
sampling:
  seed_namespace: "sop05/seen-prior/continuous/v1"
  max_attempts_per_mother: 32
  max_variants_per_mother: 1
```

No angle grid, class quota, class weight, or risk threshold belongs in this
configuration.
`seen_then_occluded` is retained as a compatibility label; eligibility means H0
is visible and does not require H7 to be hidden.

## Source Contract

- Source is `seen_then_occluded` and uses finite floating `[8,3]`, `[3]`, and
  `[32,3]` target arrays of one dtype.
- `current_pose` is byte-equal to history index 7.
- History frame H0 is visible. H1..H7 do not determine the A/B identity, and
  their actual visibility mask is preserved through SOP6 rendering.
- The environment holds a Long40 grid, target footprint, static occupancy,
  occluder occupancy, and explicit context sweeps. It has no robot trajectory
  or risk field.

## Transform And Gate

- `transform_seen_prior_future` copies history/current and rotates only all 32
  future positions and yaws about history index 7.
- The only gate outcomes are `future_nonfinite`, `future_out_of_bounds`,
  `future_static_collision`, `future_occluder_collision`, and
  `future_context_collision`.
- Robot collision, clearance, future visibility, and motion smoothness must not
  affect the gate.

## Sampling And Handoff

- Every mother consumes at most 32 normal draws; out-of-support draws consume a
  slot.
- The first legal future is accepted. Exhaustion yields one failure record and
  no scene.
- An accepted `SeenPriorResult` contains exactly
  `mother_id`, `history_poses`, `current_pose`, `future_poses`, `theta_rad`, and
  `accepted_attempt`.
- It contains no robot pose, collision evidence, risk class, sampler weight, or
  quota field.

SOP6 receives history-safe rendering data only. SOP7 is the sole consumer that
uses oracle robot/target futures to derive labels.
