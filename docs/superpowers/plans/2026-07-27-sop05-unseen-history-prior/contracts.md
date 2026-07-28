# SOP05 Regime A Contracts

## Fixed Configuration

```yaml
schema_version: "4.0.0"
generator_version: "sop05_unseen_history_prior_v1"
contract_version: "sop05_unseen_history_prior_contract_v1"
p_hidden_human: 0.30
max_attempts_per_mother: 32
max_variants_per_mother: 1
hard_total_sample_cap: 125000
manifest_targets: [50000, 100000, 125000]
seed: 42
```

The cap applies to the combined A+B release, not to each regime separately.
`unseen_in_history_window` is retained as a compatibility label; eligibility
means H0 is hidden, not that every later history frame is hidden.

## Input And Transform

- The shared release assigns A exactly when H0 is hidden. Visibility at H1..H7
  does not change the source A/B identity.
- Input target arrays are finite `float32` arrays with shapes `[40,2]`,
  `[40,2]`, and `[40]`; layout is `history8_current7_future32_v1`.
- `transform_long40_target(target, angle_rad)` uses one finite angle in
  `[-pi, pi)`, preserves shapes/dtypes, does not mutate the source, and keeps
  position 7 fixed.
- The mother supplies production geometry and visibility inputs. Do not recreate
  a separate collision or visibility approximation.

## Candidate Decision

`evaluate_candidate` returns exactly one of:

```text
legal
nonfinite_or_out_of_bounds
obstacle_collision
robot_history_collision
history_visible
```

`obstacle_collision` covers static occupancy, occluder occupancy, and represented
context over the Long40 sweep. `robot_history_collision` covers only target
indices `0..7`. Robot collision after index 7 is not a SOP05 rejection.

## Sampling And Output

- `u >= 0.30`: output one target-empty realization.
- `u < 0.30`: try at most 32 independent uniform angles and output the first
  legal target-present realization.
- All failures yield `no_legal_angle`, zero published scenarios, and complete
  rejection accounting.
- One mother yields zero or one scenario. It never yields empty and present
  siblings or an alternate candidate list.

The published scenario separates model-safe history from oracle-only future.
Target future, angle, attempt history, rejection counts, collision evidence, and
risk class are unavailable to SOP6 rendering and model-visible metadata. SOP7
alone turns the finalized oracle future into labels.
