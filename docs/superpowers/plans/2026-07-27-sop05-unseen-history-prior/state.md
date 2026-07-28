# SOP05 Regime A State

## Frozen Decisions

- Regime A belongs to late SOP5, not SOP6.
- Regime A is selected exactly when the target is not visible at the first
  history frame H0. Visibility at H1..H7 does not change the A/B identity.
- Hidden-pedestrian presence prior is exactly `0.30`.
- Present trajectories use independent continuous `Uniform[-pi, pi)` rotations
  of all 40 frames around index 7, with at most 32 attempts.
- Reject environment/context contact, history robot contact, and any history
  visibility; do not reject future robot collision.
- One mother publishes at most one scene. No paired variants, class balancing,
  class weights, or fallback empty sample are allowed.

## Current Progress

- Module: `src/generation/sop05_unseen_prior.py`
- Config: `configs/sop05_unseen_prior.yaml`
- Tests: `tests/test_sop05_unseen_prior.py` and
  `tests/test_sop05_scenario_stage_boundary.py`
- Slurm validation: `56 passed` for the A module and `1 passed` for the stage
  boundary test.
- Final H0-only shards:
  `outputs/sop05_final_blindspot_train_m6_first10k_p30_initial_visibility_v3`
  and
  `outputs/sop05_final_blindspot_train_m6_after10k_115004_p30_initial_visibility_v3`.
- Across `125,004` unique mothers, A contains `5,456` mothers: `3,469`
  accepted scenes (`3,328` empty and `141` pedestrian-present) and `1,987`
  accounted deficits. Of `4,813` A mothers reaching the presence draw,
  `1,485` took the present branch (`30.85%`); `643` lacked complete context
  before the draw.
- The earlier `*_p30_20260727_v1` and `*_p30_fast_v1` releases used the wrong
  any-history-visible boundary and must not be consumed as the final A+B set.
- The `*_initial_visibility_v2` shards also must not be consumed: their H0
  classification was correct, but they incorrectly excluded H7-visible mothers.

## Next Step

SOP6 consumes the two v3 shards. They cover all `125,004` unique mothers with
zero overlap and no unsupported-history records.
