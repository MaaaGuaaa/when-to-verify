# SOP05 Regime B State

## Frozen Decisions

- Regime B belongs to late SOP5, not SOP6.
- Regime B is selected exactly when the target is visible at the first history
  frame H0. Visibility at H1..H7 does not change the A/B identity; an H7-visible
  target remains visible to the SOP6 model input.
- The first eight history frames and current pose stay fixed; only future 32
  frames rotate about index 7.
- Turn angles follow truncated `Normal(0, (pi/12)^2)` on `[-pi, pi)`.
- Up to 32 attempts are filtered only by future environment legality.
- Sharp turns are allowed. Robot collision and risk classification occur later
  in SOP7, not in this generator.
- One mother emits at most one finalized scenario. There are no pairs.

## Current Progress

- Module: `src/generation/sop05_seen_prior.py`
- Config: `configs/sop05_seen_prior.yaml`
- Tests: `tests/test_sop05_seen_prior.py`,
  `tests/test_sop05_scenario_stage_boundary.py`, and focused integration cases
  in `tests/test_sop06_pipeline.py`.
- Slurm validation: `38 passed` for B, `1 passed` for the stage boundary, and
  `5 passed` for the focused SOP6/SOP7 handoff and spec-boundary cases.
- Final H0-only shards:
  `outputs/sop05_final_blindspot_train_m6_first10k_p30_initial_visibility_v3`
  and
  `outputs/sop05_final_blindspot_train_m6_after10k_115004_p30_initial_visibility_v3`.
- Across `125,004` unique mothers, B contains `119,548` mothers: `79,210`
  accepted scenes and `40,338` accounted deficits. No mother is unsupported for
  H7 visibility.
- The two final shards contain `82,679` accepted A+B scenes in total, have zero
  mother-ID overlap, and both carry `COMPLETE.json` with valid checksums.
- The earlier `*_p30_20260727_v1` and `*_p30_fast_v1` releases used the wrong
  any-history-visible boundary and must not be consumed as the final A+B set.
- The `*_initial_visibility_v2` shards also must not be consumed: their H0
  classification was correct, but they incorrectly excluded 13,402 H7-visible
  mothers.

## Next Step

SOP6 consumes the two v3 shards. They cover all `125,004` unique mothers with
zero overlap and no unsupported-history records.
