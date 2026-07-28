# SOP06 Single-Scene State

## Frozen Decisions

- SOP5 owns A/B scenario synthesis.
- SOP6 renders one history BEV per finalized scenario.
- SOP7 computes risk labels from the safe render and oracle future.
- A mother produces at most one scenario and therefore at most one SOP6 BEV.
- Paired variants are not part of the current release.

## Current Progress

- Renderer integration: `src/generation/sop06_pipeline.py`
- Core renderer: `src/generation/observation_renderer.py`
- Boundary/integration tests: `tests/test_sop05_scenario_stage_boundary.py` and
  focused cases in `tests/test_sop06_pipeline.py`.
- Slurm validation: `1 passed` for the stage boundary and `5 passed` for the
  single-renderer/SOP7 handoff and spec-boundary cases.

## Next Step

Let SOP7 build labels from the oracle side without widening the SOP6 renderer
interface.
