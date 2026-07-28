# Learning When to Verify

Decision-calibrated hidden-risk learning for robot local planning under occlusion.

## Current Status

The working repository surface includes the Schema 4 Long40 path through the
SOP08 occupancy baselines. Event generation is current through SOP05R; the
SOP06/SOP07 producers must still be rebuilt directly on Long40. SOP08 consumes
authenticated Long40 risk shards, seals, and sidecars without reviving their
retired short-horizon producers.

The SOP09-SOP13 risk-model, calibration CLI, verification, and closed-loop
pipelines remain retired, as do all superseded short-horizon implementations.

## Supported Entry Points

- scripts/00_freeze_thor_recording_split.py
- scripts/00_make_splits.py
- scripts/00_validate_contracts.py
- scripts/01_index_recordings.py
- scripts/02_build_long_snippet_library.py
- scripts/03_extract_base_states.py
- scripts/04_backfill_risk_sidecars.py
- scripts/04_seal_risk_dataset.py
- scripts/05_audit_risk_collection.py
- scripts/05_generate_events.py --generator-mode obstacle_first_teb
- scripts/05_train_occupancy_baseline.py

## Setup

Python 3.10 is the verified interpreter line.

    python -m pip install -e '.[test,train]'
    python scripts/00_validate_contracts.py --config configs/base.yaml

Compute-heavy validation and tests must run through Slurm in this workspace.

## Documentation

- [Long40 system contract](./docs/long40_system_contract.md)
- [Current constructive visibility placement design](./docs/superpowers/specs/2026-07-26-sop05r-m5-constructive-visibility-placement-design.md)

## Scientific Scope

The active implementation covers recording splits, THOR indexing, Long40 motion
snippets, base-state extraction, lightweight-TEB event construction, visibility,
continuous swept-footprint collision evidence, authenticated risk-dataset input,
and SOP08 occupancy baselines. It does not claim a completed SOP09 risk model,
calibration CLI, verification, closed-loop, or paper-results pipeline.
