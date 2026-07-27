# Learning When to Verify

Decision-calibrated hidden-risk learning for robot local planning under occlusion.

## Current Status

The working repository surface is SOP-01 through SOP-06 on the Schema 4
Long40 contract. The executable path is current through SOP05R; the SOP06
handoff must be rebuilt directly on Long40 rather than reviving its retired
short-horizon implementation.

SOP-07 through SOP-13 and the superseded short-horizon generation, training,
risk, calibration, and verification pipelines are retired from the active tree.

## Supported Entry Points

- scripts/00_freeze_thor_recording_split.py
- scripts/00_make_splits.py
- scripts/00_validate_contracts.py
- scripts/01_index_recordings.py
- scripts/02_build_long_snippet_library.py
- scripts/03_extract_base_states.py
- scripts/05_generate_events.py --generator-mode obstacle_first_teb

## Setup

Python 3.10 is the verified interpreter line.

    python -m pip install numpy==1.24.4 PyYAML==6.0.1 pytest==8.3.5
    python -m pip install -e . --no-deps
    python scripts/00_validate_contracts.py --config configs/base.yaml

Compute-heavy validation and tests must run through Slurm in this workspace.

## Documentation

- [Long40 system contract](./docs/long40_system_contract.md)
- [Current constructive visibility placement design](./docs/superpowers/specs/2026-07-26-sop05r-m5-constructive-visibility-placement-design.md)

## Scientific Scope

The active implementation covers recording splits, THOR indexing, Long40 motion
snippets, base-state extraction, lightweight-TEB event construction, visibility,
and continuous swept-footprint collision evidence. It does not claim a completed
training, calibration, closed-loop, or paper-results pipeline.
