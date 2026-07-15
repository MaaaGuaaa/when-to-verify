# Learning When to Verify

Decision-calibrated hidden-risk learning for robot local planning under occlusion.

This project studies when a mobile robot should execute a local trajectory, perform a short verification action, or reject the current plan when pedestrians may be hidden behind environmental occluders or outside the current field of view.

## Current status

The project is in active development. SOP-00 is complete:

- frozen data contracts and channel ordering
- deterministic seed and stable-ID utilities
- strict observed/oracle information separation
- reproducible base configuration
- hand-verifiable toy risk and verification-value fixtures
- 34 passing contract tests

See [`STATUS.md`](./STATUS.md) for the current implementation gate and [`DECISIONS.md`](./DECISIONS.md) for frozen engineering decisions.

## Repository layout

```text
src/          Current implementation
configs/      Reproducible configuration
scripts/      Supported command-line entry points
tests/        Unit tests and deterministic fixtures
docs/         Current method specification and Agent SOPs
reports/      Curated, reproducible public results
```

Local datasets, generated runs, environments, previous implementations, and historical documents are intentionally excluded from Git.

## Quick start

Python 3.10 or newer is required.

```bash
python -m pip install -e ".[test]"
python -m pytest tests/test_contracts.py tests/test_toy_fixture.py -q
python scripts/00_validate_contracts.py --config configs/base.yaml
```

The validation command checks schema consistency, oracle-leakage guards, deterministic generation, hand-derived toy answers, and NPZ/JSON serialization.

## Documentation

- [`docs/event_centered_blind_spot_implementation_spec.md`](./docs/event_centered_blind_spot_implementation_spec.md): scientific and system specification
- [`docs/parallel_acceleration_implementation_plan.md`](./docs/parallel_acceleration_implementation_plan.md): parallel implementation plan
- [`docs/event_centered_blind_spot_agent_sops.md`](./docs/event_centered_blind_spot_agent_sops.md): SOP-00 through SOP-16

## Scientific scope

The first version uses 2D robot-centric BEV state, differential-drive local trajectories, real pedestrian motion snippets, procedural occlusion, trajectory-conditioned hidden-risk prediction, conformal calibration, and simulator-defined counterfactual verification value.

Continuous risk severity is an oracle-defined target, not a real collision probability. Scenario-bank verification value is a simulator-defined decision target, not exact Bayesian ground truth. The method does not claim unconditional safety.

## Reproducibility policy

- Split recordings and participants before generating any sample
- Fit learned statistics only on the appropriate training split
- Keep model inputs separate from oracle and future-label information
- Record configuration, seed, schema version, source IDs, and artifact digests
- Generate public figures and tables from structured results
- Keep debug output and machine-specific files outside Git

## License

A project license has not yet been selected. Add one before making the repository public.
