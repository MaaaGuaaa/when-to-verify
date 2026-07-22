# Learning When to Verify

Decision-calibrated hidden-risk learning for robot local planning under occlusion.

This project studies when a mobile robot should execute a local trajectory, perform a short verification action, or reject the current plan when pedestrians may be hidden behind environmental occluders or outside the current field of view.

## Current status

The project is in active development. The schema-3 data pipeline through
SOP-07, the SOP-08–10 occupancy/risk training, calibration, and offline
evaluation path, and the SOP-11–14 verification toy/smoke path are
implemented and test-covered. Target-scale training, distributed execution,
and final paper results are not yet frozen as a cross-server release.

See [`DECISIONS.md`](./DECISIONS.md) for frozen engineering decisions and
[`docs/environment_reproduction.md`](./docs/environment_reproduction.md) for
the verified environment boundary and cross-server setup commands.

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

Python 3.10 is the currently verified interpreter line. For an exact
cross-server setup, including the tested single-GPU smoke layer, follow
[`docs/environment_reproduction.md`](./docs/environment_reproduction.md).
The minimal CPU data-pipeline setup is:

```bash
python -m pip install numpy==1.24.4 PyYAML==6.0.1 pytest==8.3.5
python -m pip install -e . --no-deps
python -m pytest tests/test_contracts.py tests/test_toy_fixture.py -q
python scripts/00_validate_contracts.py --config configs/base.yaml
```

The validation command checks schema consistency, oracle-leakage guards,
deterministic generation, hand-derived toy answers, and NPZ/JSON
serialization. This minimal command does not install PyTorch and is not a
complete training environment.

### Formal baseline comparison

The production SOP-08--10 path uses the repository ConvGRU occupancy baseline.
It does not require an external occupancy model. The fixed calibration protocol
is [`configs/prediction_protocol_production.json`](./configs/prediction_protocol_production.json).

1. Publish one evaluation-record collection per calibration/test member with
   `scripts/04_publish_risk_evaluation_records.py`.
2. Train B3/B4 with `scripts/05_train_occupancy_baseline.py --stage formal_50k`;
   supply the typed family plus explicit train/val risk, sidecar, seal, and
   authenticated snapshot roots.
3. Run `scripts/09_predict_risk.py --stage calibration` once. It writes R0,
   R1, and B1--B4 tables over one ordered calibration cohort.
4. Run `scripts/07_calibrate_risk.py` for each table with the same
   `--prediction-protocol`, placing the six sealed outputs under method-named
   directories.
5. Run `scripts/09_predict_risk.py --stage complete` with the calibration
   prediction and calibration-artifact roots. Test data is opened only after
   all six calibration artifacts pass authentication.

Use each command's `--help` for the explicit immutable input roots. Formal
artifacts remain ineligible for paper claims until the target-scale runs finish.

The exact core, test, and current training dependency versions are also
recorded in `pyproject.toml`. CUDA 11.8 requires the additional PyTorch wheel
index documented in the environment guide.

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
