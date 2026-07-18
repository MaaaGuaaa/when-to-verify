# SOP08--10 Toy-First Risk Learning Implementation Plan

> **For agentic workers:** use test-driven development task by task. All Python
> tests and generators run through Slurm; the login node is for Git and text
> inspection only.

**Goal:** Build a deterministic, schema-shaped toy software closure for SOP08,
SOP09, and SOP10 while failing closed on unavailable production risk data.

**Architecture:** Implement disjoint occupancy-baseline, risk-model, and
calibration/evaluation components connected by strict tensors and prediction
records. Model inputs use only deployable history/state/trajectory data. Toy
future occupancy is a label-only sidecar. Every production loader rejects the
ambiguous current shard layout until a v2 publication contract exists.

**Tech stack:** Python 3.10, NumPy, existing PyYAML and pytest, preinstalled
PyTorch 2.0.1, Slurm CPU jobs, exact Git executable from the workspace SOP.

---

### Task 1: Shared deterministic toy fixtures

**Files:**
- Create: `src/datasets/toy_risk_learning.py`
- Create: `tests/fixtures/toy_risk_learning.py`

- [ ] Build schema-valid `RiskSample` objects, validate, then collate their real
  field names; keep the test fixture a thin wrapper.
- [ ] Build 8-history/15-future small-grid batches using frozen channel names.
- [ ] Cover collision, near miss, temporal-safe, same-area-safe,
  irrelevant-hidden, empty, and OOD cases.
- [ ] Keep hidden-risk future occupancy label-only and bind both sidecars by
  sample ID with endpoint times 0.2--3.0 s.
- [ ] Generate disjoint recording/session/source-object/snippet/base-state/
  pair-group/seed identities for every split.
- [ ] Validate shape, float32, finite, bounds, determinism, and split isolation.

### Task 2: SOP08 aggregation and analytic baselines

**Files:**
- Create: `src/models/__init__.py`
- Create: `src/models/occupancy_aggregation.py`
- Create: `src/models/occupancy_baseline.py`
- Create: `src/evaluation/__init__.py`
- Create: `src/evaluation/risk_baselines.py`
- Create: `tests/test_occupancy_aggregation.py`
- Create: `tests/test_risk_baselines.py`

- [ ] RED: hand-computed weighted-sum and probabilistic-union tests.
- [ ] RED: zero-risk, monotonicity, bounds, duplicate-mask normalization.
- [ ] RED: last-observation hold and age-decay formulas/channel lookup.
- [ ] Implement strict NumPy/Torch aggregation and analytic predictors.
- [ ] Freeze B1/B2 mapping and normalized weighted-sum time weights.
- [ ] Run targeted tests on Slurm and record the job/result.

### Task 3: SOP08 learned occupancy baseline

**Files:**
- Modify: `src/models/occupancy_baseline.py`
- Create: `configs/occupancy_baseline.yaml`
- Create: `scripts/05_train_occupancy_baseline.py`
- Create: `tests/test_occupancy_baseline.py`
- Create: `tests/test_occupancy_training_smoke.py`

- [ ] RED: ConvGRU forward shape/dtype/finite/probability tests.
- [ ] RED: label isolation and checkpoint provenance mismatch tests.
- [ ] Implement B3 ConvGRU + hand aggregation and B4 learned aggregator.
- [ ] Implement deterministic toy trainer/checkpoint v2 with toy-only digest.
- [ ] Overfit a bounded toy batch and show material loss reduction.
- [ ] Write structured toy artifacts without overwriting existing paths.

### Task 4: SOP09 dataloader, R0/R1, and losses

**Files:**
- Create: `src/models/bev_encoder.py`
- Create: `src/models/risk_model.py`
- Create: `src/models/losses.py`
- Create: `src/datasets/risk_dataloader.py`
- Create: `tests/test_risk_model.py`
- Create: `tests/test_risk_losses.py`

- [ ] RED: exact channel/schema/shape/dtype/finite/oracle isolation validation.
- [ ] RED: R0/R1 output contract and non-crossing quantiles.
- [ ] RED: Q95 can remain below one and monotone parameter gradients are finite.
- [ ] RED: hand-computed pinball and BCE composition.
- [ ] Implement R0 and R1 with one unified output structure.
- [ ] Implement checkpoint v2 identity and fail-closed reload validation.

### Task 5: SOP09 toy training closure

**Files:**
- Create: `configs/risk_model.yaml`
- Create: `scripts/06_train_risk_model.py`
- Create: `tests/test_risk_training_smoke.py`

- [ ] RED: deterministic toy overfit and checkpoint round-trip tests.
- [ ] Implement seeded train/validation loops without using test for selection.
- [ ] Require decreasing loss, zero crossings, and identical reload output.
- [ ] Emit R0/R1 structured comparison; do not claim real-data ranking.

### Task 6: SOP10 split and grouped conformal calibration

**Files:**
- Create: `src/calibration/__init__.py`
- Create: `src/calibration/split_conformal.py`
- Create: `src/calibration/grouped_calibration.py`
- Create: `tests/test_split_conformal.py`
- Create: `tests/test_calibration_isolation.py`

- [ ] RED: hand-computed one-sided residual and finite-sample rounding tests.
- [ ] RED: calibration/test overlap, test-label perturbation, sparse fallback,
  artifact tampering, and old-version rejection.
- [ ] Implement pure NumPy global and one-dimension grouped calibration.
- [ ] Record fitted identities, counts, fallback reasons, and semantic digest.

### Task 7: SOP10 metrics and failure subsets

**Files:**
- Create: `src/evaluation/risk_metrics.py`
- Create: `tests/test_risk_metrics.py`

- [ ] RED: hand-computed AUROC/AP/Brier/NLL/ECE/coverage/tightness examples.
- [ ] RED: distinguish trapezoidal AUPRC from average precision.
- [ ] RED: false-safe, pair eligibility, undefined subset, and no-object rules.
- [ ] Implement finite JSON-safe metrics and structured reason reporting.
- [ ] Compare toy main and toy occupancy under the same calibration protocol.

### Task 8: SOP10 CLIs and end-to-end toy run

**Files:**
- Create: `scripts/07_calibrate_risk.py`
- Create: `scripts/10_eval_offline.py`

- [ ] Implement validated prediction-table input and atomic artifacts.
- [ ] Run train, calibration, and evaluation twice through Slurm.
- [ ] Require stable semantic digests and strict source isolation.
- [ ] Report G2 scientific metrics as `not_evaluated_real_data`, never pass.

### Task 9: Final verification and handoff

- [ ] Run all new SOP08--10 tests plus direct contract/toy-fixture dependencies
  through Slurm.
- [ ] Check shapes, dtypes, NaN/Inf, probability bounds, deterministic seeds,
  non-crossing quantiles, oracle isolation, and split/source isolation.
- [ ] Confirm production mode rejects ambiguous/missing v2 inputs.
- [ ] Remove temporary files; keep generated toy outputs ignored and outside
  the commit.
- [ ] Inspect exact Git diff and stage only owned files.
- [ ] Create a local commit on `feat/sop-08-10-risk-learning`.
- [ ] Return commit, files, Slurm commands/results, artifact paths, limitations,
  Contract changes requested, and the next safe production task.
