# SOP08--10 Formal Baselines and Shared Prediction Tables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add authenticated evaluation records, formal B3/B4 50k training with validation checkpoint selection, and one producer for risk-model/B1--B4 calibration and test prediction tables.

**Architecture:** Keep risk samples, model inputs, and occupancy sidecars unchanged. Replay the oracle/renderer triple into an independently sealed evaluation-record collection. Extend the single-device occupancy trainer with a gated two-phase formal path, then use one authenticated cohort stream to generate six method tables and one shared calibration protocol.

**Tech Stack:** Python 3.10, NumPy, PyTorch 2.0, pytest, existing JSON/checksum/atomic-publication helpers, Slurm; no new dependencies.

---

### Task 1: Integrate and lock the server runtime fixes

**Files:**
- Modify: `src/datasets/risk_dataloader.py`
- Modify: `src/training/distributed.py`
- Test: `tests/test_distributed.py`
- Test: `tests/test_occupancy_production_training.py`

- [ ] **Step 1: Verify the supplied patch is already applied**

```bash
git apply --check when-to-verify-training-fixes-20260722.patch
git diff --check
```

Expected: the patch check reports that its four changes are already present;
`git diff --check` exits 0. If the patch check instead succeeds, stop before
applying it twice.

- [ ] **Step 2: Run the patch regression tests**

```bash
.conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_distributed.py \
  tests/test_occupancy_production_training.py
```

Expected: all tests pass, including eager process-group synchronization,
long-timeout Gloo control broadcast, and float32 primitive restoration.

- [ ] **Step 3: Compile the touched modules**

```bash
.conda-envs/sop4-risk/bin/python -m py_compile \
  src/datasets/risk_dataloader.py src/training/distributed.py \
  tests/test_distributed.py tests/test_occupancy_production_training.py
```

Expected: exit 0 with no generated files outside ignored `__pycache__`.

- [ ] **Step 4: Commit only the runtime patch files**

```bash
git add -- src/datasets/risk_dataloader.py src/training/distributed.py \
  tests/test_distributed.py tests/test_occupancy_production_training.py
git diff --cached --check
git commit -m "fix(training): harden distributed setup and query replay"
```

Do not stage `.gitignore`, `src/planning/`, verification tests, or the patch
file itself.

### Task 2: Publish authenticated evaluation-record collections

**Files:**
- Create: `src/datasets/risk_evaluation_store.py`
- Modify: `scripts/04_generate_risk_dataset.py`
- Create: `scripts/04_publish_risk_evaluation_records.py`
- Test: `tests/test_risk_evaluation_store.py`
- Test: `tests/test_evaluation_record_cli.py`

- [ ] **Step 1: Write the failing collection-contract tests**

Add tests with these exact behaviors:

```python
def test_evaluation_collection_round_trips_against_sealed_shards(tmp_path): ...
def test_evaluation_collection_rejects_label_or_sample_id_drift(tmp_path): ...
def test_evaluation_collection_rejects_unknown_files_and_partial_publish(tmp_path): ...
def test_replay_publisher_refuses_risk_semantic_digest_mismatch(tmp_path): ...
def test_replay_cli_publishes_records_without_changing_risk_collection(tmp_path): ...
```

Each test must construct records through the existing aligned triple builder
or the formal fixture, assert exact ordered IDs and labels, and verify that a
failed publication leaves no destination directory.

- [ ] **Step 2: Run only the new tests to confirm RED**

```bash
.conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_risk_evaluation_store.py tests/test_evaluation_record_cli.py
```

Expected: collection imports/API failures, not fixture construction errors.

- [ ] **Step 3: Implement the immutable store API**

Implement `LoadedRiskEvaluationCollection`,
`publish_risk_evaluation_collection()`, and
`load_risk_evaluation_collection()` with these checks:

- exact `risk_evaluation_record_collection_v1` manifest and marker schemas;
- one record per authenticated risk sample in shard and row order;
- `validate_production_evaluation_record()` for every row;
- equality of record collision/severity/clearance fields and risk-shard labels;
- independent ordered sample-ID and collection semantic digests;
- regular-file/no-symlink checks, exact recursive file coverage, checksums,
  marker-last publication, and atomic no-replace rename;
- optional expected risk manifest digest must match before reading records.

Use existing canonical JSON, checksum, directory-fsync, and atomic rename
helpers rather than adding a second publication mechanism.

- [ ] **Step 4: Add replay output to the generation boundary**

Extend `scripts/04_generate_risk_dataset.py` with an explicit evaluation-record
output option. When enabled, call the aligned triple builder and write records
by the same shard index and row order as the risk/sidecar output. When
disabled, retain byte-compatible risk-only and risk-plus-sidecar behavior.
Reject missing replay inputs, reference digest mismatch, and an existing
immutable evaluation destination.

- [ ] **Step 5: Add the explicit replay/publish CLI**

Create `scripts/04_publish_risk_evaluation_records.py` with required explicit
arguments for SOP03/SOP04/SOP05 roots, config/paired config, split, seed,
reference risk seal/collection, replay shard range, and output root. The CLI
must compare replayed risk shard semantic digests before calling the store
publisher and return exit code 2 for contract errors.

- [ ] **Step 6: Run the collection tests and commit**

```bash
.conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_risk_evaluation_store.py tests/test_evaluation_record_cli.py \
  tests/test_production_evaluation_metadata.py
git diff --check
git add -- src/datasets/risk_evaluation_store.py \
  scripts/04_generate_risk_dataset.py \
  scripts/04_publish_risk_evaluation_records.py \
  tests/test_risk_evaluation_store.py tests/test_evaluation_record_cli.py
git commit -m "feat(risk-data): publish authenticated evaluation records"
```

Expected: all new and existing evaluation-record tests pass.

### Task 3: Implement formal B3/B4 occupancy training and selection

**Files:**
- Modify: `src/training/occupancy_trainer.py`
- Modify: `scripts/05_train_occupancy_baseline.py`
- Modify: `configs/occupancy_baseline_production.yaml`
- Create: `tests/test_occupancy_formal_training.py`
- Modify: `tests/test_occupancy_production_trainer.py`

- [ ] **Step 1: Write failing formal-gate and selection tests**

Add tests with these exact assertions:

```python
def test_formal_occupancy_requires_family_val_and_exact_50k(tmp_path): ...
def test_formal_b3_selects_lowest_validation_occupancy_loss(tmp_path): ...
def test_formal_b4_selects_lowest_validation_collision_loss_and_freezes_b3(tmp_path): ...
def test_formal_occupancy_never_loads_calibration_or_test(tmp_path, monkeypatch): ...
def test_formal_best_checkpoint_reloads_with_selection_provenance(tmp_path): ...
```

Use a small deterministic training-double fixture for selection arithmetic;
do not weaken the production exact-50k gate. Assert earliest-epoch tie
breaking, train-only class weights, selected/final state distinction, B3
state digest equality before/after B4, and zero test sample usage.

- [ ] **Step 2: Run the new tests to confirm RED**

```bash
.conda-envs/sop4-risk/bin/python -m pytest -q tests/test_occupancy_formal_training.py
```

Expected: formal stage is rejected as not implemented or the formal API is
absent; no test may pass merely because the existing smoke path runs.

- [ ] **Step 3: Add formal configuration and provenance gates**

Extend `ProductionOccupancyTrainingConfig` and the production config parser so
`formal_50k` requires family/val digests, sidecar digests, exactly 50,000
selected train IDs, and validated `PROVEN` leakage status. Keep smoke and
overfit behavior unchanged. Add a new explicit formal checkpoint layout
version rather than silently changing the smoke schema.

- [ ] **Step 4: Implement the B3 phase**

Use the existing occupancy batch/snapshot interfaces. Compute occupancy and
collision positive weights from train only, run configured accumulation,
evaluate full val at each epoch boundary, and retain the minimum finite
validation occupancy loss with earliest-step tie breaking. Persist phase
cursor, optimizer, RNG, loss history, and best B3 state in training state.

- [ ] **Step 5: Implement the frozen-B3 B4 phase**

Restore the selected B3 state, set every B3 parameter to
`requires_grad=False`, train B4 on predicted occupancy plus robot footprints,
evaluate weighted collision loss on full val, and retain the selected B4 state
with the same tie rule. Fail if B3 changes, any loss/gradient is non-finite, or
a calibration/test object is accessed.

- [ ] **Step 6: Publish formal best/final artifacts and wire the CLI**

Add `best_checkpoint.pt`, `final_checkpoint.pt`, interval state, selection
records, exact checksums, and atomic manifest publication. Update
`scripts/05_train_occupancy_baseline.py` to require explicit train/val seal,
collection, sidecar, family, and cache roots for formal mode and to print only
the single-process publication result. Keep WORLD_SIZE>1 rejected for
occupancy.

- [ ] **Step 7: Run formal unit/integration tests and commit**

```bash
.conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_occupancy_formal_training.py \
  tests/test_occupancy_production_trainer.py \
  tests/test_occupancy_production_training.py
git diff --check
git add -- src/training/occupancy_trainer.py \
  scripts/05_train_occupancy_baseline.py \
  configs/occupancy_baseline_production.yaml \
  tests/test_occupancy_formal_training.py \
  tests/test_occupancy_production_trainer.py
git commit -m "feat(occupancy): add formal validation-selected B3 and B4"
```

Expected: all formal fixture tests pass; existing smoke tests remain green.

### Task 4: Define prediction-table and protocol contracts

**Files:**
- Create: `src/evaluation/prediction_tables.py`
- Modify: `src/calibration/split_conformal.py`
- Modify: `scripts/07_calibrate_risk.py`
- Test: `tests/test_prediction_tables.py`
- Test: `tests/test_production_calibration_contract.py`

- [ ] **Step 1: Write failing contract tests**

Add tests for a shared ordered cohort, method-independent evaluation fields,
baseline-spec/checkpoint provenance, protocol digest construction, and rejection
of mismatched family, cohort, prediction key, alpha, or grouped-calibration
configuration.

- [ ] **Step 2: Run contract tests RED**

```bash
.conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_prediction_tables.py tests/test_production_calibration_contract.py
```

Expected: the new public API or production protocol arguments are absent.

- [ ] **Step 3: Implement immutable protocol primitives**

Add typed protocol/cohort/table records, canonical row serialization, stable
sample-ID and semantic digests, and strict joins against the authenticated
evaluation-record collection and risk/sidecar members. Keep calibration fit
separate from test loading and preserve the existing toy API.

- [ ] **Step 4: Add production calibration validation**

Extend split-conformal loading and calibration CLI validation so every method
binds one shared protocol and calibration cohort, and test calibration cannot
run until the selected formal checkpoints and protocol inputs are authenticated.

- [ ] **Step 5: Run focused tests and commit**

```bash
.conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_prediction_tables.py tests/test_production_calibration_contract.py \
  tests/test_split_conformal.py tests/test_calibration_isolation.py
git diff --check
git add -- src/evaluation/prediction_tables.py src/calibration/split_conformal.py \
  scripts/07_calibrate_risk.py tests/test_prediction_tables.py \
  tests/test_production_calibration_contract.py
git commit -m "feat(evaluation): bind shared prediction protocol"
```

### Task 5: Produce unified six-method prediction tables

**Files:**
- Create: `scripts/09_predict_risk.py`
- Modify: `scripts/10_eval_offline.py`
- Test: `tests/test_prediction_producer.py`
- Test: `tests/test_production_eval_cli.py`

- [ ] **Step 1: Write failing producer tests**

Cover all six methods, exact calibration/test sample-ID equality, evaluation
record joins, deterministic reruns, selected-checkpoint binding, and refusal to
read test rows before calibration artifacts are sealed.

- [ ] **Step 2: Run producer tests RED**

```bash
.conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_prediction_producer.py tests/test_production_eval_cli.py
```

- [ ] **Step 3: Implement shared six-method producer**

Load one authenticated cohort per split, score R0/R1 and B1--B4 through their
existing model/baseline APIs, write method tables with common fields and
provenance, and publish the complete immutable prediction collection atomically.

- [ ] **Step 4: Gate offline comparison**

Require matching family, member, cohort, evaluation-record, calibration, and
protocol digests before calculating test metrics. Reject incomplete or mixed
prediction publications without filtering rows.

- [ ] **Step 5: Run integration tests and commit**

```bash
.conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_prediction_producer.py tests/test_production_eval_cli.py \
  tests/test_risk_baselines.py tests/test_risk_metrics.py
git diff --check
git add -- src/evaluation/prediction_tables.py scripts/09_predict_risk.py \
  scripts/10_eval_offline.py tests/test_prediction_producer.py \
  tests/test_production_eval_cli.py
git commit -m "feat(evaluation): publish unified baseline predictions"
```

### Task 6: Complete validation ladder and Slurm checks

**Files:**
- Modify: `configs/occupancy_baseline_production.yaml`
- Modify: `configs/risk_model_production.yaml`
- Modify: `README.md` or existing experiment documentation
- Test: existing focused and integration suites

- [ ] **Step 1: Validate configuration and CLI help**

Check formal gates, explicit paths, no hidden discovery, and stable output
layout without touching raw or accepted publications.

- [ ] **Step 2: Run fixture and regression ladder**

Run new collection, occupancy, producer, calibration, strict-loader, risk-model,
and runtime-patch tests, then compile all changed Python modules.

- [ ] **Step 3: Run authenticated smoke checks**

Exercise the smallest existing sealed fixture through replay, formal gate
failure/success doubles, calibration, and offline comparison; record failures
without claiming target-scale scientific completion.

- [ ] **Step 4: Inspect Slurm launch paths**

Verify formal mode is single-device as specified, R0/R1 distributed paths retain
the runtime patch, and commands expose configuration, seeds, and provenance.

- [ ] **Step 5: Finish status and diff audit**

Run `git status --short`, `git diff --check`, inspect only intended files, clean
temporary `.tmp/agent` outputs, and report every changed file and validation
result. Do not push or commit unrelated concurrent changes.
