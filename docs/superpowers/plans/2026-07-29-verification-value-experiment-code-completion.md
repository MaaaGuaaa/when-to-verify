# Verification Value Experiment Code Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the production-capable Schema-4 verification-value experiment path from finalized SOP5 releases through train-only value calibration, V0/no-ranking training, held-out evaluation, multi-seed aggregation, and evidence-bound closed-loop matrices.

**Architecture:** Keep the existing SOP11–16 implementations and add the missing boundaries around them. A verification release retains sufficient label-side decision losses for deterministic revaluation; a signed train-only calibration artifact freezes `reject_cost`; a release collection adapter applies that artifact in memory and binds it into checkpoints and evaluation outputs. Closed-loop suites carry an explicit method binding that the matrix runner verifies before any result is accepted.

**Tech Stack:** Python 3, NumPy, PyTorch, PyYAML, pytest, immutable JSON/NPZ artifacts, Slurm `srun`.

---

## File map

- Modify `configs/verify_model.yaml`: active V0 Schema-4 configuration.
- Create `configs/verify_model_no_ranking.yaml`: identical V0 ablation with `ranking_weight: 0`.
- Create `configs/verification_value_calibration.yaml`: deterministic train-only selection rules.
- Modify `src/generation/verification_pipeline.py`: retain safe target/footprint/blind-type slice metadata.
- Modify `src/generation/verification_release.py`: publish and strictly load raw revaluation records.
- Create `src/evaluation/verification_value_calibration.py` additions: select, publish, and load the frozen calibration artifact.
- Create `scripts/08_calibrate_verification_value.py`: calibration CLI.
- Create `src/datasets/verification_release_collection.py`: load release shards and apply frozen labels without changing model inputs.
- Modify `src/evaluation/verification_metrics.py`: bind calibration in checkpoint manifests and expose calibration bins.
- Modify `src/models/verification_training.py`: populate all required evaluation slices.
- Modify `src/evaluation/verification_baselines.py`: report the same slice dimensions as V0.
- Modify `scripts/09_train_verification_model.py`: production release input, explicit seed override, calibration binding.
- Modify `scripts/10_evaluate_verification_model.py`: production release input, prediction rows, bins, and authenticated run metadata.
- Create `src/evaluation/verification_experiment_aggregation.py`: strict multi-seed aggregation.
- Create `scripts/10_aggregate_verification_value.py`: aggregation CLI.
- Modify `src/evaluation/closed_loop_replay.py`: serialize method/evidence binding in replay suites.
- Modify `src/evaluation/experiment_matrix.py`: reject suites whose method binding differs from matrix declarations.
- Modify focused `tests/test_*.py` files and create tests for each new public boundary.

### Task 1: Close all active Schema-4 gaps

**Files:**
- Modify: `configs/verify_model.yaml`
- Create: `configs/verify_model_no_ranking.yaml`
- Modify: `src/datasets/verification_sources.py`
- Modify: `src/models/verification_model.py`
- Modify: `scripts/09_train_verification_model.py`
- Modify: `tests/test_verification_sources.py`
- Modify: `tests/test_verification_training_smoke.py`
- Modify: `tests/test_verification_evaluation_smoke.py`

- [ ] **Step 1: Change positive fixtures and the active model config to Schema 4**

Use `schema_version: "4.0.0"` everywhere a positive fixture is intended to load. Keep one explicit negative test:

```python
legacy = {**valid_handoff, "schema_version": "3.0.0"}
with pytest.raises(ValueError, match="schema"):
    validate_verification_collection_handoff(
        legacy_path,
        shard_dirs=(shard,),
        loaded_shards=(loaded,),
        expected_split="train",
    )
```

- [ ] **Step 2: Add the no-ranking config**

Copy every field from `configs/verify_model.yaml`, changing only:

```yaml
loss:
  ranking_weight: 0.0
```

- [ ] **Step 3: Remove stale Schema-3 wording from active code**

The public docstrings must say Schema 4 / Long40 and must not imply production runs are smoke-only.

- [ ] **Step 4: Run the pre-existing failure set through Slurm**

Run:

```bash
srun --partition=gpu --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:10:00 \
  .conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_verification_model.py tests/test_verification_losses.py \
  tests/test_verification_training_smoke.py tests/test_verification_evaluation_smoke.py \
  tests/test_verification_sources.py
```

Expected: all selected tests pass.

### Task 2: Persist raw revaluation evidence in each release

**Files:**
- Modify: `src/generation/verification_pipeline.py`
- Modify: `src/generation/verification_release.py`
- Modify: `tests/test_verification_release.py`
- Modify: `tests/test_verification_pipeline.py`

- [ ] **Step 1: Write failing release-record tests**

Assert that one accepted action exposes only label-side scalars required for revaluation:

```python
record = loaded.revaluation_records[0]
assert record.sample_id == sample.sample_id
assert record.action_id == sample.verification_action_id
assert record.realized_execute_loss == value.realized_execute_loss
assert record.unclipped_best_policy_loss == value.unclipped_best_policy_loss
assert record.action_cost == value.action_cost
assert record.original_reject_cost == value.reject_cost
```

Also mutate one scalar in `task_summary.json` and assert strict load fails by checksum or semantic validation.

- [ ] **Step 2: Add the immutable record type and `_TaskOutcome` payload**

Implement:

```python
@dataclass(frozen=True)
class VerificationRevaluationRecord:
    release_request_identity: str
    split: str
    task_id: str
    mother_id: str
    sample_id: str
    ranking_group_id: str
    action_id: str
    realized_execute_loss: float
    unclipped_best_policy_loss: float | None
    action_cost: float
    original_reject_cost: float
```

Construct records from `VerificationGroupResult.values` before discarding the group result. Validate finite non-negative losses, canonical action IDs, one record per sample, and exact task/sample/action alignment.

- [ ] **Step 3: Serialize and strictly reload records**

Store the records in the authenticated task summary and add:

```python
records = load_verification_revaluation_records(release_dir)
loaded_release = load_verification_release(release_dir)
assert len(records) == loaded_release.sample_count
assert len({record.sample_id for record in records}) == len(records)
```

The loader must first call `load_verification_release`, then reload every authenticated task checkpoint, reject duplicate sample IDs, and require record count to equal release `sample_count`.

- [ ] **Step 4: Add safe slice provenance**

For finalized inputs, retain:

```python
{
    "blind_type": publication.regime,
    "target_object_type": world.dynamic_object_specs[target_id]["object_type"],
    "target_footprint_kind": world.dynamic_object_specs[target_id]["footprint"]["kind"],
}
```

These fields are model-safe metadata only; no world identity, future pose, collision, or post-action observation is added to model inputs.

- [ ] **Step 5: Run focused release/pipeline tests through Slurm**

Run:

```bash
srun --partition=gpu --ntasks=1 --cpus-per-task=4 --mem=12G --time=00:15:00 \
  .conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_verification_release.py tests/test_verification_pipeline.py \
  tests/test_verification_input_isolation.py
```

Expected: all selected tests pass and single/fork release digests agree.

### Task 3: Build and seal train-only reject-cost calibration

**Files:**
- Modify: `src/evaluation/verification_value_calibration.py`
- Create: `scripts/08_calibrate_verification_value.py`
- Create: `configs/verification_value_calibration.yaml`
- Modify: `tests/test_verification_value_calibration.py`
- Create: `tests/test_verification_value_calibration_cli.py`

- [ ] **Step 1: Write hand-checkable selection tests**

Use records whose signs differ across candidates and assert:

```python
result = calibrate_reject_cost(
    records,
    candidates=(0.2, 0.3, 0.5),
    criteria=criteria,
)
assert result.selected_reject_cost == 0.3
assert result.candidates["0.2"]["status"] == "fail"
assert result.candidates["0.3"]["status"] == "pass"
```

Add failures for a non-train release, too few groups, action-determined signs, no passing candidate, duplicate sample IDs, and non-finite fields.

- [ ] **Step 2: Implement deterministic calibration types and rules**

Implement:

```python
@dataclass(frozen=True)
class RejectCostCalibrationCriteria:
    minimum_group_count: int
    minimum_positive_fraction: float
    maximum_positive_fraction: float
    minimum_mixed_action_count: int

@dataclass(frozen=True)
class LoadedRejectCostCalibration:
    root: Path
    selected_reject_cost: float
    calibration_digest: str
    source_release_manifest_digests: tuple[str, ...]
```

Candidate reports contain sign counts/fractions, value quantiles, per-action sign counts, mixed-action count, rejection-selection rate, and the check that every positive value has risk reduction greater than action cost.

- [ ] **Step 3: Publish an immutable calibration artifact**

Implement:

```python
publish_reject_cost_calibration(
    output_dir,
    *,
    release_dirs,
    config_path,
    gt_config_path,
) -> LoadedRejectCostCalibration
```

Write `calibration.json`, `verification_gt_frozen.yaml`, `manifest.json`, and `COMPLETE.json` via staging and no-replace rename. The frozen YAML differs from the input GT config only at `decision.reject_cost`. The manifest binds all source release manifest digests and both config SHA-256 values.

- [ ] **Step 4: Implement the CLI**

The CLI accepts repeated `--release-dir`, `--config`, `--gt-config`, and `--output-dir`; it exits non-zero if no candidate passes and never writes a partial success directory.

- [ ] **Step 5: Run calibration tests through Slurm**

Run:

```bash
srun --partition=gpu --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:10:00 \
  .conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_verification_value_calibration.py \
  tests/test_verification_value_calibration_cli.py
```

Expected: all selected tests pass.

### Task 4: Load production releases with calibrated labels

**Files:**
- Create: `src/datasets/verification_release_collection.py`
- Create: `tests/test_verification_release_collection.py`
- Modify: `src/evaluation/verification_metrics.py`
- Modify: `tests/test_verification_metrics.py`

- [ ] **Step 1: Write split, alignment, and binding tests**

Assert that the adapter:

```python
loaded = load_calibrated_verification_release(
    release_dir,
    grid=grid,
    library=library,
    expected_split="train",
    calibration=calibration,
)
assert loaded.samples[0].value_target == pytest.approx(expected_revalued)
assert loaded.calibration_digest == calibration.calibration_digest
```

Reject non-train calibration sources, missing/extra audit records, wrong split, calibration not bound to the training release, and duplicate cross-split groups.

- [ ] **Step 2: Implement the collection adapter**

Implement an immutable result containing samples, loaded shard roots, raw release digest, calibrated input digest, split digest, calibration digest, and reject cost. Apply `dataclasses.replace` to `value_target`, `useful_target`, `br_before`, and `post_risk`; never alter the five model input tensors.

- [ ] **Step 3: Bind calibration into checkpoint manifest v3**

Extend checkpoint construction/validation with:

```python
"value_calibration_digest": calibration_digest,
"reject_cost": reject_cost,
```

Both fields are mandatory together for production release inputs and both `None` for legacy bounded smoke fixtures. Reject v1/v2 manifests as legacy.

- [ ] **Step 4: Run adapter/manifest tests through Slurm**

Run:

```bash
srun --partition=gpu --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:10:00 \
  .conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_verification_release_collection.py \
  tests/test_verification_metrics.py tests/test_verification_dataset.py
```

Expected: all selected tests pass.

### Task 5: Complete production training and held-out evaluation outputs

**Files:**
- Modify: `scripts/09_train_verification_model.py`
- Modify: `scripts/10_evaluate_verification_model.py`
- Modify: `src/models/verification_training.py`
- Modify: `src/evaluation/verification_metrics.py`
- Modify: `src/evaluation/verification_baselines.py`
- Modify: `tests/test_verification_training_smoke.py`
- Modify: `tests/test_verification_evaluation_smoke.py`
- Modify: `tests/test_verification_metrics.py`

- [ ] **Step 1: Write release-input and seed-override tests**

Exercise mutually exclusive input modes:

```python
parser.parse_args(
    [
        "--release-dir",
        "/tmp/verification-release",
        "--value-calibration",
        "/tmp/value-calibration",
    ]
)
```

and assert `--seed 17` is present in the embedded config and checkpoint manifest. Supplying release and shard/handoff inputs together must fail.

- [ ] **Step 2: Add production input options**

Both CLIs accept:

```text
--release-dir PATH --value-calibration PATH
```

Training also accepts `--seed`; evaluation derives the seed from its checkpoint. Existing `--shard-dir/--collection-handoff` remains only for bounded fixtures.

- [ ] **Step 3: Add complete held-out evidence**

Evaluation writes deterministic `predictions.jsonl` rows:

```python
{
    "sample_id": sample.sample_id,
    "ranking_group_id": sample.metadata["ranking_group_id"],
    "action_id": sample.verification_action_id,
    "value_target": sample.value_target,
    "value_prediction": prediction,
    "useful_target": sample.useful_target,
    "useful_probability": probability,
}
```

Metrics include useful-probability bin counts, mean confidence, empirical positive rate, ECE, and Brier score. Learned and baseline reports use action, source mode, blind type, target object type, and footprint kind slices.

- [ ] **Step 4: Authenticate run directories**

Training and evaluation publish a `COMPLETE.json` that binds every output payload SHA-256. Strict reload occurs before atomic publication; an existing output remains immutable.

- [ ] **Step 5: Run focused model-chain tests through Slurm**

Run:

```bash
srun --partition=gpu --ntasks=1 --cpus-per-task=2 --mem=10G --time=00:15:00 \
  .conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_verification_model.py tests/test_verification_losses.py \
  tests/test_verification_training_smoke.py tests/test_verification_evaluation_smoke.py \
  tests/test_verification_metrics.py
```

Expected: all selected tests pass.

### Task 6: Aggregate V0/no-ranking multi-seed experiments

**Files:**
- Create: `src/evaluation/verification_experiment_aggregation.py`
- Create: `scripts/10_aggregate_verification_value.py`
- Create: `tests/test_verification_experiment_aggregation.py`

- [ ] **Step 1: Write aggregation identity tests**

Provide three authenticated evaluation directories and assert mean, population standard deviation, and exact per-seed values. Reject duplicate seeds, mixed splits, different held-out split digests, different calibration digests, or model configs that differ anywhere except `training.seed`.

- [ ] **Step 2: Implement strict aggregation**

Implement:

```python
aggregate_verification_evaluations(
    evaluation_dirs,
    *,
    experiment_id,
    output_dir,
) -> LoadedVerificationAggregate
```

Flatten finite numeric learned/loss/baseline metrics into long rows and retain nested slice summaries. Publish `summary.json`, `metrics_long.csv`, `manifest.json`, and `COMPLETE.json` atomically.

- [ ] **Step 3: Implement the CLI**

Accept repeated `--evaluation-dir`, `--experiment-id`, and `--output-dir`. The CLI is identical for `v0` and `without-ranking`; the authenticated model configs distinguish them.

- [ ] **Step 4: Run aggregation tests through Slurm**

Run:

```bash
srun --partition=gpu --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:10:00 \
  .conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_verification_experiment_aggregation.py
```

Expected: all selected tests pass.

### Task 7: Bind matrix declarations to replay evidence

**Files:**
- Modify: `src/evaluation/closed_loop_replay.py`
- Modify: `src/evaluation/experiment_matrix.py`
- Modify: `tests/test_closed_loop_replay.py`
- Modify: `tests/test_experiment_matrix.py`

- [ ] **Step 1: Write mismatch tests**

Publish a suite bound to `risk_calibration + learned_value`, then declare `risk_only`, `value_without_ranking`, or a different non-runtime sensitivity parameter. Each matrix run must be recorded as failed before evaluation starts.

- [ ] **Step 2: Add replay method binding**

Add a canonical `experiment_binding` to `ReplayEvidence`:

```python
{
    "risk_method": "risk_calibration",
    "value_method": "learned_value",
    "parameters": {"calibrated": True, "target_type": "human"},
}
```

Serialize it inside the authenticated replay manifest. Standalone replay may use `{}`, but matrix execution requires a non-empty binding.

- [ ] **Step 3: Validate matrix semantics**

Compare exact risk/value methods and evidence-affecting parameters. Parameters represented by `runtime_overrides` are validated against the constructed runtime and excluded from suite identity; generation/model parameters such as ablation, controlled test, scenario-bank size, posterior temperature, composition, prior, signature, and verification-cost scale remain mandatory.

- [ ] **Step 4: Run closed-loop/matrix tests through Slurm**

Run:

```bash
srun --partition=gpu --ntasks=1 --cpus-per-task=2 --mem=10G --time=00:15:00 \
  .conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_closed_loop_replay.py tests/test_closed_loop_runtime.py \
  tests/test_experiment_matrix.py tests/test_result_registry.py
```

Expected: all selected tests pass.

### Task 8: Final focused integration and workspace audit

**Files:**
- Modify only files listed above when failures identify an in-scope defect.

- [ ] **Step 1: Run the complete verification-value fixture chain through Slurm**

Run one bounded fixture chain that produces a release, calibration artifact, V0 and no-ranking checkpoints, held-out evaluations, aggregates, replay suite, and matrix report. Keep outputs under `.tmp/agent/outputs/` and do not claim target-scale numerical evidence.

- [ ] **Step 2: Run all directly affected tests through Slurm**

Use one quiet pytest command containing only the verification, closed-loop replay, matrix, and registry test files modified by this plan. Expected: all selected tests pass.

- [ ] **Step 3: Inspect scientific invariants**

Run:

```bash
rg -n 'schema_version: \"3\\.0\\.0\"|schema-3' \
  configs/verify_model*.yaml scripts/09_train_verification_model.py \
  scripts/10_evaluate_verification_model.py src/models/verification_model.py
git diff --check
git status --short
```

Expected: no active Schema-3 match, no whitespace errors, and only task-owned files attributed to this implementation.

- [ ] **Step 4: Clean bounded outputs**

Move disposable fixture outputs from `.tmp/agent/outputs/` into `.trash/` if retained for diagnosis; otherwise leave no temporary experiment artifacts. Do not delete or revert concurrent/user-owned files.
