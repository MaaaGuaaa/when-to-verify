# Verification Value Relative Task-Cost Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace absolute TEB task cost with scale-invariant task regret so real verification values naturally contain positive and negative examples.

**Architecture:** Keep the existing scenario bank, posterior, risk loss, action cost, and sign rule. Convert policy task cost to regret relative to the nominal trajectory, retain unclipped post-policy losses for reject-cost calibration, and calibrate only on eligible training scenes.

**Tech Stack:** Python 3.10, NumPy, pytest, YAML, Slurm.

---

## File Map

- Modify `src/generation/verification_gt.py`: relative task-regret helper,
  decision-loss construction, result audit field, GT version.
- Modify `tests/test_verification_gt.py`: hand-computed positive/negative cases,
  scale invariance, zero-baseline rejection, unclipped-loss checks.
- Modify `tests/test_verification_dataset.py`: construct the versioned result with
  the new label-side audit field.
- Create `src/evaluation/verification_value_calibration.py`: pure reject-cost
  revaluation from stored losses.
- Create `tests/test_verification_value_calibration.py`: exact calibration
  arithmetic and invalid-input tests.
- Temporarily restore the bounded five-scene audit scripts under
  `.tmp/agent/scripts/`; move them back to `.trash/` after use.
- Conditionally modify `configs/verification_gt.yaml`, `configs/base.yaml`, and
  `src/utils/config.py` only if train calibration selects `0.3` or `0.5`.

### Task 1: Relative Task-Regret Primitive

**Files:**
- Modify: `tests/test_verification_gt.py`
- Modify: `src/generation/verification_gt.py`

- [ ] **Step 1: Add failing helper tests**

Add the import and test:

```python
from src.generation.verification_gt import relative_task_regret


def test_relative_task_regret_is_scale_invariant_and_nonnegative():
    assert relative_task_regret(8.0, nominal_task_cost=8.0) == 0.0
    assert relative_task_regret(6.0, nominal_task_cost=8.0) == 0.0
    assert relative_task_regret(10.0, nominal_task_cost=8.0) == pytest.approx(0.25)
    assert relative_task_regret(1000.0, nominal_task_cost=800.0) == pytest.approx(0.25)


def test_relative_task_regret_rejects_zero_nominal_cost():
    with pytest.raises(ValueError, match="nominal_task_cost must be positive"):
        relative_task_regret(1.0, nominal_task_cost=0.0)
```

- [ ] **Step 2: Run the tests through Slurm and confirm failure**

Run:

```bash
srun -p gpu -N1 -n1 -c1 --mem=2G -t 00:03:00 \
  .conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_verification_gt.py \
  -k 'relative_task_regret'
```

Expected: collection error because `relative_task_regret` is not defined.

- [ ] **Step 3: Implement the helper**

Add beside `_trajectory_task_cost`:

```python
def relative_task_regret(
    policy_task_cost: Any,
    *,
    nominal_task_cost: Any,
) -> float:
    policy = _finite_nonnegative(policy_task_cost, name="policy_task_cost")
    nominal = _finite_nonnegative(
        nominal_task_cost,
        name="nominal_task_cost",
    )
    if nominal <= 0.0:
        raise ValueError("nominal_task_cost must be positive")
    return max(0.0, policy / nominal - 1.0)
```

Export it from `__all__`.

- [ ] **Step 4: Run the focused helper tests**

Use the Step 2 command. Expected: `2 passed`.

### Task 2: Relative Decision Loss and Unclipped Audit

**Files:**
- Modify: `tests/test_verification_gt.py`
- Modify: `src/generation/verification_gt.py`

- [ ] **Step 1: Update the hand-computed exact-posterior test**

For the existing `_HandRisk` fixture, assert the new arithmetic:

```python
assert result.mean_execute_loss == pytest.approx(0.50)
assert result.br_before == pytest.approx(0.50)
np.testing.assert_allclose(
    result.post_decision_risks,
    np.asarray(
        [
            0.60
            if item.variant_kind in {"current", "temporal", "speed"}
            else 0.0
            for item in bank.hypotheses
        ],
    ),
    atol=1e-12,
)
assert result.mean_post_decision_risk_before_action_cost == pytest.approx(0.30)
assert result.value_target == pytest.approx(0.1005)
assert result.useful_target == 1
assert result.unclipped_best_policy_losses == tuple(
    1.0 if item.variant_kind in {"current", "temporal", "speed"} else 0.0
    for item in bank.hypotheses
)
```

Add a negative no-information case:

```python
def test_uninformative_observation_stays_negative_after_relative_task_cost():
    toy, bank = _bank()
    shape = (toy.grid.height, toy.grid.width)
    observations = tuple(_observation(False, shape) for _ in bank.hypotheses)
    _, _, _, _, result, _ = _evaluate(observations)

    assert result.br_before == pytest.approx(0.50)
    np.testing.assert_allclose(result.post_decision_risks, 0.50, atol=1e-12)
    assert result.value_target == pytest.approx(-result.action_cost)
    assert result.useful_target == 0
```

For every test that creates a nominal trajectory with task cost `0.0`, either
change it to `0.05` when testing unrelated behavior or assert the new explicit
zero-baseline error.

- [ ] **Step 2: Run the focused tests and confirm arithmetic failures**

Run:

```bash
srun -p gpu -N1 -n1 -c2 --mem=4G -t 00:05:00 \
  .conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_verification_gt.py \
  -k 'exact_g_star or uninformative_observation or zero_nominal'
```

Expected: failures showing the old absolute task cost and missing audit field.

- [ ] **Step 3: Change the evaluator without changing the value identity**

Set `VERIFICATION_GT_VERSION = "verification_value_gt_v5"`.

Add the required result field before defaulted fields:

```python
unclipped_best_policy_losses: tuple[float | None, ...]
```

Validate that it is a tuple of length `bank_size` and every item is either
`None` or finite and non-negative.

In `evaluate_verification_value`, replace nominal absolute task loss with:

```python
nominal_task = _trajectory_task_cost(nominal_trajectory)
if nominal_task <= 0.0:
    raise ValueError("nominal trajectory task_cost must be positive")

nominal_losses[world_index] = weight * risk
```

Build each policy loss with:

```python
task_regret = relative_task_regret(
    _trajectory_task_cost(policy),
    nominal_task_cost=nominal_task,
)
policy_world_losses[policy_index, world_index] = (
    task_regret
    + weight * _risk_value(
        risk_loss,
        policy,
        policy.poses,
        hypothesis,
    )
)
```

Track the best policy before rejection:

```python
best_policy_loss: float | None = None
best_policy_id: str | None = None
for policy_index, policy in enumerate(policies):
    expected = float(np.dot(posterior_row, policy_world_losses[policy_index]))
    if best_policy_loss is None or expected < best_policy_loss:
        best_policy_loss = expected
        best_policy_id = policy.trajectory_id

unclipped_best_policy_losses.append(best_policy_loss)
if best_policy_loss is not None and best_policy_loss < reject:
    best_loss = best_policy_loss
    best_id = str(best_policy_id)
else:
    best_loss = reject
    best_id = "reject"
```

Pass `tuple(unclipped_best_policy_losses)` into `VerificationValueResult`.

- [ ] **Step 4: Run all verification-GT tests**

Run:

```bash
srun -p gpu -N1 -n1 -c2 --mem=6G -t 00:08:00 \
  .conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_verification_gt.py
```

Expected: all tests pass.

### Task 3: Dataset Compatibility

**Files:**
- Modify: `tests/test_verification_dataset.py`

- [ ] **Step 1: Update the explicit result fixture**

Add:

```python
unclipped_best_policy_losses=(0.20,),
```

to `_value`.

- [ ] **Step 2: Run focused dataset and pipeline tests**

Run:

```bash
srun -p gpu -N1 -n1 -c2 --mem=8G -t 00:10:00 \
  .conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_verification_dataset.py \
  tests/test_verification_pipeline.py
```

Expected: all tests pass and existing model-input fields remain unchanged.

### Task 4: Reject-Cost Revaluation

**Files:**
- Create: `src/evaluation/verification_value_calibration.py`
- Create: `tests/test_verification_value_calibration.py`

- [ ] **Step 1: Write failing calibration tests**

```python
from src.evaluation.verification_value_calibration import revalue_reject_cost


def test_revalue_reject_cost_uses_unclipped_policy_losses():
    row = revalue_reject_cost(
        nominal_execute_losses=np.asarray([0.9, 0.1], dtype=np.float64),
        unclipped_best_policy_losses=(0.1, 0.8),
        action_cost=0.05,
        reject_cost=0.3,
    )
    assert row.br_before == pytest.approx(0.3)
    assert row.mean_post_before_action_cost == pytest.approx(0.2)
    assert row.value_target == pytest.approx(0.05)
    assert row.useful_target == 1


def test_revalue_reject_cost_treats_missing_policy_as_reject():
    row = revalue_reject_cost(
        nominal_execute_losses=np.asarray([0.9, 0.1], dtype=np.float64),
        unclipped_best_policy_losses=(None, 0.8),
        action_cost=0.05,
        reject_cost=0.3,
    )
    assert row.mean_post_before_action_cost == pytest.approx(0.3)
    assert row.value_target == pytest.approx(-0.05)
    assert row.useful_target == 0
```

- [ ] **Step 2: Run the tests and confirm import failure**

Run:

```bash
srun -p gpu -N1 -n1 -c1 --mem=2G -t 00:03:00 \
  .conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_verification_value_calibration.py
```

Expected: module import failure.

- [ ] **Step 3: Implement the pure revaluation module**

Create an immutable `RejectCostRevalue` with `br_before`,
`mean_post_before_action_cost`, `post_risk`, `value_target`, and
`useful_target`. Implement `revalue_reject_cost` by validating inputs and
computing:

```python
mean_execute = float(np.mean(nominal_execute_losses, dtype=np.float64))
br_before = min(mean_execute, reject_cost)
post = np.asarray(
    [
        reject_cost if value is None else min(float(value), reject_cost)
        for value in unclipped_best_policy_losses
    ],
    dtype=np.float64,
)
mean_post = float(np.mean(post, dtype=np.float64))
post_risk = mean_post + action_cost
value_target = br_before - post_risk
```

- [ ] **Step 4: Run the calibration tests**

Use the Step 2 command. Expected: `2 passed`.

### Task 5: Real-Scene Smoke and Train Calibration

**Files:**
- Temporarily restore:
  `.tmp/agent/scripts/20260728-2316-sop12-five-scene-audit.py`
- Temporarily restore:
  `.tmp/agent/scripts/20260728-2326-sop12-five-revealing-action-audit.py`
- Create output:
  `reports/sop12_value_relative_task_smoke5_20260729_v1/results.json`
- Create output:
  `reports/sop12_value_relative_task_train100_20260729_v1/results.json`
- Conditionally modify configs listed in the File Map.

- [ ] **Step 1: Restore the bounded scripts and add audit serialization**

Move the two source scripts from
`.trash/20260729-0001-sop12-five-scene-audit/` back to
`.tmp/agent/scripts/`. In `_attach_values`, serialize:

```python
"unclipped_best_policy_losses": [
    None if value is None else float(value)
    for value in result.unclipped_best_policy_losses
],
```

Use `revalue_reject_cost` to report values for reject costs
`0.2`, `0.3`, and `0.5` without rerunning simulation.
Aggregate each candidate without filtering:

```python
candidate_summary = {}
for reject_cost in (0.2, 0.3, 0.5):
    rows = [
        revalue_reject_cost(
            nominal_execute_losses=result.nominal_execute_losses,
            unclipped_best_policy_losses=result.unclipped_best_policy_losses,
            action_cost=result.action_cost,
            reject_cost=reject_cost,
        )
        for result in all_action_results
    ]
    candidate_summary[str(reject_cost)] = {
        "positive_count": sum(row.useful_target for row in rows),
        "negative_count": len(rows) - sum(row.useful_target for row in rows),
        "per_action_positive_counts": {
            action_id: sum(
                row.useful_target
                for row, observed_action_id in zip(
                    rows,
                    all_action_ids,
                    strict=True,
                )
                if observed_action_id == action_id
            )
            for action_id in canonical_action_ids
        },
    }
```

- [ ] **Step 2: Run the five-scene Slurm smoke**

Use the same complete-mother and final-scenario roots as the prior audit,
`M=8`, exact posterior, four candidates, and seed `42`. Write to the smoke
output above. Expected: at least one positive and one negative action across
the 30 actions; otherwise stop and report the replanner failure branch from
the approved design.

- [ ] **Step 3: Run 100 eligible training groups**

Run the same bounded selector with `--scene-count 100`,
`--max-source-attempts 3000`, and a new immutable output directory. Use Slurm;
do not fall back to the login node. The report must include per-candidate
reject-cost sign counts and per-action sign counts.

- [ ] **Step 4: Freeze the smallest passing reject cost**

Select the smallest of `0.2`, `0.3`, `0.5` whose positive fraction is in
`[0.10, 0.70]`. If `0.2` passes, leave configs unchanged. If `0.3` or `0.5`
passes, update that exact scalar in `configs/verification_gt.yaml`,
`configs/base.yaml`, and `src/utils/config.py`, then update exact-value tests.
If none passes, do not alter action costs or labels; report that replanning
requires a separate design.

- [ ] **Step 5: Run robustness checks**

On a bounded subset, rerun with `M=16` and seeds `43` and `44`. Expected: every
run contains both signs and preserves the value identity within `1e-12`.

- [ ] **Step 6: Clean temporary files**

Move the restored scripts and generated bytecode back into a new
`.trash/20260729-sop12-relative-task-calibration/` directory. Confirm no
staging directory remains.

### Task 6: Final Focused Validation

**Files:**
- Verify all files changed by Tasks 1-5.

- [ ] **Step 1: Run focused tests through Slurm**

```bash
srun -p gpu -N1 -n1 -c2 --mem=8G -t 00:12:00 \
  .conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_verification_gt.py \
  tests/test_verification_dataset.py \
  tests/test_verification_pipeline.py \
  tests/test_verification_value_calibration.py
```

Expected: all tests pass.

- [ ] **Step 2: Check provenance and workspace state**

Run `git diff --check`, inspect `git status --short`, list every task-owned
file, and verify report JSON digests. Do not stage or commit pre-existing
changes in files already dirty before this task.
