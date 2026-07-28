# Risk R2 Formal Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a runnable and provenance-bound risk experiment matrix whose formal comparison contains R0, R1, R2, and B1-B4, with R2 defined as cross-attention trajectory risk plus occupancy auxiliary supervision.

**Architecture:** Keep one shared production prediction protocol and one calibration/test cohort for all seven formal methods. R2 consumes only deployment-visible scene and trajectory inputs; its scene-only occupancy decoder is trained from label-only sidecars and is discarded by formal inference. R2-no-aux and R2-concat reuse the same trainer as controlled ablations but do not receive separate formal method IDs.

**Tech Stack:** Python 3.10, PyTorch, YAML/JSON experiment configs, pytest through Slurm.

---

## Frozen Experiment Contract

| Experiment | Model | Occupancy auxiliary | Formal method ID | Purpose |
|---|---|---:|---|---|
| R0 | concat CNN | no | `risk-r0` | minimal direct-risk baseline |
| R1 | temporal trajectory-conditioned concat | no | `risk-r1` | temporal direct-risk baseline |
| R2 | cross-attention trajectory queries | yes | `risk-r2` | main method |
| R2-no-aux | same as R2 | no | none | auxiliary-task ablation |
| R2-concat | concat fusion control | yes | none | attention-fusion ablation |
| B1-B4 | occupancy baselines | n/a | `B1`-`B4` | occupancy-to-risk comparison |

All formal methods use seeds `42`, `43`, and `44`, the same authenticated dataset family, the same calibration/test cohorts, and the same prediction protocol digest.

### Task 1: Make R2 a formal prediction method

**Files:**
- Modify: `tests/test_prediction_tables.py`
- Modify: `tests/test_prediction_producer.py`
- Modify: `src/evaluation/prediction_tables.py`
- Modify: `scripts/09_predict_risk.py`
- Modify: `configs/prediction_protocol_production.json`

- [ ] **Step 1: Write failing method-set tests**

Add assertions equivalent to:

```python
assert UNIFIED_PREDICTION_METHODS == (
    "risk-r0", "risk-r1", "risk-r2", "B1", "B2", "B3", "B4"
)
assert protocol["required_methods"] == list(UNIFIED_PREDICTION_METHODS)
```

Score one batch with `RiskModel(variant="r2", occupancy_aux_enabled=True)` and require a `risk-r2` result. Add a producer test requiring `--risk-r2-training-root` and rejecting an R2 checkpoint with `occupancy_aux_enabled=false`.

- [ ] **Step 2: Verify the tests fail for the missing R2 contract**

Run through Slurm:

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G \
  --time=00:20:00 env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI/.conda-envs/sop4-risk/bin/pytest \
  -q tests/test_prediction_tables.py tests/test_prediction_producer.py
```

Expected: failures report the six-method contract and missing R2 producer field.

- [ ] **Step 3: Implement the seven-method protocol**

Set:

```python
PREDICTION_PROTOCOL_LAYOUT_VERSION = "shared_risk_prediction_protocol_v2"
UNIFIED_PREDICTION_METHODS = (
    "risk-r0", "risk-r1", "risk-r2", "B1", "B2", "B3", "B4"
)
```

Bind `required_methods` into the protocol digest, load the R2 publication in `scripts/09_predict_risk.py`, and pass all three risk models to `score_unified_prediction_batch`. Formal `risk-r2` loading must require `variant == "r2"`, `r2_fusion_mode == "cross_attention"`, and `occupancy_aux_enabled is True`.

- [ ] **Step 4: Verify the method-set tests pass**

Run the same two test files through Slurm. Expected: all pass.

### Task 2: Correct Long40 trajectory-query semantics

**Files:**
- Modify: `tests/test_r2_trajectory_query_model.py`
- Modify: `src/models/risk_model.py`

- [ ] **Step 1: Write a failing late-horizon query test**

Construct legal swept cells with TTA values `0.2`, `3.2`, and `6.4` seconds. Call `TrajectoryQueryTransformer._trajectory_queries` and assert that early, middle, and final time bins are valid. The existing implementation must fail because it bins raw seconds only inside `[0,1]`.

- [ ] **Step 2: Verify the late-horizon test fails**

Run through Slurm:

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G \
  --time=00:20:00 env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI/.conda-envs/sop4-risk/bin/pytest \
  -q tests/test_r2_trajectory_query_model.py -k late_horizon
```

Expected: the middle/final query bins are invalid.

- [ ] **Step 3: Normalize TTA with the frozen Long40 horizon**

Import `LONG40_FUTURE_HORIZON_S` from `src.contracts` and compute normalized arrival time before binning:

```python
arrival = arrival_seconds / LONG40_FUTURE_HORIZON_S
```

Keep off-swept cells masked and preserve the existing empty-query fallback.

- [ ] **Step 4: Verify all R2 model tests pass**

Run `tests/test_r2_trajectory_query_model.py` through Slurm. Expected: all pass.

### Task 3: Harden formal evidence and manifests

**Files:**
- Modify: `tests/test_risk_production_training.py`
- Modify: `tests/test_prediction_producer.py`
- Modify: `tests/test_production_eval_cli.py`
- Modify: `src/training/risk_trainer.py`
- Modify: `scripts/09_predict_risk.py`
- Modify: `scripts/07_calibrate_risk.py`
- Modify: `scripts/10_eval_offline.py`
- Modify: `src/datasets/sidecar_writer.py`
- Remove from this merge: `scripts/04_publish_risk_evaluation_records.py`

- [ ] **Step 1: Write failing integrity tests**

Require `_training_data_scale(formal_50k, 49_999)` to return `("fixture_standin", False)` and exactly `50_000` to return `("formal_50k", True)`. Tamper or swap `best_checkpoint.pt` in a signed training publication and require the prediction loader to fail. Assert calibration/evaluation manifests use `SCHEMA_VERSION` and Long40 validation errors say `6.4 s`.

- [ ] **Step 2: Verify the integrity tests fail**

Run the affected test nodes through Slurm. Expected failures must identify the over-permissive scale flag, partial publication loader, Schema 3 stamp, and stale horizon message.

- [ ] **Step 3: Implement the integrity fixes**

Expose a public, snapshot-based production training publication loader from `src/training/risk_trainer.py` and reuse it in the prediction producer rather than reimplementing partial checks. Return scientific eligibility only for exactly 50,000 selected samples. Stamp calibration/evaluation manifests with `SCHEMA_VERSION`. Remove the replay CLI that defaults to a retired producer; authenticated evaluation-record loading remains available through `src/datasets/risk_evaluation_store.py`.

- [ ] **Step 4: Verify all integrity tests pass**

Run the affected test files through Slurm. Expected: all pass.

### Task 4: Publish runnable experiment configurations

**Files:**
- Create: `configs/risk_model_r0_production.yaml`
- Create: `configs/risk_model_r1_production.yaml`
- Modify: `configs/risk_model_r2_production.yaml`
- Create: `configs/risk_model_r2_no_aux_production.yaml`
- Modify: `configs/risk_model_r2_concat_control_production.yaml`
- Remove: `configs/risk_model_r2_aux_production.yaml`
- Create: `configs/risk_experiment_matrix.yaml`
- Modify: `tests/test_r2_trajectory_query_model.py`
- Modify: `docs/event_centered_blind_spot_agent_sops.md`
- Modify: `docs/event_centered_blind_spot_implementation_spec.md`

- [ ] **Step 1: Write failing configuration-contract tests**

Load every experiment config and assert the frozen table above, including three seeds and R2 auxiliary settings. Require `risk_model_r2_production.yaml` to be the main auxiliary configuration and reject ambiguous R3 naming.

- [ ] **Step 2: Verify the configuration tests fail**

Run the R2 configuration test through Slurm. Expected: missing R0/R1/no-aux/matrix files and wrong main R2 auxiliary flag.

- [ ] **Step 3: Add the experiment configs and align method documentation**

Use the existing trainer fields without introducing a new variant. Set R2 main and concat control to `occupancy_aux_enabled: true`; set no-aux to false. Record seeds, stages, formal method IDs, roles, and config paths in `risk_experiment_matrix.yaml`. Update the two active method documents so R2, not R3, names the auxiliary-head main method.

- [ ] **Step 4: Verify the configuration tests pass**

Run the R2 configuration tests through Slurm. Expected: all pass.

### Task 5: Final verification and publication

**Files:**
- Verify all changed files from Tasks 1-4.

- [ ] **Step 1: Run targeted tests through Slurm**

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G \
  --time=00:30:00 env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI/.conda-envs/sop4-risk/bin/pytest \
  -q tests/test_prediction_tables.py tests/test_prediction_producer.py \
  tests/test_r2_trajectory_query_model.py tests/test_risk_production_training.py \
  tests/test_production_eval_cli.py
```

Expected: all pass.

- [ ] **Step 2: Run the complete suite through Slurm**

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G \
  --time=00:30:00 env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI/.conda-envs/sop4-risk/bin/pytest -q
```

Expected: zero failures; the known environment-dependent skip may remain.

- [ ] **Step 3: Review and push**

Check `git diff --check`, perform an independent review of the latest remote-main range, fetch `main` again, and push without force only when the remote head is still the tested base.
