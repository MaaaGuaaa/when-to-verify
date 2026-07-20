# SOP11–14 Verification-Value Implementation Plan

> **For Codex:** Execute this plan task by task with test-driven development.
> Do not delegate, edit frozen contract files, or rerun the already accepted
> SOP05–07 full audit.

**Goal:** Deliver the SOP11–14 verification-value path with hand-checkable toy
truth, a 10–100-event real-train smoke joined to audited SOP03/04/05/07
evidence, deterministic schema-3 shards, and a CPU-testable PyTorch V0 model.

**Architecture:** Keep deployment-visible tensors and oracle label state in
separate types and call boundaries. SOP11 generates action geometry,
observations, signatures, and replans. SOP12 evaluates a validated scenario
bank through an injected typed-footprint loss. SOP13 serializes only frozen
`VerificationSample` fields plus non-input audit metadata. SOP14 consumes the
legal tensors through a concat CNN and reports group-local ranking metrics.

**Tech stack:** Python 3.10, NumPy, PyYAML, existing geometry/planning and
SOP03–07 loaders, optional PyTorch 2.x, pytest, compressed NPZ + JSONL.

**Fixed commands:**

- Git: `/home/home/ccnt_zq/zq_zhouyiqun/.local/git/bin/git`
- Python: `/home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI/.conda-envs/sop4-risk/bin/python`
- Worktree: `/home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI-worktrees/sop-11-14-verification-toy`

---

### Task 1: Package discovery and optional PyTorch boundary

**Files:**

- Modify: `pyproject.toml`
- Create: `tests/test_verification_packaging.py`

**Steps:**

1. Write a failing test that parses `pyproject.toml`, requires optional group
   `verification = ["torch>=2.0,<3"]`, and requires setuptools discovery to
   include every `src.*` package.
2. Run `python -m pytest -q tests/test_verification_packaging.py`; confirm the
   assertion fails for the missing group/discovery.
3. Replace the explicit two-package list with `[tool.setuptools.packages.find]`
   and `include = ["src*"]`; add only the approved optional dependency.
4. Re-run the test and import `torch`, recording its version and CUDA status.
5. Commit only the two files.

### Task 2: Canonical six-action library and analytic motion

**Files:**

- Create: `src/planning/verification_actions.py`
- Create: `configs/verification_actions.yaml`
- Create: `tests/test_verification_actions.py`

**Steps:**

1. Write failing tests for exact action order/vectors, ±10°/±20° endpoints,
   forward peek, stop scan, no yaw-only lateral motion, finite config values,
   deterministic IDs, and static/dynamic feasibility for circle and rotated
   rectangle footprints.
2. Run the new test and confirm import/behavior failures.
3. Implement immutable `VerificationAction`, strict YAML loading, analytic
   post-pose/trace generation, action cost, and conservative feasibility.
4. Re-run until green; include a mutation/invalid-config rejection test.
5. Commit the config, module, and test.

### Task 3: Counterfactual geometry, oracle observation, and signature

**Files:**

- Create: `src/generation/counterfactual_verify.py`
- Create: `tests/fixtures/verification_world.py`
- Create: `tests/test_counterfactual_verify.py`

**Steps:**

1. Add an independent mixed circle/rectangle toy world with hand-authored
   expected visibility outcomes for one critical and one irrelevant action.
2. Write failing tests proving the expected-FOV mask is unchanged when hidden
   oracle occupancy changes, while the label-side observation changes.
3. Add failing tests for multi-occluder ray casting, typed footprints, the
   seven-feature signature, no oracle metadata in the signature, and refusal
   to fit normalization statistics outside train.
4. Implement separate static-only FOV and oracle-label observation functions,
   an immutable observable result, seven-feature extraction, and a train-only
   normalizer.
5. Run the targeted tests, then commit.

### Task 4: Post-action replanning from the new pose

**Files:**

- Create: `src/planning/replanning.py`
- Create: `tests/test_replanning.py`

**Steps:**

1. Write failing tests requiring every implicit rollout seed to equal the
   post-action pose, the original nominal endpoint/direction to be the task
   anchor, the same configured differential-drive sampler, static filtering,
   stop/reject retention, and absence of the nominal suffix.
2. Implement anchored local-frame resampling and query-map regeneration using
   existing SOP04 modules; do not alter SOP04 code.
3. Test straight/arc/stop cases, determinism, shape/dtype/finite values, and
   static-collision rejection.
4. Commit module and tests.

### Task 5: Validated M=8/16/32 scenario banks

**Files:**

- Create: `src/generation/scenario_bank.py`
- Create: `configs/verification_gt.yaml`
- Create: `tests/test_scenario_bank.py`

**Steps:**

1. Write failing tests for presets and the M=16 composition
   `1 current + 2 empty + 5 temporal + 4 spatial + 2 speed + 2 irrelevant`.
2. Add failures for visible-occupancy mismatch, static-map mutation,
   trajectory/spec key mismatch, non-target removal/retyping, split/source
   namespace reuse, NaN/Inf, and unsupported M.
3. Implement immutable hypotheses/banks, deterministic transforms, strict
   composition validation, and semantic digesting. Circle and rectangle yaw
   semantics must remain typed.
4. Test deterministic digests and supported temperatures `0.1/0.2/0.5`.
5. Commit module, config, and tests.

### Task 6: Exact and soft observation posterior

**Files:**

- Create: `src/generation/observation_posterior.py`
- Create: `tests/test_observation_posterior.py`

**Steps:**

1. Write failing hand-grouped exact-posterior tests and soft-posterior tests
   whose rows are finite, nonnegative, and sum to one.
2. Test signature dimensionality, train-normalizer provenance, exact mismatch
   rejection, temperature sensitivity, and deterministic ties.
3. Implement exact discrete grouping and stabilized softmax over normalized
   signature distance with default `tau_o=0.2`.
4. Re-run tests and commit.

### Task 7: Net verification value and scientific invariants

**Files:**

- Create: `src/generation/verification_gt.py`
- Create: `tests/test_verification_gt.py`

**Steps:**

1. Write failing tests with hand-enumerated trajectory/world losses for
   `br_before=min(mean execute loss, reject)`, posterior-weighted best replan,
   reject fallback, action cost exactly once, `G*=br_before-post_risk`, and
   `useful=int(G*>0)`.
2. Add mixed circle/rectangle typed-loss tests, action-cost monotonicity,
   critical-versus-irrelevant action ordering, empty-bank behavior, and proof
   that post-action replans—not nominal suffixes—are scored.
3. Implement an injected finite loss protocol plus immutable audit result; use
   the merged typed geometry/risk functions for the real adapter and the hand
   table for toy tests.
4. Run SOP11–12 tests together and commit.

### Task 8: Schema-3 verification sample construction

**Files:**

- Create: `src/datasets/verification_dataset.py`
- Create: `tests/test_verification_dataset.py`
- Create: `tests/test_verification_input_isolation.py`

**Steps:**

1. Write failing tests that build all six actions per `(state, nominal)` group,
   validate through frozen `validate_verification_sample`, and retain group
   ranking identities only in metadata.
2. Add recursive leakage tests: model inputs may contain only BEV history,
   state channels, trajectory channels, static expected-FOV geometry, and the
   action vector. Oracle/post-observation/world identities stay label/audit
   side only.
3. Implement sample construction by copying deployment-side arrays and
   separating labels/audit. Enforce action ID/vector agreement, split/group
   isolation, float32, finite values, and deterministic IDs without `hash()`.
4. Run tests and commit.

### Task 9: Deterministic NPZ+JSONL shards and loader

**Files:**

- Create: `src/datasets/verification_dataloader.py`
- Extend: `src/datasets/verification_dataset.py`
- Extend: `tests/test_verification_dataset.py`

**Steps:**

1. Write failing round-trip tests for compressed numeric NPZ, canonical JSONL,
   no object arrays/pickle, staging + atomic rename, overwrite rejection,
   checksums, manifest/semantic digests, and deterministic index/batch order.
2. Implement strict writer/loader/auditor with exact root file set and sampled
   `G*` recomputation callback.
3. Test corrupt dtype/shape/NaN/checksum/metadata, cross-split groups, duplicate
   IDs, and action imbalance reporting.
4. Commit modules and tests.

### Task 10: Audited SOP03/04/05/07 real-train source adapter

**Files:**

- Create: `src/datasets/verification_sources.py`
- Create: `tests/test_verification_sources.py`

**Steps:**

1. Write failing miniature-handoff tests for schema/code/digest/split/count
   mismatch, unsafe paths, missing trust anchors, train-only status, and
   deterministic shard/event selection.
2. Implement strict handoff parsers. Resolve per-shard SOP05 publication
   digests from the external batch handoff, call `load_complete_sop05_events`,
   join SOP03 pairs and SOP04 trajectories through formal loaders, and bind the
   SOP07 collection digest without loading its numeric shards.
3. Regenerate required paired/scenario inputs deterministically from the
   restored mother event and trusted SOP03 snippet library. Never interpret
   temporal near-miss samples as ordinary safe negatives.
4. Run unit tests and a one-event read-only adapter smoke; commit.

### Task 11: Toy and real-train generation CLI

**Files:**

- Create: `scripts/08_generate_verification_dataset.py`
- Extend: `tests/test_verification_dataset.py`
- Extend: `tests/test_verification_sources.py`

**Steps:**

1. Write failing CLI tests for explicit `toy|sop05-train` mode, required path
   arguments, no silent fallback, bounded 10–100 sample count, immutable output,
   and scientific status `toy_smoke_only|train_smoke_only`.
2. Implement orchestration only; reusable logic remains in `src/` modules.
3. Generate toy data twice (separate output dirs) and compare semantic digests.
4. Run a real 10-event / 60-sample smoke from one SOP05 shard, audit all
   shapes/dtypes/finite values and provenance, and record runtime/resources.
5. Commit script and tests. Keep outputs untracked.

### Task 12: PyTorch V0 model and group-local losses

**Files:**

- Create: `src/models/__init__.py`
- Create: `src/models/verification_model.py`
- Create: `configs/verify_model.yaml`
- Create: `tests/test_verification_model.py`
- Create: `tests/test_verification_losses.py`

**Steps:**

1. Write failing output-shape/API tests that accept exactly legal input tensors
   and reject extra oracle/post-observation inputs.
2. Add hand-calculated Huber, BCE, and pairwise ranking tests; cross-group pairs
   must be rejected/ignored, and correct within-group order must reduce loss.
3. Implement a small concat CNN, action MLP, value/useful heads, deterministic
   initialization, and weighted composite loss.
4. Test finite forward/loss/gradients on CPU and commit.

### Task 13: Baselines, metrics, and checkpoint provenance

**Files:**

- Create: `src/evaluation/__init__.py`
- Create: `src/evaluation/verification_baselines.py`
- Create: `src/evaluation/verification_metrics.py`
- Create: `tests/test_verification_metrics.py`

**Steps:**

1. Write hand-calculated tests for visible-area, swept-coverage, and occupancy-
   entropy baselines plus F1, MSE/Huber, Spearman, Kendall, pairwise accuracy,
   top-1 regret, and action/type slices.
2. Implement deterministic tie policies and group-local aggregation.
3. Add checkpoint-manifest validation helpers binding schema, channel order,
   action order, input digest, model config, seed, and code version; reject
   legacy/mismatched manifests.
4. Run tests and commit.

### Task 14: Training smoke and complete verification

**Files:**

- Create: `scripts/09_train_verification_model.py`
- Create: `tests/test_verification_training_smoke.py`

**Steps:**

1. Write a failing deterministic toy-overfit test with decreasing loss and
   correct top action on a tiny fixed dataset.
2. Implement CPU training orchestration, deterministic seeds/order, checkpoint
   + manifest atomic writes, resume validation, and metric JSON output.
3. Run the targeted model suite and toy overfit smoke. Do not claim paper F1 or
   ranking thresholds from toy/train-only data.
4. Run central contracts plus every new test, then the toy and real 10–100
   sample smoke/audit. Check shape, dtype, NaN/Inf, determinism, time indexing,
   cost-once, oracle isolation, and split status.
5. Run `git diff --check`, `git status --short`, stage only exact owned files,
   and create one final local implementation commit.

### Final validation commands

```bash
/home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI/.conda-envs/sop4-risk/bin/python \
  scripts/00_validate_contracts.py --config configs/base.yaml

/home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI/.conda-envs/sop4-risk/bin/python \
  -m pytest -q tests/test_contracts.py tests/test_toy_fixture.py \
  tests/test_trajectory_rollout.py tests/test_verification_packaging.py \
  tests/test_verification_actions.py tests/test_counterfactual_verify.py \
  tests/test_replanning.py tests/test_scenario_bank.py \
  tests/test_observation_posterior.py tests/test_verification_gt.py \
  tests/test_verification_dataset.py tests/test_verification_input_isolation.py \
  tests/test_verification_sources.py tests/test_verification_model.py \
  tests/test_verification_losses.py tests/test_verification_metrics.py \
  tests/test_verification_training_smoke.py
```

For the real smoke and model smoke, run the same Python executable inside a
CPU-only Slurm allocation (`-p gpu --cpus-per-task=2`, no GPU request), save
outputs under `outputs/`, and do not stage them.
