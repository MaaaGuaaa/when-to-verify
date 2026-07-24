# SOP05 History-Visibility Strata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate auditable `seen_then_occluded` collision mothers alongside existing `unseen_in_history_window` mothers without running production-scale generation.

**Architecture:** Add one pure history-visibility contract module, classify only after existing exact physics and visibility validation, keep deterministic per-stratum candidate prefixes, and enforce the configured composition again at global selection/publication. Preserve schema 3 tensors while advancing SOP05 semantic version tokens.

**Tech Stack:** Python 3.10, NumPy, PyYAML, pytest, existing stable-digest and atomic-publication helpers.

---

### Task 1: Freeze The History-Visibility Contract

**Files:**
- Create: `src/generation/history_visibility.py`
- Modify: `configs/generator_train.yaml`
- Modify: `configs/generator_test.yaml`
- Test: `tests/test_history_visibility.py`

- [ ] **Step 1: Write classification and allocation tests**

Add tests for all-false unseen history, a visible prefix followed by two hidden
frames, only one trailing hidden frame, malformed shape/dtype, normalized
`80/20` policy, exact ten-event `8/2` allocation, deterministic repeated
allocation, and both strata appearing across single-event pair seeds.

- [ ] **Step 2: Verify the tests fail for the missing module**

Run:

```bash
.conda-envs/sop4-risk/bin/python -m pytest -q tests/test_history_visibility.py
```

Expected: collection fails because `src.generation.history_visibility` does
not exist.

- [ ] **Step 3: Implement the pure contract module**

Define exact constants and immutable values:

```python
HISTORY_VISIBILITY_POLICY_VERSION = "target_history_visibility_policy_v1"
SEEN_THEN_OCCLUDED = "seen_then_occluded"
UNSEEN_IN_HISTORY_WINDOW = "unseen_in_history_window"
HISTORY_VISIBILITY_REGIMES = (
    SEEN_THEN_OCCLUDED,
    UNSEEN_IN_HISTORY_WINDOW,
)
```

Implement `HistoryVisibilityPolicy`, `HistoryVisibilityAssessment`,
`normalize_history_visibility_policy()`, `classify_history_visibility()`, and
`allocate_history_visibility_counts()`. Reject non-boolean or non-eight-frame
vectors rather than coercing them.

- [ ] **Step 4: Add the strict config block**

Add the design's `target_history_visibility` mapping to both production
generator configs and include its normalized canonical form in the generator
semantic digest.

- [ ] **Step 5: Run the focused contract tests**

Run the command from Step 2. Expected: all tests pass.

### Task 2: Enforce Pair-Local Stratum Requests

**Files:**
- Modify: `src/generation/event_sampler.py`
- Test: `tests/test_dynamic_object_transplant.py`

- [ ] **Step 1: Write failing pair-generation tests**

Add one controlled exact-candidate test where the visibility histories arrive
in this order:

```python
[
    [False] * 8,
    [True, True, True, True, False, False, False, False],
]
```

Request only `seen_then_occluded` and assert that the first candidate cannot
satisfy the quota. Add a real fixture test that calls `generate_events()` and
asserts the accepted event has a true historical frame, two trailing false
frames, current hidden, a future collision, and future continuous emergence.

- [ ] **Step 2: Run the two new tests and confirm RED**

Run only their pytest node IDs. Expected: failure because pair generation
still accepts the first exact-valid candidate regardless of history regime.

- [ ] **Step 3: Bucket exact-valid candidates by regime**

Normalize the policy once, allocate pair-local requested counts from
`event_count` and the pair seed, classify `target_visibility_history`, and keep
one `_BoundedAcceptedCandidates` instance per regime. Stop only when every
bucket reaches its requested count.

- [ ] **Step 4: Publish pair-local evidence**

Add canonical policy, requested counts, exact-valid counts, accepted counts,
and deficits to `EventGenerationReport.summary`. Add the regime, last-visible
index, trailing-hidden count, and policy version to accepted world metadata.

- [ ] **Step 5: Run event-sampler tests**

Run:

```bash
.conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_history_visibility.py tests/test_dynamic_object_transplant.py
```

Expected: all tests pass without reading real datasets.

### Task 3: Enforce Run-Level Composition

**Files:**
- Modify: `src/generation/sop05_selection.py`
- Modify: `src/generation/sop05_run.py`
- Modify: `src/generation/sop05_output_loader.py`
- Test: `tests/test_sop05_selection.py`
- Test: `tests/test_sop05_run.py`
- Test: `tests/test_sop05_output_loader.py`

- [ ] **Step 1: Write failing selection and publication tests**

Extend `Sop05SelectionCandidate` with `history_visibility_regime`. Test exact
`80/20` global selection, no cross-stratum backfill, explicit deficits,
worker-order independence, run `quota_unmet` on a missing seen stratum, and
loader rejection after tampering with regime metadata.

- [ ] **Step 2: Confirm the new tests fail under total-only selection**

Run the new node IDs. Expected: total-only selection returns the requested
total despite a stratum deficit.

- [ ] **Step 3: Implement hard stratified selection**

Partition candidates by regime, allocate required counts from
`accepted_quota`, call the existing diversity ranker within each partition,
and never backfill a short partition. Return deterministic selected IDs plus
required/selected/deficit evidence.

- [ ] **Step 4: Advance and validate publication contracts**

Advance generator, producer, pair-report, generation-summary, and selection
version tokens. Recompute pair/run stratum counts from accepted visibility
vectors and reject old or internally inconsistent artifacts.

- [ ] **Step 5: Run SOP05 contract tests**

Run:

```bash
.conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_sop05_selection.py tests/test_sop05_run.py \
  tests/test_sop05_output_loader.py
```

Expected: all tests pass; quota-shortfall fixtures publish no completion
marker.

### Task 4: Preserve SOP06 Strata And Update Authority

**Files:**
- Modify: `src/generation/paired_variants.py`
- Modify: `docs/event_centered_blind_spot_implementation_spec.md`
- Modify: `docs/event_centered_blind_spot_agent_sops.md`
- Modify: `docs/parallel_acceleration_implementation_plan.md`
- Test: `tests/test_pair_variants.py`

- [ ] **Step 1: Write a failing non-empty variant drift test**

Construct a seen-then-occluded mother and a physically valid candidate whose
history is all false. Assert the variant is rejected with
`target_history_visibility_regime_changed`.

- [ ] **Step 2: Implement the invariant and rerun the test**

Classify mother and candidate histories with the same frozen policy. Keep only
non-empty variants whose regime matches the mother. Leave target-empty
semantics unchanged and explicitly document that it is not an identical-input
counterfactual for a previously visible target.

Persist the normalized policy/digest in mother metadata and advance the paired
producer/group contract to `independent_partial_pairs_v2` /
`sop06_partial_pair_group_v2` so the acceptance change cannot be confused with
v1 output.

- [ ] **Step 3: Correct the authoritative wording**

Replace “历史窗口内满足不可见约束” with the two explicit strata, their current
hidden invariant, the configured mixture, and the required split-level
reporting. Do not claim the mixture is a natural-world frequency.

- [ ] **Step 4: Run the complete small validation set**

Run:

```bash
.conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_history_visibility.py \
  tests/test_dynamic_object_transplant.py \
  tests/test_sop05_selection.py \
  tests/test_sop05_run.py \
  tests/test_sop05_output_loader.py \
  tests/test_pair_variants.py \
  tests/test_observation_renderer.py
```

Expected: zero failures. Do not invoke any production data CLI.

- [ ] **Step 5: Review the final diff and workspace state**

Run:

```bash
git diff --check
git status --short
```

Confirm only the planned files changed, plus pre-existing unrelated user/agent
changes. Do not commit unless the user explicitly requests it.
