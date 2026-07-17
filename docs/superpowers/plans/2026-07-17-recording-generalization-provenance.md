# Recording-Generalization and MotionSnippet v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the frozen 37/5/5/5 THÖR recording assignment, state its known-session evaluation scope honestly, and rebuild SOP-03 with measured 8-frame history, current index 7, and 15-frame future in every MotionSnippet v2 artifact.

**Architecture:** Keep strict split isolation as the generic default and opt THÖR into an explicit recording-generalization policy where recording/source-object overlap is forbidden and recording-day session overlap is enumerated but allowed. Freeze official FILE_ID metadata before SOP-03, then propagate one validated split digest through recording, base-state, oracle, and snippet outputs. Replace the old 16-point snippet writer/loader with one strict 23-point layout and rebuild into a new versioned Slurm artifact root.

**Tech Stack:** Python 3.10, standard library (`csv`, `hashlib`, `json`, `pathlib`), NumPy, existing PyYAML, pytest, Git worktrees, Slurm.

---

### Task 1: Policy-aware, coverage-aware split audit

**Files:**
- Modify: `src/datasets/split_manager.py`
- Modify: `tests/test_split_leakage.py`
- Modify: `tests/test_split_manager.py`

- [x] **Step 1: Write policy RED tests**

Test the approved policy:

```python
SplitAuditPolicy(
    evaluation_scope="unseen_recording_within_known_sessions",
    required_fields=("recording", "session", "seed_namespace"),
    allowed_overlap_fields=("session",),
    unavailable_fields=("participant",),
)
```

Require complete session coverage, one reported/allowed toy session overlap,
zero disallowed overlap, and failure when a required session is missing.
Retain the generic strict test that rejects session overlap.

- [x] **Step 2: Verify RED on Slurm**

Run `$PY -m pytest tests/test_split_leakage.py tests/test_split_manager.py -q`.
Observed expected failure because the policy API did not exist.

- [x] **Step 3: Implement the minimal policy API**

Add frozen `SplitAuditPolicy`, policy validation, field coverage, detected versus
allowed versus disallowed overlap counts, and policy-aware
`assert_no_split_leakage` while preserving strict default behavior.

- [x] **Step 4: Implement deterministic preassigned freezing**

Add `freeze_preassigned_split(records, split_by_recording, *, seed, policy)`.
Require exact recording ID equality, stable seed namespaces, canonical JSONL,
BLAKE2b-128 manifest digest, complete required provenance, and explicit policy
fields in every row/summary/audit.

- [x] **Step 5: Verify GREEN on Slurm**

Observed 29 passing targeted split tests.

### Task 2: Official THÖR metadata index and frozen split CLI

**Files:**
- Create: `src/datasets/thor_split.py`
- Create: `scripts/00_freeze_thor_recording_split.py`
- Create: `configs/data_thor_recording_generalization.yaml`
- Create: `tests/test_thor_split.py`

- [x] **Step 1: Write metadata/CLI RED tests**

Use toy first rows `FILE_ID,120522_SC1A_R1`. Require filename/FILE_ID agreement,
six-digit recording-day sessions, exact assignment coverage, normalization of
`validation` to `val`, and byte-identical repeated artifacts.

- [x] **Step 2: Verify RED on Slurm**

Observed seven expected failures because the module and CLI did not exist.

- [x] **Step 3: Implement metadata indexing and assignment loading**

Read only `Scenario_*/THOR-Magni_*.csv` first rows, validate official IDs, and
import the existing 37/5/5/5 recording assignment without regrouping sessions.

- [x] **Step 4: Implement atomic artifacts and semantic config**

Write `recording_metadata.jsonl`, `split_manifest.jsonl`, `split_summary.json`,
and `overlap_report.json` through a staging directory. Freeze:

```yaml
evaluation_scope: unseen_recording_within_known_sessions
grouping_unit: recording_id
recording_overlap_policy: forbidden
session_overlap_policy: allowed_reported
participant_overlap_policy: unavailable
```

- [x] **Step 5: Verify GREEN on Slurm**

Observed 36 passing split/THÖR metadata tests.

### Task 3: Propagate the approved split provenance through SOP-03

**Files:**
- Modify: `src/datasets/split_manager.py`
- Modify: `src/datasets/thor_adapter.py`
- Modify: `src/datasets/base_state_index.py`
- Modify: `src/datasets/snippet_library.py`
- Modify: `scripts/01_index_recordings.py`
- Modify: `scripts/02_build_snippet_library.py`
- Modify: `scripts/03_extract_base_states.py`
- Modify: `tests/test_split_manager.py`
- Modify: `tests/test_thor_adapter.py`
- Modify: `tests/test_base_state_index.py`
- Modify: `tests/test_snippet_library.py`

- [x] **Step 1: Write provenance RED tests**

Use this exact payload:

```python
provenance = {
    "split_manifest_digest": "0123456789abcdef0123456789abcdef",
    "evaluation_scope": "unseen_recording_within_known_sessions",
    "grouping_unit": "recording_id",
    "field_policies": {
        "recording": "forbidden",
        "session": "allowed_reported",
        "participant": "unavailable",
    },
}
```

Assert identical `split_provenance` in recording, base-state, oracle-context,
and snippet manifests/summaries and in snippet NPZ metadata. Reject missing
keys, NaN/Inf, malformed BLAKE2b-128 hex, and mismatched split digests.

- [x] **Step 2: Run RED on Slurm**

Run:

```bash
$PY -m pytest \
  tests/test_split_manager.py tests/test_thor_adapter.py \
  tests/test_base_state_index.py tests/test_snippet_library.py -q
```

Expected: signature/assertion failures because writers do not yet accept the
payload.

- [x] **Step 3: Implement shared validation and writers**

Add `validate_split_provenance` to `split_manager.py`. Require exactly a
32-character lowercase hex `split_manifest_digest`, non-empty evaluation scope
and grouping unit, JSON-safe finite values, and policy values from
`forbidden/allowed_reported/unavailable`. Return a canonical copied mapping.
Make all SOP-03 writers require the mapping and stamp it consistently.

- [x] **Step 4: Implement CLI provenance loading**

Script 01 reads the sibling split summary, recomputes BLAKE2b-128 over the
canonical manifest bytes, and rejects mismatch before raw parsing. Recording
summaries become the source of truth for scripts 02/03. Both downstream scripts
require identical provenance across loaded splits. The old SHA-256
`c8777a22dac12d12bce64c31c97290cf1791eb73507077478e40b59ed4eef061`
may appear only as read-only source-assignment provenance, never as
`split_manifest_digest`.

- [x] **Step 5: Run GREEN on Slurm**

Run the same tests and require zero failures, no pickle/object arrays, and
deterministic JSON bytes.

### Task 4: Freeze MotionSnippet v2 as a strict 23-point layout

**Files:**
- Modify: `src/datasets/snippet_library.py`
- Modify: `scripts/02_build_snippet_library.py`
- Modify: `tests/test_snippet_library.py`

- [x] **Step 1: Write layout RED tests**

Request the production API without a duration override and assert:

```python
assert snippet.positions.shape == (23, 2)
assert snippet.velocities.shape == (23, 2)
assert snippet.headings.shape == (23,)
assert snippet.positions.dtype == np.float32
assert snippet.velocities.dtype == np.float32
assert snippet.headings.dtype == np.float32
assert library.summary["motion_snippet_layout_version"] == (
    "history8_current7_future15_v1"
)
assert library.summary["sample_count"] == 23
assert library.summary["history_steps"] == 8
assert library.summary["future_steps"] == 15
assert library.summary["current_index"] == 7
assert library.summary["sample_dt_s"] == 0.2
assert library.summary["duration_s"] == 4.4
```

Verify source times `0.0 ... 1.4 ... 4.4`, measured history span 1.4 s, and
future span 3.0 s. Create a missing-layout NPZ and a 16-point/3.0 s NPZ; both
load attempts must fail explicitly.

- [x] **Step 2: Write gap, no-extrapolation, and determinism RED tests**

Split a track into two segments with a gap larger than 0.3 s and assert no
snippet crosses it. A 4.2 s track yields zero accepted snippets and an explicit
`insufficient_contiguous_duration` rejection. Repeat builds under distinct
`PYTHONHASHSEED` values and assert identical IDs, source manifest bytes, and
stored array SHA-256.

- [x] **Step 3: Run RED on Slurm**

Run `$PY -m pytest tests/test_snippet_library.py -q`. Expected failures must
show the current 16-point default, permissive loader, or missing rejection.

- [x] **Step 4: Implement the minimal frozen layout**

Define one immutable mapping:

```python
MOTION_SNIPPET_LAYOUT = {
    "motion_snippet_layout_version": "history8_current7_future15_v1",
    "sample_count": 23,
    "history_steps": 8,
    "future_steps": 15,
    "current_index": 7,
    "sample_dt_s": 0.2,
    "duration_s": 4.4,
}
```

Make `build_snippet_library` accept only this layout. If `--duration-s` remains,
it is an exact-value guard: 3.0 and every non-4.4 value fail. Add
`source_session_id`, keep first-point/initial-motion normalization, headings,
footprint/type/raw provenance, and include layout version in ID derivation.
Count each too-short contiguous segment as a rejected candidate, preserving
`candidate_count == accepted_count + rejected_count`.

- [x] **Step 5: Implement strict writer/loader validation**

Stamp every layout field plus `split_manifest_digest` in NPZ metadata, library
summary, and source-manifest rows. Empty arrays have numeric shapes `(0,23,2)`,
`(0,23,2)`, and `(0,23)`. Compute SHA-256 over array names, shapes, dtypes, and
C-order bytes. Validate layout, shapes, float32, finite values, summary/digest
consistency, and provenance before constructing any object.

- [x] **Step 6: Run GREEN on Slurm**

Run the focused test file and require zero failures.

### Task 5: Quantify resampling kinematics before and after conversion

**Files:**
- Modify: `src/datasets/thor_adapter.py`
- Modify: `tests/test_thor_adapter.py`

- [x] **Step 1: Write diagnostic RED tests**

Use a toy curved trajectory whose speed, acceleration magnitude, and absolute
curvature are finite and hand-checkable. Require raw/resampled sample counts and
p05/p50/p95 for all three quantities, separately for robot and dynamic objects.
Straight constant-speed tracks must report approximately zero acceleration and
curvature.

- [x] **Step 2: Run RED on Slurm**

Run `$PY -m pytest tests/test_thor_adapter.py -q`; expect missing acceleration
and curvature fields.

- [x] **Step 3: Implement segment-safe diagnostics**

Within each existing segment, derive velocity, acceleration, unwrapped heading
rate, and curvature. Never differentiate across a gap, and set curvature to
zero below a small speed threshold. Aggregate finite samples deterministically
and record p05/p50/p95 plus absolute raw/resampled deltas without changing
trajectory acceptance.

- [x] **Step 4: Run GREEN on Slurm**

Require duplicate-timestamp, auxiliary-row, gap, shape/dtype, finite, and new
diagnostic tests all to pass.

### Task 6: Enforce v2 layout/provenance in production CLIs and reports

**Files:**
- Modify: `scripts/01_index_recordings.py`
- Modify: `scripts/02_build_snippet_library.py`
- Modify: `scripts/03_extract_base_states.py`
- Modify: `tests/test_thor_adapter.py`
- Modify: `tests/test_base_state_index.py`
- Modify: `tests/test_snippet_library.py`

- [x] **Step 1: Write CLI integration RED tests**

Build a toy enriched split and run scripts 01--03. Require the same split
provenance everywhere, script 02 default 4.4 s, explicit failure for
`--duration-s 3.0`, and pre-write rejection of missing/stale/mixed digests.

- [x] **Step 2: Run RED on Slurm**

Run the three SOP-03 test files and confirm failures arise from absent CLI
propagation or the old duration default.

- [x] **Step 3: Implement minimal propagation and audit**

Script 01 validates the frozen split. Scripts 02/03 validate recording summary
provenance. Script 02 uses the fixed layout for every type and runs a
policy-aware cross-split audit: session overlaps are reported/allowed, while
recording and source-object overlap are fatal. Each split/type summary reports
candidate, accepted, rejected, and deterministic reason counts.

- [x] **Step 4: Verify GREEN and serial/parallel equality**

Run toy producers with one and two workers. Require identical summaries, IDs,
arrays, manifests, and array digests.

### Task 7: Update authoritative documentation consistently

**Files:**
- Modify: `docs/event_centered_blind_spot_implementation_spec.md`
- Modify: `docs/parallel_acceleration_implementation_plan.md`
- Modify: `docs/event_centered_blind_spot_agent_sops.md`
- Modify: `docs/superpowers/specs/2026-07-17-recording-generalization-provenance-design.md`
- Modify: `docs/superpowers/plans/2026-07-17-recording-generalization-provenance.md`

- [x] **Step 1: Document generic versus THÖR split policy**

Keep recording/session/available-participant isolation as the generic default.
For THÖR, state recording overlap forbidden, session overlap allowed/reported,
participant identity unavailable, and claims limited to unseen recordings from
known recording-day sessions.

- [x] **Step 2: Document the exact snippet/SOP-05 time contract**

Replace the production 3--5 s ambiguity with the frozen layout. State:

```text
history = transformed_poses[0:8]
current = transformed_poses[7]
future = transformed_poses[8:23]
source_anchor_time = 1.4 + conflict_time_s
```

SOP-03 keeps measured source motion and first-point normalization only;
event-level SE(2) placement belongs to SOP-05.

- [x] **Step 3: Check documentation consistency**

Run `git diff --check` and targeted `rg` only in the three authority documents.
Resolve unconditional THÖR `session overlap = 0`, production 16-point/3.0 s,
and direct `source_anchor_time=conflict_time_s` claims. Do not modify
`src/contracts.py`, `configs/base.yaml`, `DECISIONS.md`, or `STATUS.md`.

### Task 8: Unit, toy, and 10-recording Slurm smoke validation

**Files:**
- Temporary only: `.tmp/agent/scripts/${STAMP}-motion-snippet-v2-smoke-repro.sh`
- Temporary only: `.tmp/agent/outputs/${STAMP}-motion-snippet-v2-smoke/`

- [x] **Step 1: Run focused tests with 8 Slurm CPUs**

Run SOP-01/SOP-03 unit tests plus `tests/test_contracts.py` and
`tests/test_toy_fixture.py`. Every Python process, including audit helpers,
runs inside the allocation.

- [x] **Step 2: Build a 10-recording smoke**

Freeze all 52 split rows, then index four train and two each calibration/val/test
recordings. Run scripts 01--03 with eight workers, `dt=0.2`, `max_gap=0.3`,
fixed 4.4 s snippets, snippet stride 1.0 s, and base-state stride 0.6 s.

- [x] **Step 3: Audit smoke invariants and repeatability**

Check split provenance equality, `[23,2]/[23]` shapes, float32, finite values,
no cross-gap window, base/oracle alignment, typed footprints, deterministic
IDs/digests, allowed session overlap, forbidden recording/object overlap, and
repeat-run byte/numerical equality. Delete smoke artifacts and scripts after
recording results.

- [x] **Step 4: Run the full repository test suite**

Run `$PY -m pytest -q` on Slurm. Require zero failures before committing.

### Task 9: Commit the implementation

**Files:**
- Stage only the exact files listed in Tasks 1--7.

- [x] **Step 1: Inspect owned changes**

Run exact-git `status --short`, `diff --check`, and `diff --stat`. Verify no raw
data, outputs, temporary files, protected files, dependency files, or other
Agent files are staged.

- [x] **Step 2: Stage exact paths and commit**

Never use `git add .`. Commit with:

```bash
$GIT commit -m "feat(data): add measured-history THOR snippets"
```

Record the full hash for artifact provenance.

### Task 10: Full SOP-03 Slurm rebuild and independent audit

**Files:**
- Generate only: `/home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI/outputs/sop03_thor_motion_snippet_v2_${SHORT}_v1/`

- [ ] **Step 1: Check for and stop only obsolete owned 3.0 s jobs**

Inspect Slurm command lines. Cancel only an identifiable job from this workflow
that is still producing the old 3.0 s library. Do not touch unrelated jobs.

- [ ] **Step 2: Submit the 8-CPU full producer**

Freeze the enriched split, run all four recording indexes, all 12 split/type
snippet libraries, and all base-state/oracle indexes. Write a root run manifest,
logs, environment/resource record, rejection report, and SHA-256 checksum
manifest. Refuse overwrite and leave `sop03_thor_full_f582bc5_v1` unchanged.

- [ ] **Step 3: Verify production gates**

Require 52 rows/sessions, 37/5/5/5 counts, zero recording/object overlap, five
reported allowed session overlaps, zero disallowed overlap, and a new split
digest. Require exact v2 metadata and total snippets at least 1,000 (5,000
preferred). Report per split/type candidate/accepted/rejected/reasons and
compare accepted counts with the old 3.0 s run; reduction is expected and must
be explained.

- [ ] **Step 4: Produce kinematic and 50-example review reports**

Aggregate raw/resampled speed, acceleration, and curvature. Deterministically
select 50 robot/object/snippet examples covering every present object type,
generate review sheets and a JSON selection/decision record inside the artifact
root, and inspect direction, continuity, gaps/jumps, footprint/type retention,
and finite values. Record every flagged case and disposition without silently
replacing it.

- [ ] **Step 5: Run an independent Slurm audit**

Use a separate job to reopen NPZ with `allow_pickle=False`, verify checksums,
shape/dtype/finite invariants, split/source policies, array digests,
determinism, no extrapolation evidence, and old-output immutability. Return
commit hash, new split digest, artifact root, job IDs, exact commands/results,
split/type counts, checksums, rejection report, 50-example review, limitations,
and the next safe SOP-05 task. SOP-05 must use
`source_anchor_time=1.4+conflict_time_s`.
