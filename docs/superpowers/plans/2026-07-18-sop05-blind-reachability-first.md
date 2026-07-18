# SOP05 Blind-Reachability-First Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the low-yield, mode-collapsed SOP05 v4 proposal with a deterministic environment-occlusion generator that uses real 23-point snippets, publishes enough valid collision mothers, and treats complete sixpacks as a separate audit subset.

**Architecture:** A strict SOP04 future-time gate feeds three pure proposal units: causal free-space occluder placement, renderer-identical footprint-safe blind masks, and snippet-specific reachable-arc queries. `event_sampler.py` orchestrates a bounded deterministic schedule and delegates all final decisions to the existing continuous signed-clearance and exact visibility validators. Producer, loader, reporting, and partial-pair contracts advance together; old v4 artifacts are rejected rather than reinterpreted.

**Tech Stack:** Python 3.11, NumPy, PyYAML, pytest, Pillow/Matplotlib for audit output, existing geometry/serialization utilities, Slurm CPU jobs.

---

## File map

- Create `src/generation/blind_reachability.py`: pure snippet displacement, finite angle schedule, exact SE(2) start-pose construction, mask queries, chord triage, stable candidate identities.
- Create `src/generation/causal_occluder.py`: deterministic free-space environment-occluder proposals independent of one conflict-point normal.
- Create `src/generation/blind_region.py`: formal-renderer input assembly and footprint/yaw-safe center-mask broad phase.
- Create `src/generation/robot_sweep_cache.py`: strict-key immutable preparation cache for the small SOP04 trajectory bank and per-base sweeps.
- Modify `src/generation/occluder_sampler.py`: allow the existing continuous validator to consume prepared sweeps without changing its signed-clearance semantics.
- Modify `src/generation/event_sampler.py`: environment-only v5 orchestration and exact continuous acceptance.
- Modify `src/generation/sop05_input_adapter.py`: require corrected SOP04 future-endpoint layout.
- Modify `src/generation/sop05_selection.py`: v5 producer/report tokens and deterministic diversity-aware total selection.
- Modify `src/generation/sop05_run.py`, `sop05_output_loader.py`, `sop05_publication_identity.py`: v5 publication evidence and stage conservation.
- Modify `src/generation/paired_variants.py`, `sop06_pipeline.py`: independent partial-pair contract and separate complete-sixpack audit gate.
- Modify `configs/generator_train.yaml`, `configs/generator_test.yaml`, `configs/paired_variants.yaml`: explicit v5 proposal and partial-pair policies.
- Add focused tests under `tests/test_blind_reachability.py`, `tests/test_causal_occluder.py`, and `tests/test_blind_region.py`; modify only directly affected SOP05/SOP06 tests.
- Modify the three authoritative SOP documents only after executable behavior is green.

### Task 1: Gate the corrected SOP04 time layout

**Files:**
- Modify: `src/generation/sop05_input_adapter.py`
- Modify: `tests/test_sop05_input_adapter.py`
- Modify: `tests/test_sop05_run.py`

- [ ] **Step 1: Write failing input-layout tests**

Add fixtures whose trajectory manifest records either
`future_endpoints_dt_to_horizon_v1` with offsets `0.2..3.0` or the old
`0.0..2.8` layout. Require exact metadata:

```python
assert trajectory.metadata["pose_time_layout_version"] == (
    "future_endpoints_dt_to_horizon_v1"
)
np.testing.assert_allclose(
    trajectory.metadata["pose_time_offsets_s"],
    (np.arange(15, dtype=np.float64) + 1.0) * 0.2,
    rtol=0.0,
    atol=1e-12,
)
```

Assert missing, old, reordered, nonfinite, and inconsistent endpoint metadata
raise `Sop05InputError` before any BaseState is loaded.

- [ ] **Step 2: Run RED on Slurm**

```bash
srun -p gpu -c 4 --mem=8G -t 00:10:00 \
  /home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI/.conda-envs/sop4-risk/bin/python \
  -m pytest -q tests/test_sop05_input_adapter.py tests/test_sop05_run.py
```

Expected: new-layout fixtures fail because the adapter does not yet require or
preserve the layout.

- [ ] **Step 3: Implement the strict gate**

Add constants and validation without a compatibility branch:

```python
SOP04_POSE_TIME_LAYOUT_VERSION = "future_endpoints_dt_to_horizon_v1"
SOP05_INPUT_LOCK_VERSION = "sop05_input_lock_v2"

expected = (np.arange(_FUTURE_STEPS, dtype=np.float64) + 1.0) * _SAMPLE_DT_S
if metadata.get("pose_time_layout_version") != SOP04_POSE_TIME_LAYOUT_VERSION:
    raise Sop05InputError("SOP04 trajectory pose-time layout is not v1 future endpoints")
if not np.array_equal(np.asarray(metadata["pose_time_offsets_s"]), expected):
    raise Sop05InputError("SOP04 trajectory pose-time offsets mismatch")
```

Bind the layout token and offsets digest into the input lock/run identity.

- [ ] **Step 4: Run GREEN and commit the exact files**

Run Step 2; expected all pass. Commit only the adapter and its tests:

```bash
/home/home/ccnt_zq/zq_zhouyiqun/.local/git/bin/git add \
  src/generation/sop05_input_adapter.py \
  tests/test_sop05_input_adapter.py tests/test_sop05_run.py
/home/home/ccnt_zq/zq_zhouyiqun/.local/git/bin/git commit \
  -m "fix: require aligned SOP04 future timestamps"
```

### Task 2: Implement snippet-specific reachable arcs

**Files:**
- Create: `src/generation/blind_reachability.py`
- Create: `tests/test_blind_reachability.py`

- [ ] **Step 1: Write RED geometry tests**

Cover both crossing sides, exact displacement preservation, finite angle
schedule endpoints, no cell-center snapping, stable IDs, and malformed inputs.
The central invariant is:

```python
candidate = build_reachability_candidate(
    conflict_point=np.array([2.0, 0.5]),
    source_current_xy=np.array([0.0, 0.0]),
    source_anchor_xy=np.array([1.2, 0.0]),
    desired_crossing_direction=np.array([0.0, 1.0]),
    identity=identity,
)
transformed_anchor = candidate.current_xy + candidate.rotation @ np.array([1.2, 0.0])
np.testing.assert_allclose(transformed_anchor, [2.0, 0.5], atol=1e-9)
```

Add a mask-query test where two exact starts fall in one cell but retain
different float64 coordinates and different stable identities.

- [ ] **Step 2: Run RED**

```bash
srun -p gpu -c 4 --mem=8G -t 00:10:00 \
  /home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI/.conda-envs/sop4-risk/bin/python \
  -m pytest -q tests/test_blind_reachability.py
```

Expected: import failure for the new module.

- [ ] **Step 3: Implement the minimal pure module**

Define frozen dataclasses `ReachabilityIdentity`, `ReachabilityCandidate`, and
`ChordTriage`; validate float64 finite inputs; derive rotation from source and
desired displacement headings; calculate exact `x0`; query but never replace
`x0` with grid coordinates. Define:

Expose `BLIND_REACHABILITY_ALGORITHM_VERSION =
"blind_reachability_first_v1"`, `REACHABLE_ARC_SCHEDULE_VERSION =
"reachable_arc_schedule_v1"`, `scheduled_crossing_directions(normal_xy,
maximum_angle_deg, angle_step_deg)`, and
`build_reachability_candidate(conflict_point, source_current_xy,
source_anchor_xy, desired_crossing_direction, identity)`. The schedule uses
inclusive integer step indices in degrees, emits negative-side directions
before positive-side directions, and rejects a maximum angle that is not an
integer multiple of the step. The builder normalizes the desired direction,
computes `rotation_rad = atan2(desired) - atan2(source_anchor-source_current)`,
constructs the 2x2 rotation, computes `current_xy = conflict_point -
rotation @ delta_s`, and derives its stable ID from the complete identity plus
the float64 hexadecimal values of `rotation_rad` and `current_xy`.

`ChordTriage` has only `certified_clear` and `unresolved`; obstacle intersection
cannot prove collision or silently discard curved snippets.

- [ ] **Step 4: Run GREEN, repeat for determinism, and commit**

Run the Step 2 test twice, then commit the exact two files.

### Task 3: Build causal occluders and renderer-identical blind masks

**Files:**
- Create: `src/generation/causal_occluder.py`
- Create: `src/generation/blind_region.py`
- Create: `src/generation/robot_sweep_cache.py`
- Create: `tests/test_causal_occluder.py`
- Create: `tests/test_blind_region.py`
- Create: `tests/test_robot_sweep_cache.py`
- Modify: `src/generation/occluder_sampler.py`
- Create: `tests/test_occluder_sampler.py`

- [ ] **Step 1: Write RED causal-obstacle tests**

In a toy empty 160x160 grid, assert the seeded schedule covers at least four
bearing quadrants, never overlaps the continuous robot/context sweeps, is not
derived from a conflict-point normal, emits one stable proposal identity, and
records explicit free-space/shadow rejection reasons.

- [ ] **Step 2: Write RED robot-sweep cache tests**

Build a toy bank with repeated use of the same trajectory. Require exactly one
future-sweep preparation for one strict key, read-only cached arrays, stable
cache identity, and byte-identical cold/warm entries. The key must bind
trajectory ID, exact pose and persisted swept-mask digests, robot footprint,
grid, corrected pose-time layout, and preparation algorithm version. Changing
any bound input must miss or fail; the same trajectory ID with different poses
must never alias.

Require prepared and legacy unprepared paths through
`occluder_collision_sweep_rejection_reason` to return identical results for
clearance, frame contact, and between-frame contact. Cache only canonical
future geometry globally per worker; prepare robot history per base and context
motion per base/object. A stored SOP04 `swept_mask` is broad-phase evidence
only and cannot decide collision by itself.

- [ ] **Step 3: Write RED blind-mask tests**

Require `build_blind_region` to call the same ray-casting kernel used by the
formal renderer and to combine base static, causal obstacle, current context,
FOV, and range while rejecting any oracle-future input. Assert byte equality
with a direct renderer-kernel call. Test circle masks and yaw-binned rectangle
masks, followed by exact polygon visibility.

- [ ] **Step 4: Run RED**

```bash
srun -p gpu -c 8 --mem=12G -t 00:15:00 \
  /home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI/.conda-envs/sop4-risk/bin/python \
  -m pytest -q tests/test_causal_occluder.py tests/test_blind_region.py \
  tests/test_robot_sweep_cache.py tests/test_occluder_sampler.py \
  tests/test_occluder_visibility.py tests/test_observation_renderer.py
```

- [ ] **Step 5: Implement minimal modules using existing geometry**

Reuse `OccluderCollisionSweep`, continuous signed-clearance certification,
`rasterize_footprint`, `raycast_visibility`, and
`footprint_visibility_sequence`. Do not duplicate collision or renderer
semantics. Proposal IDs bind type, exact pose/dimensions, schedule index, seed,
base ID, and trajectory ID.

Prepare the current bank once per worker through
`ROBOT_SWEEP_CACHE_VERSION = "robot_sweep_cache_v1"`. Reuse canonical
dense future poses and interval motion geometry; do not cache
occluder-specific clearances, target motion, visibility, static occupancy, or
context from another base. Keep the old unprepared validator entry point for
v4 callers and require numerical/verdict equivalence in tests.

- [ ] **Step 6: Run GREEN and commit**

Run Step 4. Require no change to legacy v4 tests except version-specific
fixtures. Commit the three new modules, three new tests, and the directly
modified occluder sampler/test. The formal config files advance atomically with
the v5 event-sampler normalizer in Task 4 so Task 3 cannot leave checked-in
configs unreadable by the production entry point.

### Task 4: Orchestrate environment collision mothers

**Files:**
- Modify: `src/generation/event_sampler.py`
- Modify: `src/generation/dynamic_object_transplant.py`
- Modify: `tests/test_dynamic_object_transplant.py`
- Modify: `tests/test_occluder_visibility.py`
- Modify: `configs/generator_train.yaml`
- Modify: `configs/generator_test.yaml`

- [ ] **Step 1: Add explicit config with the v5 normalizer tests**

Replace the v4 event-kind mixture with this identical scientific section in
train/test configs; only `obstacle_proposals_per_trajectory` may be larger in
train after the benchmark:

```yaml
production_event_kind: environment
blind_reachability:
  algorithm_version: blind_reachability_first_v1
  obstacle_proposals_per_trajectory: 64
  interaction_range_m: [1.0, 4.0]
  bearing_bin_count: 12
  yaw_step_deg: 30.0
  crossing_angle_step_deg: 5.0
  minimum_shadow_center_cells: 32
  chord_deviation_fastpath_m: 0.15
  unresolved_exact_fallback_per_anchor: 16
```

Write a RED test that the v5 normalizer accepts exactly these keys and rejects
v4 `event_type_weights`/`structural_fov`. Change the configs and normalizer in
the same GREEN commit so no checked-in config is transiently invalid.

- [ ] **Step 2: Write RED mother-event fixtures**

Add left-, right-, and oblique-occluder toy cases. Each accepted event must use
one real `MotionSnippet`, one causal environment occluder, exact same-index
robot/target collision, current full-footprint invisibility, future emergence,
and no target/occluder/context collision over the continuous 23-frame sweep.

Add rejection tests for no mask/arc intersection, invalid source anchor,
partial current visibility, chord-unresolved-but-exact-safe curved motion,
continuous between-frame obstacle contact, and an unexpected exception that
must abort rather than count as physical rejection.

- [ ] **Step 3: Run RED**

```bash
srun -p gpu -c 8 --mem=16G -t 00:20:00 \
  /home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI/.conda-envs/sop4-risk/bin/python \
  -m pytest -q tests/test_dynamic_object_transplant.py \
  tests/test_occluder_visibility.py
```

- [ ] **Step 4: Replace the production branch, not v4 semantics in place**

Set `SOP05_GENERATOR_ALGORITHM_VERSION = "blind_reachability_first_v1"`.
Normalize `production_event_kind=environment` and the exact config keys. In a
bounded work item:

1. enumerate all eligible trajectory anchors at aligned timestamps;
2. iterate the stable free-space obstacle schedule;
3. build and digest the blind mask;
4. query split/type/anchor-valid snippets and scheduled crossing directions;
5. retain exact float64 reachability candidates whose current footprints query
   hidden/free;
6. prioritize certified-clear chords, then bounded unresolved candidates;
7. transplant the untouched snippet by SE(2);
8. run exact renderer visibility and continuous-sweep validation; and
9. retain every accepted result in the bounded schedule before stable
   selection.

Do not retain a structural/mixed fallback in the formal v5 function.

- [ ] **Step 5: Emit stable stage identities and conservation counters**

Every report contains obstacle proposal IDs, arc-query IDs, transform IDs,
exact-validation IDs, stage counts, rejection reasons, and equations:

```text
obstacle_proposals = obstacle_rejected + obstacle_passed
transform_candidates = chord_certified + chord_unresolved + transform_rejected
exact_validations = exact_accepted + exact_rejected
```

- [ ] **Step 6: Run GREEN plus shape/dtype/determinism checks and commit**

Run Step 3 twice and the focused reachability tests. Commit only Task 4 files.

### Task 5: Advance producer, selector, loader, and publication evidence

**Files:**
- Modify: `src/generation/sop05_selection.py`
- Modify: `src/generation/sop05_run.py`
- Modify: `src/generation/sop05_output_loader.py`
- Modify: `src/generation/sop05_publication_identity.py`
- Modify: `scripts/05_generate_events.py`
- Modify: `tests/test_sop05_selection.py`
- Modify: `tests/test_sop05_run.py`
- Modify: `tests/test_sop05_output_loader.py`
- Modify: `tests/test_05_generate_events_cli.py`

- [ ] **Step 1: Write RED exact-version and tamper tests**

Require only these tokens:

```text
sop05_generation_run_v5
sop05_run_manifest_v3
sop05_generation_summary_v3
sop05_pair_generation_report_v3
sop05_producer_complete_v3
sop05_publication_semantic_digest_v2
sop05_input_lock_v2
blind_reachability_first_v1
```

Add resealed attacks on stage counts/IDs, SOP04 time-layout evidence, causal
occluder identity, candidate order, and CPU accounting. Old v4 publications
must fail explicitly.

- [ ] **Step 2: Run RED on Slurm**

```bash
srun -p gpu -c 8 --mem=20G -t 00:25:00 \
  /home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI/.conda-envs/sop4-risk/bin/python \
  -m pytest -q tests/test_sop05_selection.py tests/test_sop05_run.py \
  tests/test_sop05_output_loader.py tests/test_05_generate_events_cli.py
```

- [ ] **Step 3: Implement deterministic two-stage selection**

Workers emit canonical candidates and never consult shared frequency counts.
The reducer first canonicalizes by stable base/trajectory/event key, then
performs a deterministic greedy diversity ranking with stable tie-breaks. It
always returns the requested total when enough candidates exist; diversity is
never a hard quota. Bind selection version `sop05_diversity_total_selection_v1`.

- [ ] **Step 4: Advance all evidence atomically**

Require exact schemas, conservation equations, corrected SOP04 lock, code
identity, configuration digest, candidate counts, allocated CPU seconds, and
external semantic digest. Run the formal loader on staging before rename.

- [ ] **Step 5: Run GREEN and commit**

Run Step 2 and `tests/test_event_target_motion_shard.py`; commit exact owned
files.

### Task 6: Decouple partial pairs from complete-sixpack audit

**Files:**
- Modify: `src/generation/paired_variants.py`
- Modify: `src/generation/sop06_pipeline.py`
- Modify: `configs/paired_variants.yaml`
- Modify: `tests/test_pair_variants.py`
- Modify: `tests/test_sop06_pipeline.py`

- [ ] **Step 1: Write RED sparse-coverage tests**

Require `collision` only for a singleton mother group, independently retain
each successfully generated negative, preserve six-position coverage order and
enumerated missing reasons, and keep a stable mother-derived `pair_group_id`.
Assert complete audit mode still requires all six and is explicitly
conditional.

- [ ] **Step 2: Change the config contract**

Replace the old hard minimum with:

```yaml
paired_generator_algorithm_version: independent_partial_pairs_v1
group_contract_version: sop06_partial_pair_group_v1
mother_required_variants: [collision]
training_minimum_contrast_count: 0
audit_requires_all_variants: true
```

- [ ] **Step 3: Run RED**

```bash
srun -p gpu -c 8 --mem=16G -t 00:20:00 \
  /home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI/.conda-envs/sop4-risk/bin/python \
  -m pytest -q tests/test_pair_variants.py tests/test_sop06_pipeline.py
```

- [ ] **Step 4: Implement three explicit consumer paths**

Provide mother rendering, partial-pair rendering, and complete-audit rendering.
Do not allow the training path to silently claim complete evaluation. Reject
`joint_environment_pair_v2` in formal v5 input.

- [ ] **Step 5: Run GREEN and commit**

Run Step 3 plus `tests/test_observation_renderer.py`; commit the exact five
files.

### Task 7: Align authoritative documentation

**Files:**
- Modify: `docs/event_centered_blind_spot_implementation_spec.md`
- Modify: `docs/event_centered_blind_spot_agent_sops.md`
- Modify: `docs/parallel_acceleration_implementation_plan.md`

- [ ] **Step 1: Replace, do not append to, v4 production semantics**

Update SOP05 flow, algorithm/version gates, aligned time index, physical
environment mother acceptance, reachability arcs, continuous exact validation,
stage reporting, sparse pairs, and conditional sixpack audit. Preserve v4 only
in migration/history wording; it is not a valid current producer.

- [ ] **Step 2: Check frozen-file and contract consistency**

```bash
rg -n "joint_occluder_first_v4|joint_environment_pair_v2|sixpack|0\.0-2\.8|0\.2-3\.0" \
  docs/event_centered_blind_spot_implementation_spec.md \
  docs/event_centered_blind_spot_agent_sops.md \
  docs/parallel_acceleration_implementation_plan.md
```

Expected: every old token is explicitly historical/rejected; current clauses
name only v5. Confirm no diff in `src/contracts.py`, `configs/base.yaml`,
`DECISIONS.md`, or `STATUS.md`.

- [ ] **Step 3: Commit exact documentation files**

Commit only the three authoritative documents and this plan if not already
committed.

### Task 8: Slurm benchmark, random-mother audit, and formal publication

**Files:**
- No committed output; write named artifacts only under ignored `outputs/`.
- Temporary helpers only under `.tmp/agent/{scripts,logs,outputs}` and delete
  them before handoff.

- [ ] **Step 1: Run the focused unit/integration suite**

```bash
srun -p gpu -c 16 --mem=32G -t 00:40:00 \
  /home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI/.conda-envs/sop4-risk/bin/python \
  -m pytest -q tests/test_blind_reachability.py tests/test_causal_occluder.py \
  tests/test_blind_region.py tests/test_dynamic_object_transplant.py \
  tests/test_occluder_visibility.py tests/test_sop05_input_adapter.py \
  tests/test_sop05_selection.py tests/test_sop05_run.py \
  tests/test_sop05_output_loader.py tests/test_05_generate_events_cli.py \
  tests/test_event_target_motion_shard.py tests/test_pair_variants.py \
  tests/test_sop06_pipeline.py tests/test_observation_renderer.py
```

Expected: all pass; no xfail/skip added to hide failures.

- [ ] **Step 2: Run same-budget v4/v5 smoke**

Use the same trusted SOP03 inputs, corrected SOP04 bank, base/trajectory work
pool, seed, 32 CPUs, memory, and wall-time accounting. Produce 10-100 accepted
mothers and report root yield, exact acceptance, accepted mothers/CPU-hour,
every rejection stage, unique IDs, and determinism. Do not launch target scale
unless v5 is at least 2x v4 throughput with all invariants green; target is 5x.

- [ ] **Step 3: Generate two separate visual audits**

From the published mother set, stable-randomly sample five mothers before any
pair-success conditioning and write `event_replay.gif` plus mother geometry
PNGs. Separately generate five complete sixpacks if available, label the set
conditional, and never substitute it for random-mother review.

- [ ] **Step 4: Inspect scientific and diversity invariants**

Check shape, dtype, NaN/Inf, exact timestamps, current invisibility, future
emergence, continuous target/obstacle clearance, intended robot collision,
source lineage, split leakage, repeat-load identity, obstacle bearings,
trajectory IDs, approach sides, conflict times, and generator-pattern alerts.

- [ ] **Step 5: Run target-scale CPU generation**

Use a new output directory, Slurm CPU parallelism, stable work-item identities,
verified checkpoints, atomic publication, and formal loader round-trip. Never
overwrite v4 or source data. Save exact command, Slurm job IDs/resources,
config/code/input digests, counts, runtime, and external publication digest.

- [ ] **Step 6: Final verification and local commit**

Run `git diff --check`, both worktree status checks, and the scoped regression
suite. Stage only exact owned source/tests/config/docs/plan files, create the
required local commit, and return commit hash, changed files, commands/results,
publication paths/digests, limitations, and the next safe downstream task.
