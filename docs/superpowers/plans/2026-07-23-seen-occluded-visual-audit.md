# Real Seen-Then-Occluded Visual Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run an immutable audit-only producer that selects three real THOR-derived `seen_then_occluded` collision mothers with complete six-position paired groups and writes one replay GIF, one paired PNG, and one scientific audit record per sample.

**Architecture:** A focused visualization module converts already validated domain objects into deterministic replay layers and figures. A separate audit module authenticates schema-3 inputs, streams the stable SOP05 pair schedule, independently checks complete SOP06 groups, and atomically publishes a checksummed collection. A thin CLI owns argument parsing only. The final run reads real SOP03/SOP04 assets and never publishes a training shard.

**Tech Stack:** Python 3.10, NumPy 1.24.4, PyYAML 6.0.1, Matplotlib 3.7.5, Pillow 12.3.0, pytest 8.3.5, existing SOP03-SOP06 contracts and atomic rename helper.

**Workspace rule:** The worktree already contains unrelated user/agent changes. Do not reset, reformat, or commit them. This plan uses test and diff checkpoints instead of per-task commits.

**Empirical correction:** The completed independent search evaluated all 512
stable pairs and 199 real selected-regime mothers but produced zero complete
sixpacks. Task 3A below adds the audit-only joint occluder layer described in
the design; formal SOP05/SOP06 publication semantics remain unchanged.

---

## File Structure

- Create `src/evaluation/seen_occluded_visuals.py`: immutable visual-scene data, visibility reconstruction, deterministic GIF/PNG rendering, and image metadata inspection.
- Create `src/evaluation/seen_occluded_visual_audit.py`: request/result contracts, real search orchestration, independent scientific checks, attempt records, manifest/checksum construction, reload validation, and atomic publication.
- Create `scripts/05_render_seen_occluded_visual_audit.py`: CLI parsing, expected-error reporting, and exit codes.
- Create `tests/test_seen_occluded_visuals.py`: layer semantics, 23-frame timing, fixed panel order, nonblank pixels, and target-removal presentation.
- Create `tests/test_seen_occluded_visual_audit.py`: history regime, sixpack checks, stable attempt reasons, checksum envelope, failure marker, and no-overwrite behavior.
- Create `tests/test_05_render_seen_occluded_visual_audit_cli.py`: parser-to-request mapping and structured CLI errors without real search.
- Create `configs/generator_seen_occluded_visual_audit.yaml`: audit-only 100/0 history composition and 8-by-8 proposal prefix with unchanged physical/acceptance parameters.
- Create `configs/paired_variants_visual_audit.yaml`: bounded 0.1 m / 2.0 m spatial proposal grid with unchanged labels and strict sixpack gates.
- Modify `pyproject.toml`: add only the optional `visual-audit` dependency group.
- Create `outputs/sop05_seen_then_occluded_visual_audit_20260723_v1/`: generated real audit collection; no source code depends on it.

### Task 1: Freeze Visual Scene Contracts

**Files:**
- Create: `src/evaluation/seen_occluded_visuals.py`
- Create: `tests/test_seen_occluded_visuals.py`

- [ ] **Step 1: Write failing replay-contract tests**

Define a small represented scene with an 8-frame target history whose vector is
`[True, True, True, False, False, False, False, False]`, a 15-step future,
one rectangular occluder, one candidate trajectory, and one visible context
actor. Assert exact times and future visibility freezing:

```python
def test_build_replay_frames_uses_real_history_and_frozen_current_future():
    bundle = _visual_bundle()
    frames = build_replay_frames(bundle)
    assert [frame.time_s for frame in frames[:8]] == pytest.approx(
        [-1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0]
    )
    assert [frame.time_s for frame in frames[8:]] == pytest.approx(
        [0.2 * index for index in range(1, 16)]
    )
    assert len(frames) == 23
    for frame in frames[8:]:
        np.testing.assert_array_equal(frame.visibility_mask, frames[7].visibility_mask)
        assert frame.phase == "oracle_replay"
```

Also assert malformed shapes/dtypes, non-seen histories, mismatched variant
orders, and changing panel skeletons fail closed.

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
.conda-envs/sop4-risk/bin/python -m pytest -q tests/test_seen_occluded_visuals.py
```

Expected: collection fails because `src.evaluation.seen_occluded_visuals` does
not exist.

- [ ] **Step 3: Implement immutable visual data contracts**

Add exact constants and dataclasses:

```python
REPLAY_HISTORY_TIMES_S = tuple((index - 7) * 0.2 for index in range(8))
REPLAY_FUTURE_TIMES_S = tuple((index + 1) * 0.2 for index in range(15))
PAIRED_PANEL_ORDER = (
    "collision",
    "near_miss",
    "temporal_safe",
    "spatial_safe",
    "irrelevant_hidden",
    "empty_blind_spot",
)

@dataclass(frozen=True)
class VisualVariant:
    kind: str
    target_history: np.ndarray | None
    target_future: np.ndarray | None
    visibility_history: np.ndarray | None
    min_clearance_m: float | None
    time_to_min_clearance_s: float | None
    temporal_offset_s: float | None

@dataclass(frozen=True)
class VisualAuditBundle:
    event_id: str
    base_state: BaseState
    oracle_context: OracleContext
    trajectory: LocalTrajectory
    static_occupancy: np.ndarray
    occluders: tuple[dict[str, object], ...]
    variants: tuple[VisualVariant, ...]
    grid: GridSpec
```

Validate finite float32 pose arrays, exact 8/15 lengths, boolean history
visibility, complete panel order, target absence only for `empty_blind_spot`,
and identical static/occluder/candidate skeleton by construction.

- [ ] **Step 4: Implement visibility reconstruction**

For each history index, rasterize represented static occupancy, the accepted
occluder, and context history at that index, then call existing
`raycast_visibility()` at `base_state.robot_history[index]`. Build future frames
from the current mask only. Return explicit `observed_history` and
`oracle_replay` phase tokens.

- [ ] **Step 5: Run focused contracts GREEN**

Run the Task 1 command. Expected: contract and replay-frame tests pass.

### Task 2: Render Deterministic GIF And Paired PNG

**Files:**
- Modify: `src/evaluation/seen_occluded_visuals.py`
- Modify: `tests/test_seen_occluded_visuals.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add failing image-semantic tests**

Render into `tmp_path` and assert real image content, not just existence:

```python
def test_render_visual_artifacts_have_fixed_layout_and_nonblank_pixels(tmp_path):
    result = render_visual_artifacts(_visual_bundle(), tmp_path)
    with Image.open(result.event_replay_path) as gif:
        assert gif.n_frames == 23
        assert gif.size == (1200, 900)
        extrema = [frame.convert("RGB").getextrema() for frame in ImageSequence.Iterator(gif)]
        assert any(any(low != high for low, high in channels) for channels in extrema)
    with Image.open(result.paired_events_path) as png:
        assert png.size == (2100, 1200)
        assert any(low != high for low, high in png.convert("RGB").getextrema())
```

Assert the six panel titles appear in frozen order through returned render
metadata, and assert the empty panel reports `target_removed=true`.

- [ ] **Step 2: Run the new tests and confirm RED**

Run only the two rendering node IDs. Expected: missing rendering functions.

- [ ] **Step 3: Add optional visualization dependencies**

Add without changing core or training dependencies:

```toml
visual-audit = [
    "matplotlib==3.7.5",
    "Pillow==12.3.0",
]
```

There is no lockfile in this repository. Do not add ImageIO.

- [ ] **Step 4: Implement fixed visual styling and replay rendering**

Use Matplotlib's Agg backend and Pillow GIF writing. Render equal metric axes,
thick teal candidate path, dark represented obstacles, cool visible space,
semi-transparent red unobservable space, solid visible target history, hollow
hidden target history, dashed magenta oracle future, current robot footprint,
and an `oracle replay` title on future frames. Use fixed output dimensions and
do not let labels change axes or canvas size.

- [ ] **Step 5: Implement paired 2-by-3 rendering**

Use `PAIRED_PANEL_ORDER`, one shared axis limit computed from the common
skeleton and all variant paths, and label each panel with minimum signed
clearance. Include temporal offset for `temporal_safe`; label the empty panel
`target removed`. Return panel metadata so tests do not use OCR.

- [ ] **Step 6: Run visual tests GREEN**

Run:

```bash
.conda-envs/sop4-risk/bin/python -m pytest -q tests/test_seen_occluded_visuals.py
```

Expected: all visual tests pass.

### Task 3: Implement Scientific Audit And Stable Search

**Files:**
- Create: `src/evaluation/seen_occluded_visual_audit.py`
- Create: `tests/test_seen_occluded_visual_audit.py`

- [ ] **Step 1: Write failing scientific-check tests**

Build a complete fixture group from existing SOP05/SOP06 test factories and
assert independent facts:

```python
def test_audit_recomputes_seen_then_occluded_and_variant_labels():
    candidate = _complete_candidate()
    audit = audit_candidate(candidate)
    assert audit["history_visibility"]["vector"] == [True, True, True, False, False, False, False, False]
    assert audit["history_visibility"]["regime"] == "seen_then_occluded"
    assert audit["history_visibility"]["trailing_hidden_frames"] == 5
    assert audit["variants"]["collision"]["label_predicate_passed"] is True
    assert audit["variants"]["near_miss"]["label_predicate_passed"] is True
    assert audit["scientific_checks"]["shared_skeleton"] is True
```

Add mutation tests for label metadata drift, changed occluder skeleton,
incomplete coverage, target/represented-static overlap, wrong history regime,
and missing source snippet.

- [ ] **Step 2: Write failing stable-search tests**

Inject bounded fake pair reports into the orchestration boundary. Include one
generator deficit, one partial group, and three complete groups. Assert attempt
order, exact stable reasons, unique selected identities, and stopping before a
later candidate is assessed:

```python
assert [row.status for row in result.attempts] == [
    "generator_deficit",
    "partial_pair_group",
    "accepted",
    "accepted",
    "accepted",
]
assert result.selected_event_ids == ("event-a", "event-b", "event-c")
assert "event-after-stop" not in {row.event_id for row in result.attempts}
```

- [ ] **Step 3: Run audit tests and confirm RED**

Run:

```bash
.conda-envs/sop4-risk/bin/python -m pytest -q tests/test_seen_occluded_visual_audit.py
```

Expected: collection fails because the audit module is missing.

- [ ] **Step 4: Implement request and search contracts**

Define `SeenOccludedAuditRequest`, `SearchAttempt`, `SelectedAuditCandidate`,
`SeenOccludedSearchResult`, and `SeenOccludedAuditError`. Validate schema-3
paths, split `train`, sample count exactly 3, nonnegative seed, positive limits,
trusted 64-hex digest, and no existing output path.

Call `prepare_sop05_run()` for authenticated input loading and stable schedule.
Iterate schedule ranks serially, call the formal generator boundary, reclassify
every event with `classify_history_visibility()`, join its source snippet by ID,
derive the paired seed with `derive_seed()`, and call
`generate_paired_variants()`. Preserve exact `PairGenerationError.reason` and
missing-variant reasons. Stop immediately at three independently audited
complete groups or the frozen bound.

- [ ] **Step 5: Implement independent audit checks**

Recompute history classification, target current visibility, continuous future
emergence, signed-clearance sequences, label ranges from the trusted paired
config, target/represented-static and target/occluder intersections,
kinematics, source joins, and skeleton equality. Store the limitation token
`unmodeled_floorplan_unknown` because the THOR SOP03 input lacks an aligned
static facility map.

- [ ] **Step 6: Run audit tests GREEN**

Run the Task 3 command. Expected: all audit and bounded-search tests pass.

### Task 3A: Add Audit-Only Joint Occluder Search

**Files:**
- Create: `src/evaluation/seen_occluded_joint_search.py`
- Create: `configs/seen_occluded_joint_visual_audit.yaml`
- Modify: `src/evaluation/seen_occluded_visual_audit.py`
- Modify: `scripts/05_render_seen_occluded_visual_audit.py`
- Modify: `tests/test_seen_occluded_visual_audit.py`
- Modify: `tests/test_05_render_seen_occluded_visual_audit_cli.py`

- [ ] **Step 1: Add failing continuation and provenance tests**

Construct a real typed mother fixture and inject two physically certified joint
placements. The first produces `unseen_in_history_window`; the second produces
`seen_then_occluded` and a complete group. Assert the search evaluates both,
selects only the second, preserves the source event ID, derives a new world ID,
and records `audit_only`, source-world ID, algorithm/config digests, temporal
offset, and rejection counts.

- [ ] **Step 2: Confirm RED**

Run:

```bash
.conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_seen_occluded_visual_audit.py -k joint
```

Expected: import or missing joint-search API failure.

- [ ] **Step 3: Implement strict joint-search config**

Load an exact-key YAML contract containing the algorithm version, shared-LOS
fractions, center weights, length/width quantiles, and occluder type order.
Reject booleans, non-finite values, duplicate values, quantiles outside
`[0, 1]`, unknown types, and algorithm-version drift. Store a canonical digest
in every accepted rebound world and collection context.

- [ ] **Step 4: Implement bounded compact placement search**

For each formal temporal offset, reject synchronized collisions and spatially
disjoint paths first. Prepare every robot/context/target sweep once. Derive the
shared feasible LOS half-space from the collision and temporal current poses,
then enumerate only configured fractions, dimensions, yaws, and center weights.
Every returned placement must clear represented static occupancy and all exact
sweeps, hide both targets now, and allow both to emerge continuously.

- [ ] **Step 5: Rebind and re-run formal SOP06**

Replace only the audit world's occluder/static mask, rebuild the strict target
motion world join, recompute all history-policy metadata, and reject every
regime except `seen_then_occluded`. Run unchanged `generate_paired_variants()`
and accept only exact six-way strict coverage. Keep formal partial groups and
published SOP05 inputs untouched.

- [ ] **Step 6: Integrate bounded fallback and run GREEN**

Invoke joint search when the formal group either contains all variants except
temporal, or additionally lacks spatial-safe for the exact reason
`target_occluder_collision`. Temporal must still fail because of current
visibility or original-occluder collision. Use robot-frame LOS coordinates and
refine long-axis centers only after the unshifted center passes physical and
visibility gates. Skip accepted base-state duplicates before expensive joint
search. Run the Task 3A tests and the full paired/audit focused suite.

The 512-pair authenticated prefix produced two distinct complete base states.
The final immutable run therefore extends the same stable schedule through
rank 903 (`max_pairs=904`), where a naturally complete third group was observed.

### Task 4: Add Atomic Publication And Strict Reload

**Files:**
- Modify: `src/evaluation/seen_occluded_visual_audit.py`
- Modify: `tests/test_seen_occluded_visual_audit.py`

- [ ] **Step 1: Add failing publication tests**

Use three fixture candidates and assert exact payload paths, canonical JSONL,
sorted checksum entries, marker binding, and strict reload. Add tampering tests
for a GIF byte, attempt row, audit field, checksum entry, and marker digest.
Assert an insufficient two-sample result has a manifest and checksums but no
`.audit-complete`, returns a nonzero result, and records stable reasons.

- [ ] **Step 2: Confirm RED at publication boundary**

Run publication test node IDs. Expected: publication API missing.

- [ ] **Step 3: Implement staging and artifact writes**

Write sample directories and visual artifacts first, then canonical
`audit.json`, canonical `search_attempts.jsonl`, and
`visual_audit_manifest.json`. Build `artifact_checksums.sha256` over every
payload except itself and `.audit-complete`. Write the marker last with both
manifest and checksum-file SHA-256 values.

- [ ] **Step 4: Reload before atomic rename**

Strictly parse finite JSON, reject unknown/missing files, verify every checksum,
open every image through Pillow, check GIF frame count and dimensions, recompute
ordered sample IDs, and require marker/status consistency. Publish with
`src.utils.atomic_publish.atomic_rename_noreplace()` and clean only the staging
directory created by this invocation on failure.

- [ ] **Step 5: Run publication tests GREEN**

Run the full audit test file. Expected: all pass, including tamper and
no-overwrite cases.

### Task 5: Add CLI And Focused Integration Validation

**Files:**
- Create: `scripts/05_render_seen_occluded_visual_audit.py`
- Create: `tests/test_05_render_seen_occluded_visual_audit_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Load the script as a module, monkeypatch `run_seen_occluded_visual_audit`, and
assert all paths and numeric limits reach the request unchanged. Assert known
contract errors print one JSON object to stderr and return `2`; an insufficient
collection returns `3`; a complete collection prints its manifest digest and
returns `0`.

- [ ] **Step 2: Confirm CLI tests RED**

Run:

```bash
.conda-envs/sop4-risk/bin/python -m pytest -q tests/test_05_render_seen_occluded_visual_audit_cli.py
```

Expected: script file missing.

- [ ] **Step 3: Implement the thin CLI**

Defaults must match the design while all scientific paths remain explicit:

```text
--split train
--seed 42
--sample-count 3
--events-per-pair 1
--max-base-states 512
--trajectory-count 21
--max-pairs 512
--max-seen-mothers 512
--checksum-workers 8
--workers 8
```

Reject any sample count other than three. Catch only expected input/audit/image
errors; do not broadly suppress programmer errors.

- [ ] **Step 4: Run focused integration suite**

Run:

```bash
.conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_seen_occluded_visuals.py \
  tests/test_seen_occluded_visual_audit.py \
  tests/test_05_render_seen_occluded_visual_audit_cli.py \
  tests/test_history_visibility.py \
  tests/test_pair_variants.py \
  tests/test_sop06_pipeline.py
```

Expected: zero failures.

### Task 6: Generate And Inspect Three Real Samples

**Files:**
- Create: `outputs/sop05_seen_then_occluded_visual_audit_20260723_v1/`

- [ ] **Step 1: Run authenticated real-input preflight**

Invoke the audit module's preflight-only path against:

```text
outputs/sop03_thor_motion_snippet_v2_schema3_47b3acd_v1
outputs/sop04_canonical_trajectory_bank_9bf19d5_v2_schema3
```

Expected: schema `3.0.0`, trusted handoff digest match, 21 trajectories,
history layout 8, future layout 15, and no output files created.

- [ ] **Step 2: Run the exact real audit command**

Run:

```bash
.conda-envs/sop4-risk/bin/python scripts/05_render_seen_occluded_visual_audit.py \
  --sop03-root outputs/sop03_thor_motion_snippet_v2_schema3_47b3acd_v1 \
  --sop04-root outputs/sop04_canonical_trajectory_bank_9bf19d5_v2_schema3 \
  --sop04-handoff-digest a9a71835f9b87487b0f8051d0850d86ad9008d9348c96ba2af9ae4bc373a33ce \
  --split train \
  --base-config configs/base.yaml \
  --generator-config configs/generator_seen_occluded_visual_audit.yaml \
  --paired-config configs/paired_variants_visual_audit.yaml \
  --output-dir outputs/sop05_seen_then_occluded_visual_audit_20260723_v1 \
  --seed 42 \
  --sample-count 3 \
  --events-per-pair 1 \
  --max-base-states 512 \
  --trajectory-count 21 \
  --max-pairs 512 \
  --max-seen-mothers 512 \
  --checksum-workers 8 \
  --workers 8 \
  --git-executable /usr/bin/git
```

Expected: status `complete`, exactly three selected events, and
`.audit-complete` present. If the bounded run is insufficient, preserve its
diagnostic output and do not relabel it as success.

- [ ] **Step 3: Strictly reload and verify the collection**

Run the public strict loader against the final path. Expected: all checksums,
manifests, audit records, GIF frame counts, image dimensions, and completion
marker bindings pass.

- [ ] **Step 4: Inspect every paired PNG and representative GIF frames**

Extract, without modifying the GIFs, frames 0, 3, 7, 8, 14, and 22 into
`.tmp/agent/outputs/` for inspection. Use image viewing to inspect all three
paired PNGs and those representative frames. Record manual pass/fail notes for
current hidden state, plausible emergence, represented-geometry collision,
label separation, text overlap, and generator-pattern repetitiveness. Delete
the extracted temporary frames after review.

- [ ] **Step 5: Run final regression and workspace checks**

Run:

```bash
.conda-envs/sop4-risk/bin/python -m pytest -q \
  tests/test_seen_occluded_visuals.py \
  tests/test_seen_occluded_visual_audit.py \
  tests/test_05_render_seen_occluded_visual_audit_cli.py \
  tests/test_history_visibility.py \
  tests/test_pair_variants.py \
  tests/test_sop06_pipeline.py
git diff --check
git status --short
```

Expected: zero relevant test failures, no whitespace errors, no retained
temporary frames, and no changes outside the planned files plus pre-existing
unrelated worktree changes.
