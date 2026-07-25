# SOP05R lightweight TEB implementation milestones

_File-level execution plan for replacing the unapproved SOP05R v1 generator with the v2 lightweight-TEB path._

> **For agentic workers:** Use `superpowers:subagent-driven-development` or
> `superpowers:executing-plans` milestone by milestone. Use TDD, stop at each gate, and
> never commit unless the user explicitly requests it.

---

## 📚 Document set

- [Full specification](./full-spec.md)
- [Contracts](./contracts.md)
- [Milestones](./milestones.md)
- [Acceptance](./acceptance.md)
- [State](./state.md)

## 🗺️ Dependency map

```mermaid
flowchart LR
    accTitle: SOP05R TEB Milestone Dependencies
    accDescr: Ordered implementation dependencies from immutable contracts through geometry, planning, placement, event publication, verification integration, and release audits

    m1[📋 M1 Contracts]
    m2[⚙️ M2 Occluder geometry]
    m3[🔧 M3 TEB planner]
    m4[⚙️ M4 Task templates]
    m5[🔄 M5 Human placement]
    m6[🎯 M6 Decision and mother]
    m7[📦 M7 Publication]
    m8[🔍 M8 Verification replan]
    m9[🔗 M9 SOP06 handoff]
    m10[🧪 M10 Release gates]

    m1 --> m2
    m1 --> m3
    m2 --> m3
    m2 --> m4
    m3 --> m4
    m4 --> m5
    m5 --> m6
    m6 --> m7
    m7 --> m8
    m7 --> m9
    m8 --> m10
    m9 --> m10

    classDef contract fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#3b0764
    classDef process fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef success fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class m1 contract
    class m2,m3,m4,m5,m6,m7,m8,m9 process
    class m10 success
```

## 📋 Milestone summary

| Milestone | Deliverable | Depends on | Completion evidence |
| --- | --- | --- | --- |
| M1 | immutable v2 versions and config | none | strict config tests |
| M2 | rectangle/circle primitive occluder authority | M1 | analytic/raster geometry tests |
| M3 | deterministic target-blind full-route TEB | M1, M2 | planner dynamics and oracle-isolation tests |
| M4 | fixed-start goal/occluder templates | M2, M3 | deterministic template fixture |
| M5 | anchored human rotation and centerline witness | M2, M4 | placement and visibility tests |
| M6 | decision binding, 3-second suffix, and one-route mother | M3, M5 | continuous collision mother fixture |
| M7 | trajectory store, run producer, loader, CLI | M6 | strict round-trip and tamper tests |
| M8 | same-goal post-action TEB replanning | M7 | revealability tests with observed/hidden branches |
| M9 | explicit SOP06 v2 handoff | M7 | paired-variant and pipeline tests |
| M10 | visual/statistical gates and completion audit | M8, M9 | 10/100/1,000 evidence ladder |

## 🏷️ M1 — Freeze versions and configuration

### Objective

Create an explicit `obstacle_first_teb` contract without changing `legacy` or current
`obstacle_first` dispatch. Freeze every scientific and bounded-search parameter into one
canonical digest. Before M3, apply the pre-publication dual-horizon correction: the TEB
planner returns a bounded full route while Schema `3.0.0` retains its three-second model
horizon.

### Files

- Create `configs/generator_obstacle_first_teb_train.yaml`
- Create `configs/generator_obstacle_first_teb_test.yaml`
- Modify `src/generation/sop05r_contracts.py`
- Modify `scripts/05_generate_events.py`
- Modify `tests/test_sop05r_contracts.py`
- Create `tests/test_sop05r_teb_contracts.py`
- Modify `tests/test_05_generate_events_cli.py`

### Inputs

- [contracts.md](./contracts.md)
- existing schema `3.0.0`
- existing `configs/base.yaml`
- existing `configs/verification_actions.yaml`

### Required implementation

- add the exact v2 version constants from [contracts.md](./contracts.md)
- add strict immutable config sections for:
  - typed rectangle, composite L-shape, and circle template families, frozen to
    `0.4/0.4/0.2` weights
  - goal bearings and distances
  - 20-node TEB band, 19 bounded interval durations, fixed iteration count,
    initialization policy, and objective weights
  - `initial_band_dt_s=0.25`, `band_dt_range_s=[0.1, 0.4]`,
    `maximum_route_time_s=5.0`, and `route_sample_dt_s=0.2`
  - kinematic limits, `[0.15, 0.75] m` clearance range, and `0.08 m` bypass
    tracking allowance
  - anchor path/time ranges
  - coarse/refined rotation schedule
  - finite temporal scales with spatial scale fixed to `1.0`
  - centerline intersection epsilon
  - initial-visible/initially-hidden mixture
  - decision, braking, and replanning margins
  - run/publication versions and finite rejection vocabulary
- require strict key equality, finite numeric values, ordered ranges, and canonical JSON
- make `--generator-mode obstacle_first_teb` require the v2 config and verification-action config
- reject v1 config versions in the new mode and v2 versions in old modes
- advance the pre-publication planner component identity to `lightweight_teb_planner_v2`
- leave `configs/base.yaml`, `future_steps=15`, `future_dt_s=0.2`, and Schema `3.0.0`
  unchanged

### Outputs

- normalized immutable `Sop05rTebConfig`
- config semantic digest
- mode-specific CLI request with no generation side effect during preflight
- explicit five-second route and three-second model-horizon contract

### TDD checkpoint

1. Add tests for every new required/unknown key and semantic version.
2. Add semantic tests for interval bounds, initial interval membership, maximum route
   support, and an integral 25-endpoint route grid.
3. Observe focused RED failures.
4. Implement strict normalization.
5. Run M1 commands from [acceptance.md](./acceptance.md).
6. Do not proceed until old mode argument tests still pass.

## 🧱 M2 — Add typed occluder geometry

### Objective

Provide one analytic and raster authority for rectangular walls/shelves and circular tree
trunks/columns.

### Files

- Create `src/geometry/static_occluders.py`
- Modify `src/geometry/__init__.py`
- Create `tests/test_static_occluders.py`
- Modify `tests/test_rasterization.py`
- Modify `tests/test_collision.py`

### Inputs

- `GridSpec`
- existing `RectangleFootprint` and `CircleFootprint`
- v2 occluder config nodes

### Required implementation

- implement immutable `RectangleOccluder`, `CircleOccluder`, and `StaticOccluder`;
  L shapes remain template-level composites of two rectangles rather than a third
  analytic primitive
- implement exact finite validation and stable JSON payloads
- implement vectorized bounds, inflation, point signed distance, rasterization, and segment intersection
- use obstacle-local slab intersection for oriented rectangles
- use closest-point segment distance for circles
- count tangency within configured epsilon as intersection
- keep analytic centerline intersection independent from footprint raycasting

### Outputs

- reusable typed occluder objects
- analytic intersection vectors for `[N, 2]` segment endpoints
- binary occupancy masks compatible with existing grid authorities

### TDD checkpoint

1. Cover crossing, tangency, miss, degenerate segment, rotation, batching, and invalid values.
2. Compare analytic intersections against hand-computed fixtures.
3. Verify raster masks are nonempty, bounded, and deterministic.
4. Run M2 commands before M3.

## 🔧 M3 — Implement the lightweight TEB planner

### Objective

Return one deterministic, target-blind, dynamically feasible route from the fixed source
state to the fixed goal within the five-second planning domain.

### Files

- Create `src/planning/lightweight_teb.py`
- Modify `src/planning/__init__.py`
- Reuse without semantic changes:
  - `src/planning/differential_drive.py`
- Create `tests/test_lightweight_teb.py`
- Modify `tests/test_replanning.py`
- Modify `tests/test_input_oracle_isolation.py`

### Inputs

- `StaticTebRequest`
- typed occluders from M2
- binary static occupancy
- frozen base and planner configs

### Required implementation

- build one deterministic straight initialization band; use the nonzero occluder lateral
  offset to provide the unique obstacle-repulsion direction
- optimize 20-pose internal bands and 19 bounded interval durations with fixed-iteration
  NumPy updates
- include path-length, time, smoothness, obstacle hinge, nonholonomic, velocity,
  acceleration, goal-heading, and initial-control terms
- hold start and goal fixed on every iteration
- project or reject invalid velocity, acceleration, angular-rate, curvature, and time steps
- perform deterministic dense sampled static-collision validation after optimization;
  the frozen clearance margin is the conservative safety mechanism, not analytic
  continuous-time collision solving
- resample the accepted continuous route at `[0.2, ..., 5.0] s` into 25 endpoint poses
  and controls
- reach the exact goal within the configured pose and heading tolerances; a separately
  verified stationary hold after arrival is not an M3 acceptance requirement
- build one immutable `PlannedTebRoute`; do not build `LocalTrajectory` or query maps
- choose one lowest-cost valid route with stable tie-breaking
- return a finite diagnostic object when no route succeeds

### Outputs

- `LightweightTebResult`
- zero or one public `PlannedTebRoute`
- 20 band poses, 19 interval durations, and 25 uniformly sampled endpoint states
- per-initialization cost and rejection diagnostics

### TDD checkpoint

1. Assert request fields contain no target/oracle/collision/label channel.
2. Cover unobstructed, rectangle bypass, composite L-shape bypass, circle bypass,
   infeasible, and symmetric cases.
3. Assert exact implicit start, fixed goal, bounded interval durations, and deterministic
   goal hold.
4. Assert initial-control continuity, dynamics, represented clearance, and continuous
   static-collision freedom on the full route.
5. Assert repeated calls are byte-identical.
6. Run M3 commands before templates use the planner.

## ⚙️ M4 — Generate fixed-start goal/occluder templates

### Objective

Generate high-yield static planning tasks without placing a provisional human target.

### Files

- Create `src/generation/sop05r_teb_templates.py`
- Create `tests/test_sop05r_teb_templates.py`
- Leave `src/generation/obstacle_first_templates.py` as the v1 implementation

### Inputs

- immutable source `BaseState`
- source `OracleContext` for protected-overlap checks only
- typed occluder templates
- goal template schedule
- M3 planner
- deterministic seed

### Required implementation

- sample goal bearing and distance relative to `base_state.robot_history[-1]`
- sample a nonzero left/right lateral offset for rectangle/circle occluders near the
  direct start-goal segment; never use a centered placement as the default fixture
- sample a deterministic signed rectangle/L-shape orientation relative to the
  start-goal direction from the frozen `±[15°, 45°]` range, then recompute lateral
  offset so rotation preserves the direct-corridor intrusion requirement
- require the straight-driving safety-corridor intrusion
  `minimum_clearance - direct_line_clearance_m` to lie in the frozen
  `[0.05, 0.15] m` range
- reject source-static, robot-history, protected-context, and bounds overlaps
- invoke M3 without snippets or target data
- retain only templates with one valid route in the configured clearance and length ranges
- cache planner results by source state, task geometry, initial control, and config digest
- preserve source BaseState and OracleContext digests

### Outputs

- `Sop05rTebTaskTemplate`
- one static occluder set
- one fixed goal
- one target-blind nominal full route
- geometry and planner denominator evidence

### TDD checkpoint

1. Prove no MotionSnippet is read before the planner returns.
2. Cover every rectangle/L-shape/circle template, stable seed ordering, and frozen
   nonparallel orientation bounds for rectangles and L shapes.
3. Assert direct-line obstruction and accepted-route clearance.
4. Assert source objects are unchanged.
5. Run M4 commands before target placement.

## 🔄 M5 — Solve anchored human placement and occlusion

### Objective

Map one internal human snippet sample to one full-route collision point, rotate the full
trajectory around that fixed point, and find a valid centerline-occlusion witness.

### Files

- Create `src/generation/anchored_human_placement.py`
- Create `tests/test_anchored_human_placement.py`
- Modify `src/generation/history_visibility.py` only to add a separate v2 assessment
- Modify `tests/test_sop05r_history_visibility.py`

### Inputs

- M4 task template and nominal full route
- split-local human MotionSnippets
- source context motion for collision rejection
- route-anchor ranges
- rotation and temporal-scale schedule
- typed occluders and centerline epsilon

### Required implementation

- shortlist internal route samples by collision time, path fraction, and remaining margin
- shortlist internal snippet anchors by time support, displacement, speed, and acceleration
- compute translation from the anchor equality
- vectorize all coarse rigid rotations as `[angle, sample, xy]`
- keep `spatial_scale == 1.0`
- rotate headings and velocity vectors consistently
- reject bounds, static, occluder, context, speed, and acceleration failures before visibility
- compute synchronized centerline intersection for all candidate times
- accept one blocked witness for the primary relaxed occlusion rule
- preserve a distinct initially-hidden stratum when requested
- refine the best configured number of coarse angles and select by stable visibility,
  clearance, crossing-angle, and minimal temporal-transform score
- return complete candidate and rejection evidence

### Outputs

- `AnchoredHumanPlacement`
- `CenterlineOcclusionWitness`
- start-visibility evidence
- requested/observed visibility stratum
- bounded-search counters

### TDD checkpoint

1. Assert the anchor is unchanged for every rotation.
2. Assert rigid pairwise distances, speed magnitudes, and accelerations are preserved.
3. Cover initial visible plus one later blocked sample for rectangle and circle fixtures.
4. Prove arbitrary cross-time point pairs do not satisfy the synchronized predicate.
5. Run M5 commands before event construction.

## 🎯 M6 — Bind the decision state and construct one-route mothers

### Objective

Turn the full encounter into a verification-ready mother event whose decision time is a
valid hidden witness and whose single nominal plan collides continuously with the human.

### Files

- Create `src/generation/sop05r_teb_event_sampler.py`
- Create `src/generation/sop05r_teb_decision_state.py`
- Create `tests/test_sop05r_teb_event_sampler.py`
- Create `tests/test_sop05r_teb_decision_state.py`
- Reuse collision authority from `src/generation/sop05r_event_sampler.py` only after its
  behavior is covered by regression tests
- Reuse `src/planning/query_maps.py` without changing its three-second semantics

### Inputs

- M4 task template and full nominal `PlannedTebRoute`
- M5 anchored placement and witness
- source BaseState and OracleContext
- base and v2 generator configs
- seed

### Required implementation

- choose a witness that leaves action, braking, and replanning margin
- derive the decision pose/control and observed history ending at that witness
- retain source-to-decision route-prefix provenance and source base identity
- sample exactly 15 future endpoints at `0.2 s` after the decision witness
- deterministically hold the goal when a suffix sample is after full-route arrival
- transform the suffix into the decision frame and build the existing query maps
- bind the nominal `LocalTrajectory` suffix to the exact same world-frame goal and
  full-route task cost
- require the continuous first collision to occur inside that three-second suffix
- compute continuous robot-human collision and reject discrete-only/endpoint-only contact
- reject target-static, target-occluder, target-context, and premature robot contact
- remove the v1 same-goal-alternative gate from this v2 path
- construct `GeneratedEvent`, `OracleWorld`, target-motion record, and one-plan
  two-horizon trajectory record with complete v2 metadata
- bind event/world identities to versions, goal, occluders, route, anchor, rotation,
  witness, collision, source identity, split, config digest, and seed

### Outputs

- `Sop05rTebMotherCandidate`
- one verification-ready `GeneratedEvent`
- one `Sop05rTebTrajectoryRecord` containing the full route and three-second suffix
- continuous collision and decision-state evidence

### TDD checkpoint

1. Assert planner execution precedes all target operations.
2. Assert decision time equals a persisted blocked witness.
3. Assert the full route reaches the same goal and has continuous collision after
   decision.
4. Assert the suffix has exactly 15 endpoints, exact query maps, and contains the first
   collision; do not require the suffix itself to reach the goal.
5. Assert no alternative route is required or serialized.
6. Assert deterministic event/world IDs and complete provenance.

## 📦 M7 — Publish, reload, and dispatch v2 collections

### Objective

Publish v2 mother events and single-plan trajectory records atomically with strict
independent identity and no v1 ambiguity.

### Files

- Create `src/generation/sop05r_teb_trajectory_store.py`
- Create `src/generation/sop05r_teb_run.py`
- Create `src/generation/sop05r_teb_output_loader.py`
- Modify `scripts/05_generate_events.py`
- Create `tests/test_sop05r_teb_trajectory_store.py`
- Create `tests/test_sop05r_teb_run.py`
- Create `tests/test_sop05r_teb_output_loader.py`
- Modify `tests/test_05_generate_events_cli.py`

### Inputs

- M6 accepted candidates
- authenticated SOP03 source evidence
- v2 configs and verification-action snapshot
- run quota, split, seed, worker count, and output directory

### Required implementation

- use deterministic process-safe schedule ordering
- select requested visibility strata and training active-revealable quota without hiding
  deficits
- publish canonical JSON and deterministic NPZ without pickle
- persist every M4–M6 denominator and stable rejection reason
- persist the variable-time band, five-second uniform full route, and three-second
  decision-relative suffix as distinct authenticated arrays
- bind array dtype, shape, bytes, semantic digests, and outer checksums
- strict self-reload before atomic no-overwrite rename
- write a completion marker only when every requested quota is met
- reject v1 stores/manifests in the v2 loader
- preserve old CLI modes byte-for-byte outside explicit dispatch

### Outputs

- versioned v2 output directory
- manifest, generation summary, event rows, worlds, target motion, full routes, nominal
  suffix trajectories, checksums, and conditional completion marker

### TDD checkpoint

1. Cover empty diagnostics, partial quota, complete quota, overwrite refusal, and tamper.
2. Compare single-worker and multi-worker semantic digests.
3. Strictly reload every fixture output.
4. Run M7 commands before downstream integration.

## 🔍 M8 — Reuse TEB for verification actions

### Objective

Evaluate moving actions and matched waits from the hidden decision state, then call the
same five-second full-route TEB engine with only deployment-observed dynamic information.

### Files

- Create `src/generation/sop05r_teb_revealability.py`
- Modify `src/planning/verification_responses.py`
- Modify `src/generation/counterfactual_verify.py`
- Create `tests/test_sop05r_teb_revealability.py`
- Modify `tests/test_verification_policy.py`
- Modify `tests/test_counterfactual_verify.py`

### Inputs

- M6 mother event and decision state
- verification action library
- same goal and static scene
- observation result at each action terminal state
- hidden world for label scoring only

### Required implementation

- sample each action and matched-duration wait from the same decision state
- enforce complete static and dynamic action-trace feasibility
- compute first target-visible time using the frozen sensor contract
- use `ObservedTebRequest` only when the target has actually been revealed
- call `StaticTebRequest` semantics when it remains hidden
- retain the exact same goal, route horizon, and task-cost normalization
- keep model/risk evaluation on the unchanged three-second post-action window
- compute realized hidden-world loss only after route generation
- keep stop out of active moving-action counts
- persist per-action visibility, replanning, feasibility, and value evidence

### Outputs

- per-action revealability and replanning evidence
- active moving-action IDs
- verification-value inputs with no oracle leakage

### TDD checkpoint

1. Prove hidden target data cannot enter post-action planner requests.
2. Prove revealed observation changes only the observed-dynamic branch.
3. Cover moving action earlier than matched wait and a natural-difficult event.
4. Assert same goal and identical task-cost authority before/after action.

## 🔗 M9 — Integrate explicit SOP06 handoff

### Objective

Allow SOP06 consumers to use v2 single-route events without changing legacy or v1
pair semantics.

### Files

- Modify `src/generation/paired_variants.py`
- Modify `src/generation/sop06_pipeline.py`
- Modify `tests/test_pair_variants.py`
- Modify `tests/test_sop06_pipeline.py`

### Inputs

- strict M7 v2 loader output
- v2 nominal trajectory store
- paired-variant config

### Required implementation

- dispatch on exact v2 generator and collection versions
- require exactly one nominal plan and exact goal binding
- require both the authenticated full route and its decision-relative nominal suffix
- preserve source/decision identities and target motion provenance
- create target-present and target-removed variants without changing static geometry,
  route, goal, source context, or observed prefix
- reject requests for v1 alternative route IDs in the v2 branch
- keep legacy and v1 branches unchanged

### Outputs

- authenticated paired variants and SOP06 reports for v2 events
- explicit compatibility errors for mixed artifacts

### TDD checkpoint

1. Cover one-route v2 pair construction and target-only removal.
2. Cover v1/v2 mixed-store rejection.
3. Run all paired and SOP06 focused tests.

## 🧪 M10 — Add audits and execute release gates

### Objective

Produce human-readable and machine-readable evidence for geometry, placement, occlusion,
collision, revealability, efficiency, and reproducibility.

### Files

- Create `src/evaluation/sop05r_teb_audit.py`
- Create `src/evaluation/sop05r_teb_visuals.py`
- Modify `scripts/05_render_sop05r_audit.py`
- Create `tests/test_sop05r_teb_audit.py`
- Create `tests/test_sop05r_teb_visuals.py`
- Create `tests/test_sop05r_teb_release_gate.py`
- Modify `tests/test_05_render_sop05r_audit_cli.py`
- Generated evidence only under `.tmp/agent/outputs/` for fixture gates and versioned
  `outputs/` directories for user-approved real audits

### Inputs

- strict M7 collection
- M8 revealability evidence
- M9 pair evidence when the handoff gate is included

### Required implementation

- recompute every denominator rather than trusting summary values
- render fixed-scale layers for source start, goal, shape, full route, three-second
  decision suffix, human trajectory, anchor, occlusion witness, decision point, action
  traces, and first visibility
- compare centerline visibility with footprint raycast as an audit-only disagreement metric
- bind source run IDs, checksums, config digest, and sample IDs
- refuse audit completion when source collection is incomplete or tampered
- execute the exact 10/100/1,000 ladder in [acceptance.md](./acceptance.md)

### Outputs

- deterministic audit metrics
- deterministic visual bundle
- release-gate report mapping each contract to code, test, or artifact evidence

### TDD checkpoint

1. Pass deterministic toy audit tests.
2. Pass the 10-template fixture gate.
3. Obtain user approval before the real 100-event visual audit.
4. Obtain user approval before the real 1,000-event statistical smoke.
5. Do not approve target-scale generation until every numeric and integrity gate passes.
