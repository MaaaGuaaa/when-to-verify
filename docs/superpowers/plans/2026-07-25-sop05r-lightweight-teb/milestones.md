# SOP05R long-horizon lightweight TEB implementation milestones

_File-level execution plan for replacing the unpublished pre-long40 path with SOP05R v3._

> **For agentic workers:** Use `superpowers:subagent-driven-development` or
> `superpowers:executing-plans` milestone by milestone. Use TDD, stop at each gate, and
> never commit unless the user explicitly requests it.

---

## 📚 Document set

- [Long40 system contract](../../../long40_system_contract.md)
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
| M1 | immutable v3 long40 versions and config | none | strict config tests |
| M2 | rectangle/circle primitive occluder authority | M1 | analytic/raster geometry tests |
| M3 | deterministic target-blind full-route TEB | M1, M2 | planner dynamics and oracle-isolation tests |
| M4 | fixed-start goal/occluder templates | M2, M3 | deterministic template fixture |
| M5 | anchored human rotation and centerline witness | M2, M4 | placement and visibility tests |
| M6 | decision binding, 6.4-second suffix, and one-route mother | M3, M5 | continuous collision mother fixture |
| M7 | trajectory store, run producer, loader, CLI | M6 | strict round-trip and tamper tests |
| M8 | same-goal post-action TEB replanning | M7 | revealability tests with observed/hidden branches |
| M9 | explicit SOP06 v3 handoff | M7 | paired-variant and pipeline tests |
| M10 | visual/statistical gates and completion audit | M8, M9 | 10/100/1,000 evidence ladder |

## 🏷️ M1 — Freeze versions and configuration

### Objective

Create the canonical `obstacle_first_teb` long40 contract and make it the only current
production trajectory layout. Freeze every scientific and bounded-search parameter into
one canonical digest. The TEB planner returns an eight-second moving route while Schema
`4.0.0` consumes 32 future endpoints over `6.4 s`.

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
- Schema `3.0.0` fixtures used only to prove current-loader rejection
- existing `configs/base.yaml`
- existing `configs/verification_actions.yaml`

### Required implementation

- add the exact v3 version constants from [contracts.md](./contracts.md)
- add strict immutable config sections for:
  - typed rectangle, composite L-shape, and circle template families, frozen to
    `0.4/0.4/0.2` weights
  - goal bearings and distances
  - 21-node TEB band, 20 bounded interval durations, fixed iteration count,
    initialization policy, and objective weights
  - `initial_band_dt_s=0.25`, `band_dt_range_s=[0.1, 0.4]`,
    `maximum_route_time_s=8.0`, `route_sample_dt_s=0.2`, and 40 route endpoints
  - kinematic limits, `[0.15, 0.75] m` clearance range, and `0.08 m` bypass
    tracking allowance
  - goal distance range `[4.0, 4.5] m`, collision route path-fraction range
    `[0.20, 0.95]`, and decision-relative time constraints
  - finite coarse-angle schedule and seed-derived first-fit traversal
  - finite temporal scales with spatial scale fixed to `1.0`
  - centerline intersection epsilon
  - initial-visible/initially-hidden mixture
  - decision, braking, and replanning margins
  - run/publication versions and finite rejection vocabulary
- require strict key equality, finite numeric values, ordered ranges, and canonical JSON
- make `--generator-mode obstacle_first_teb` require the v3 config and
  verification-action config
- reject v1/v2 config versions and every 23-sample/15-step artifact in current modes
- advance all unpublished v2 semantic identities to the exact v3 identities in
  [contracts.md](./contracts.md)
- freeze `history_steps=8`, `current_index=7`, `future_steps=32`,
  `future_dt_s=0.2`, `future_horizon_s=6.4`, and Schema `4.0.0` in the v3 branch
- keep historical loaders outside current dispatch; do not use them for compatibility

### Outputs

- normalized immutable `Sop05rTebConfig`
- config semantic digest
- mode-specific CLI request with no generation side effect during preflight
- explicit eight-second route and 6.4-second model-horizon contract

### TDD checkpoint

1. Add tests for every new required/unknown key and semantic version.
2. Add semantic tests for interval bounds, initial interval membership, maximum route
   support, and integral 40-route/32-suffix endpoint grids.
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
- v3 occluder config nodes

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
state to the fixed goal within the eight-second planning domain.

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

- build deterministic straight, single-waypoint left-bypass, and single-waypoint
  right-bypass initialization bands; optimize all three and select by frozen task cost
- optimize 21-pose internal bands and 20 bounded interval durations with fixed-iteration
  NumPy updates
- include path-length, time, smoothness, obstacle hinge, nonholonomic, velocity,
  acceleration, goal-heading, and initial-control terms
- hold start and goal fixed on every iteration
- project or reject invalid velocity, acceleration, angular-rate, curvature, and time steps
- perform deterministic dense sampled static-collision validation after optimization;
  the frozen clearance margin is the conservative safety mechanism, not analytic
  continuous-time collision solving
- resample the accepted continuous route at `[0.2, ..., 8.0] s` into 40 endpoint poses
  and controls
- reach the exact goal within the configured pose and heading tolerances; a separately
  verified stationary hold after arrival is not an M3 acceptance requirement
- build one immutable `PlannedTebRoute`; do not build `LocalTrajectory` or query maps
- choose one lowest-cost valid route with stable tie-breaking
- return a finite diagnostic object when no route succeeds

### Outputs

- `LightweightTebResult`
- zero or one public `PlannedTebRoute`
- 21 band poses, 20 interval durations, and 40 uniformly sampled endpoint states
- per-initialization cost and rejection diagnostics

### TDD checkpoint

1. Assert request fields contain no target/oracle/collision/label channel.
2. Cover unobstructed, rectangle bypass, composite L-shape bypass, circle bypass,
   infeasible, and symmetric cases.
3. Assert exact implicit start, fixed goal, bounded interval durations, and a finite,
   deterministic `goal_arrival_time_s` for later M6 binding.
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
- Treat `src/generation/obstacle_first_templates.py` as archival v1 code, outside current dispatch

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
  `minimum_clearance - direct_line_clearance_m` to be at least the frozen
  `0.15 m` minimum
- require at least one static primitive to analytically intersect the direct start-goal
  centerline, then retain the largest nonzero lateral offset satisfying both conditions
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

Map one of the 32 future samples from a long40 human trajectory to one full-route
collision point, rotate the complete 8-history-plus-32-future trajectory around that
fixed point, and find a valid centerline-occlusion witness.

### Files

- Create `src/generation/anchored_human_placement.py`
- Create `tests/test_anchored_human_placement.py`
- Modify `src/generation/history_visibility.py` only to add a separate v3 assessment
- Modify `tests/test_sop05r_history_visibility.py`

### Inputs

- M4 task template and nominal full route
- split-local `history8_current7_future32_v1` human trajectories
- source context motion for collision rejection
- route-anchor ranges
- rotation and temporal-scale schedule
- typed occluders and centerline epsilon

### Required implementation

- shortlist route samples by path fraction and support through
  `t_decision + 6.4 s`; do not use the old absolute encounter-time range
- derive each human anchor from the route endpoint's shared Long40 index `8..39`;
  all 32 future samples, including index `39`, enter collision search, after which the
  `1.2 s` decision-margin gate rejects too-early first collisions
- compute translation from the anchor equality
- vectorize all coarse rigid rotations as `[angle, sample, xy]`
- keep `spatial_scale == 1.0`
- rotate headings and velocity vectors consistently
- reject bounds, static, occluder, context, speed, and acceleration failures before visibility
- compute synchronized centerline intersection for all candidate times
- accept one blocked witness for the primary relaxed occlusion rule
- preserve a distinct initially-hidden stratum when requested
- enumerate route anchors, temporal scales, and coarse angles in a seed-derived finite
  order; accept the first candidate that passes every frozen gate, without ranking,
  local refinement, or continued search
- return complete candidate and rejection evidence

### Outputs

- `AnchoredHumanPlacement`
- `CenterlineOcclusionWitness`
- eight-frame visibility evidence with at least four visible samples
- a 40-sample transformed target trajectory with current index `7` and 32 future samples
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
- Modify `src/planning/query_maps.py` to emit the canonical 32-step Schema `4.0.0` maps;
  current production behavior must not retain a 15-step branch

### Inputs

- M4 task template and full nominal `PlannedTebRoute`
- M5 anchored placement and witness
- source BaseState and OracleContext
- base and v3 generator configs
- seed

### Required implementation

- choose a hidden witness whose eight-frame history contains at least four visible frames
- leave the frozen `1.2 s` verification-action plus braking margin before collision;
  replanning completion is not an acceptance requirement
- bind human snippet index `7` to the decision witness
- derive the decision pose/control and observed history by joining source `BaseState`
  history to the M4 route prefix
- rebase robot, target, source context, static occupancy, and occluders into the
  decision-local frame and emit the concrete decision `BaseState`
- retain source-to-decision route-prefix provenance and source base identity
- sample exactly 32 future endpoints at `0.2 s` after the decision witness
- require a pre-goal collision anchor but do not apply a goal-arrival suffix horizon gate
  padding
- transform the suffix into the decision frame and build Schema `4.0.0` query maps
- bind the nominal `LocalTrajectory` suffix to the exact same world-frame goal and
  full-route task cost
- require `1.2 s <= t_collision - t_decision <= 6.4 s`
- accept collision in the final `6.2–6.4 s` future interval when continuous
  swept-footprint interpolation proves it; reject discrete-only endpoint equality
- compute continuous robot-human collision and reject discrete-only/endpoint-only contact
- reject target-static, target-occluder, target-context, and premature robot contact
- remove the v1 same-goal-alternative gate from this v3 path
- construct `GeneratedEvent`, `OracleWorld`, target-motion record, and one-plan
  dual-timeline trajectory record with complete v3 metadata
- bind event/world identities to versions, goal, occluders, route, anchor, rotation,
  witness, collision, source identity, split, config digest, and seed

### Outputs

- `Sop05rTebMotherCandidate`
- one verification-ready `GeneratedEvent`
- one `Sop05rTebTrajectoryRecord` containing the full route and 6.4-second suffix
- continuous collision and decision-state evidence

### TDD checkpoint

1. Assert planner execution precedes all target operations.
2. Assert decision time equals a persisted blocked witness.
3. Assert the full route reaches the same goal and has continuous collision after
   decision.
4. Assert the suffix has exactly 32 endpoints, exact Schema `4.0.0` query maps, and
   contains the first collision; include a regression for the final `6.2–6.4 s`
   future interval.
5. Assert no alternative route is required or serialized.
6. Assert deterministic event/world IDs and complete provenance.

## 📦 M7 — Publish, reload, and dispatch v3 collections

### Objective

Publish v3 mother events and single-plan trajectory records atomically with strict
independent identity and no v1/v2 ambiguity.

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
- v3 configs and verification-action snapshot
- run quota, split, seed, worker count, and output directory

### Required implementation

- use deterministic process-safe schedule ordering
- select requested visibility strata and training active-revealable quota without hiding
  deficits
- publish canonical JSON and deterministic NPZ without pickle
- persist every M4–M6 denominator and stable rejection reason
- persist the 21-pose variable-time band, eight-second/40-endpoint uniform full route,
  and 6.4-second/32-endpoint decision-relative suffix as distinct authenticated arrays
- bind array dtype, shape, bytes, semantic digests, and outer checksums
- strict self-reload before atomic no-overwrite rename
- write a completion marker only when every requested quota is met
- reject v1 and pre-long40 v2 stores/manifests in the v3 loader
- prevent archival CLI modes from writing or joining current long40 collections

### Outputs

- versioned v3 output directory
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
same eight-second full-route TEB engine with only deployment-observed dynamic information.

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
- keep model/risk evaluation on the original decision-relative 6.4-second window; an
  action consumes part of this window and does not create target motion beyond sample 39
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

Allow SOP06 consumers to use v3 long40 single-route events and reject every v1,
pre-long40 v2, or 15-step handoff in the current production path.

### Files

- Modify `src/generation/paired_variants.py`
- Modify `src/generation/sop06_pipeline.py`
- Modify `tests/test_pair_variants.py`
- Modify `tests/test_sop06_pipeline.py`

### Inputs

- strict M7 v3 loader output
- v3 nominal trajectory store
- paired-variant config

### Required implementation

- dispatch on exact v3 generator, schema, layout, and collection versions
- require exactly one nominal plan and exact goal binding
- require both the authenticated 40-endpoint full route and its decision-relative
  32-endpoint nominal suffix
- preserve source/decision identities and target motion provenance
- create target-present and target-removed variants without changing static geometry,
  route, goal, source context, or observed prefix
- reject requests for v1 alternative route IDs in the v3 branch
- keep archival branches outside current dispatch and fail closed on mixed artifacts

### Outputs

- authenticated paired variants and SOP06 reports for v3 events
- explicit compatibility errors for mixed artifacts

### TDD checkpoint

1. Cover one-route v3 pair construction and target-only removal.
2. Cover v1/v2/v3 mixed-store rejection.
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
- render fixed-scale layers for source start, goal, shape, eight-second full route,
  6.4-second decision suffix, 40-sample human trajectory, anchor, occlusion witness,
  decision point, action traces, and first visibility
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
