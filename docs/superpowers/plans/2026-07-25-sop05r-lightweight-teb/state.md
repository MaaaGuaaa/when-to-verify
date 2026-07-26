# SOP05R long-horizon lightweight TEB implementation state

_Current status, frozen decisions, evidence, risks, and next action as of 2026-07-26._

---

> ⚠️ **Historical status snapshot:** This file records the documentation-only boundary
> before the later M1–M6 Long40 implementation work. For the current operational handoff,
> actual test results, artifacts, and remaining M7–M9 gaps, read
> [HANDOFF.md](./HANDOFF.md). The shared time authority remains
> [Long40 system contract](../../../long40_system_contract.md).

## 📚 Document set

- [Current operational handoff](./HANDOFF.md)
- [Long40 system contract](../../../long40_system_contract.md)
- [Full specification](./full-spec.md)
- [Contracts](./contracts.md)
- [Milestones](./milestones.md)
- [Acceptance](./acceptance.md)
- [State](./state.md)

This is the entry point for resuming work. Update it only after obtaining concrete code,
test, or artifact evidence.

## 📍 Current status

| Area | Status | Evidence |
| --- | --- | --- |
| cross-SOP long40 time contract | documentation frozen | `docs/long40_system_contract.md` |
| v3 long40 method decisions | documentation frozen | B2 contract correction in this document set |
| implementation conformance | not assessed by this documentation task | no implementation-complete claim |
| pre-long40 code/artifacts | archival only | cannot feed current production collections |
| focused tests | not run for this documentation-only change | no new test evidence claimed |
| historical v1 real smoke | failed scientific gate | 12-base diagnostic below |
| v3 10-template gate | not run | depends on long40 M1–M6 migration |
| v3 100-event audit | not run | requires user approval after fixture gate |
| v3 1,000-event smoke | not run | requires user approval after visual gate |

No implementation-completion claim is valid at this state. Earlier 15-step tests do not
satisfy the long40 contracts, and this documentation update does not modify code.

## 🔍 Existing v1 implementation

The interrupted implementation created or modified the current SOP05R v1 chain:

- `src/generation/sop05r_contracts.py`
- `src/generation/obstacle_first_templates.py`
- `src/planning/obstacle_corner_planner.py`
- `src/generation/sop05r_event_sampler.py`
- `src/generation/sop05r_trajectory_store.py`
- `src/generation/sop05r_revealability.py`
- `src/generation/sop05r_run.py`
- `src/generation/sop05r_output_loader.py`
- SOP06, audit, CLI, and focused tests

Its current causal flow is:

```text
place provisional target and rectangle
→ plan left/right corner routes
→ realign target after planning
→ require strict history classification
→ require nominal collision plus same-goal non-collision alternative
→ audit active revealability
```

That flow is not approved for target-scale generation.

## 📊 Diagnostic evidence

The latest bounded diagnostic evaluated 12 real train base states with 32 templates each:

```text
base states                                      12
templates                                       384
accepted collision mothers                        2
active-revealable mothers                         0
best moving-action visibility lead             -0.025 s
accepted mothers with an avoiding post-action route 0
```

Terminal template rejection counts were:

```text
target_obstacle_collision                       234
context_overlap                                  76
history_ineligible                               55
target_context_collision                         12
target_acceleration_limit                         4
conflict_path_fraction_out_of_range               1
```

The evidence is stored in the temporary diagnostic:

```text
.tmp/agent/outputs/20260725-1252-sop05r-revealability-debug.json
```

It may be deleted after the v3 implementation has permanent regression coverage. It is
not a formal artifact and must not become an implementation dependency.

## ✅ Frozen decisions

The following decisions came from the user discussion and must not be silently reversed:

1. Keep the source robot start fixed; do not independently reposition the BaseState.
2. Choose a suitable local goal and place static occluders near the direct start-goal line.
3. Support rectangular wall-like obstacles and circular tree-trunk/column obstacles.
4. Replace the current corner planner with a pure-NumPy lightweight TEB-style planner.
5. Constrain initial velocity continuity, route length/time, dynamics, smoothness, and
   obstacle clearance.
6. The initial planner is target-blind and sees static geometry only.
7. Publish one initial nominal route; a precomputed non-collision alternative is not
   required.
8. Select an internal robot collision point/time and an internal human snippet anchor.
9. Translate the human anchor to the collision point, then rotate the complete human
   trajectory around that fixed point.
10. Keep primary spatial scale exactly `1.0`; use only frozen bounded temporal scaling
    when anchor times otherwise cannot align.
11. Define fast visibility using synchronized robot-human centerline intersection with
    rectangle/circle occluders.
12. The primary eight-frame decision history contains at least four centerline-visible
    samples; the first frame and transition index are not fixed.
13. The decision frame itself is a synchronized blocked sample; consecutive hidden
    frames are not required.
14. Small centerline-versus-footprint visibility disagreement is acceptable but must be
    measured.
15. A verification decision must use a blocked witness with the frozen `1.2 s`
    verification-action plus braking margin. Replanning completion is not required
    inside that margin.
16. Verification actions re-use the same lightweight TEB and the same world-frame goal.
17. Post-action replanning may use a target only when that target has become
    deployment-observable; Oracle future remains forbidden.
18. Active revealability must arise from geometry and executable actions, not relaxed
    label thresholds.
19. Use the B2 dual timeline: a bounded eight-second full planner route for reachability,
    collision anchoring, and provenance, and a decision-relative 6.4-second,
    32-step model/data/label horizon.
20. M3 returns `PlannedTebRoute`; M6 alone derives the decision-relative
    `LocalTrajectory` and query maps.
21. Goal arrival does not gate the 6.4-second decision suffix; M6 samples the same frozen
    route throughout that suffix.
22. The collision authority scans the complete 32-step future rather than truncating at
    three seconds. Accepted first collisions satisfy
    `1.2 s <= t_collision - t_decision <= 6.4 s`; the final `6.2–6.4 s` interval is
    eligible when continuous swept-footprint interpolation proves contact.
23. Static occluders use a nonzero left/right lateral offset and only shallowly intrude
    into the straight-driving safety corridor. The frozen direct-corridor intrusion range
    is `[0.05, 0.15] m`; a centerline-centered obstruction is not the default template.
24. M3 uses exactly one `straight` initialization. Its escape direction is determined by
    the nonzero occluder offset; an infeasible template is rejected and resampled.
25. M3 must evaluate and publish all frozen objective terms. Dense sampled collision
    checks plus frozen clearance margins are sufficient; analytic continuous-time
    collision solving is not required.
26. The accepted-route represented-obstacle clearance is `0.15 m`; the earlier
    `0.20 m` hand-tuned bypass margin is removed. An explicit `0.08 m` seed-only
    tracking allowance remains to compensate for bounded-control path cutting and is
    not an accepted-route safety margin. V2 obstacle templates are enlarged without
    changing the global robot footprint, velocity limits, or legacy SOP behavior.
27. Through M3, template-family selection is frozen to `rectangle=0.4`,
    `l_shape=0.4`, and `circle=0.2`. An L shape is represented by two rectangle
    components, so M2 keeps its rectangle/circle primitive geometry contract and M3
    must clear both components.
28. M4 rotates rectangle and L-shape templates by a deterministic signed angle in
    `±[15°, 45°]` relative to the start-goal direction. It recalculates the lateral
    placement after rotation with fixed-iteration bisection, preserving the frozen
    shallow direct-corridor intrusion; circles remain unoriented.
29. M5 uses a seed-derived finite first-fit order over route anchors, temporal scales,
    and coarse rotations. It accepts the first candidate that satisfies every gate;
    it does not score, refine, or continue searching after acceptance.
30. Long40 sample index `7` is the decision time, not the route start. Indices `0..7`
    form the eight-frame history/current layout; robot history is formed by joining the
    source `BaseState` history to the M4 route prefix. M6 publishes the complete
    40-sample target record and the `LocalTrajectory`/query-map future for indices
    `8..39`, exactly 32 endpoints.
31. Pre-long40 Schema `3.0.0` artifacts remain version-isolated. The v3 branch uses
    Schema `4.0.0` and layout `history8_current7_future32_v1`; implicit conversion is
    forbidden.
32. M4 goal distances move to `[4.0, 4.5] m`, M3 uses 21 band poses and 20 bounded
    intervals, and the uniform route has 40 endpoints through `8.0 s`.
33. Absolute encounter-time collision bounds are removed. Collision timing is
    decision-relative, and all 32 future intervals are eligible.
34. Verification actions consume time inside the same decision-relative 6.4-second
    target horizon; they do not create an additional 6.4 seconds of hidden target future.

## 🔁 Supersession boundary

Under the unified long40 documentation authority:

- dated v1/v2 plans remain historical implementation context only
- any earlier SOP05R design record using 23 samples or 15 future steps is superseded
- this five-file directory becomes the v3 implementation authority
- `docs/long40_system_contract.md` governs the shared SOP-03–16 time layout

Historical documents need not be deleted, but they must carry an archival notice and
cannot override the current contract.

## ⚠️ Known implementation risks

### Decision-state seam

The full encounter begins at the source BaseState, while verification must begin at a
later blocked witness. M6 must create an auditable decision-state seam that retains source
identity and the route prefix. Treating the source start as the verification state would
make the target initially visible and invalidate the intended task.

### TEB local minima

The template contract prevents centered obstacle layouts. M3 therefore uses one straight
initial band and derives a deterministic escape direction from the nonzero lateral
occluder offset; rejected templates are resampled rather than retried with alternate
initializations.

### Dual-horizon seam

The full route is expressed in the source world timeline, while model inputs and labels
are expressed in a decision-relative 6.4-second timeline. M6 must sample exactly 32
future endpoints from the fixed index-7 decision time, transform them into the decision
frame, and authenticate their relationship to the 40-endpoint full route. The first
collision may occur in any future interval subject to the lower margin. Goal arrival does
not add a suffix-horizon rejection.

### Circle visibility approximation

A centerline can intersect a tree trunk while part of the human footprint remains
visible. This disagreement is accepted by the v3 generation contract, but M10 must expose
its rate and examples.

### Current dirty working tree

The branch contains extensive pre-existing uncommitted SOP05R and downstream changes,
plus unrelated documentation changes/deletions. Implementation must:

- preserve unrelated changes
- avoid broad formatting or cleanup
- inspect `git status --short` before and after each milestone
- stage or commit nothing without explicit user instruction

## ▶️ Next action

Stop at the documentation boundary. Before any implementation work resumes, obtain an
explicit user request to audit code against the unified contract. A later implementation
task must begin with a gap report; it must not infer completion from these documents or
silently continue the interrupted M7–M10 work.

## 📝 Superseded decision: five-second/three-second dual horizon

This record is retained as history. It is superseded by the B2 long40 decision below and
must not guide new implementation.

```text
decision: separate planner and model horizons
date: 2026-07-25
planner domain:
  band poses: 20
  interval durations: 19, each in [0.1, 0.4] s
  initial interval: 0.25 s
  maximum route time: 5.0 s
  uniform route endpoints: 25 at 0.2 s
model/data/label domain:
  future endpoints: 15 at 0.2 s
  future horizon: 3.0 s
  schema: 3.0.0 unchanged
reason: a three-second task route cannot reliably both bypass the represented occluder
  and reach the configured goal under the frozen kinematic limits
publication status: no v2 artifacts published; pre-publication correction is allowed
required follow-up: rerun M1 before M3
```

## 📝 Decision update: B2 long40 horizon

```text
decision: extend both robot and human futures; keep distinct source and decision timelines
date: 2026-07-25
planner domain:
  band poses: 21
  interval durations: 20, each in [0.1, 0.4] s
  initial interval: 0.25 s
  maximum route time: 8.0 s
  uniform route endpoints: 40 at 0.2 s
  post-goal suffix: sampled from the same frozen route; no M6 horizon gate
human/model/data/label domain:
  total samples: 40
  history/current: indices 0..7; current index 7
  future endpoints: indices 8..39, 32 at 0.2 s
  future horizon: 6.4 s
  schema: 4.0.0
collision domain:
  1.2 s <= t_collision - t_decision <= 6.4 s
  all 32 future intervals scanned, including the final interval
  positive mothers retain the 1.2 s lower margin
  absolute encounter-time collision range removed
goal/template domain:
  goal distance: [4.0, 4.5] m
  collision route path fraction: [0.20, 0.95]
  goal arrival: collision anchor < t_goal <= 8.0 s
verification domain:
  actions consume time within the same decision-relative 6.4 s horizon
  no second post-action 6.4 s target future is synthesized
publication status: no completed v2 collection published; semantic version bump required
required follow-up: migrate M1-M8 code/tests before M9-M10 and release gates
```

## 📝 Decision update: shallow one-sided obstruction

```text
decision: offset occluders from the direct centerline
date: 2026-07-25
rule:
  - sample a nonzero signed lateral offset on either side of the start-goal line
  - require direct-corridor intrusion =
    minimum_represented_clearance - direct_line_clearance_m
  - freeze accepted intrusion to [0.05, 0.15] m
reason: direct travel must be blocked without constructing an unnecessarily sharp,
  centered S-turn; the M3 route remains the feasibility authority
planner fixture: offset rectangle/circle regressions pass with the frozen robot size,
  speed, and dynamics
train config digest: ec762bf92a3a296adbd857cc36f88b6c7165c6e070cf2787c9e6c20726068758
test config digest: 5c96c0725511c4f4e8d0a4940868724ed7d19b56f8f4673512780e2a65b0bcff
```

## 🧾 Evidence: M2 complete

```text
milestone: M2 — Add typed occluder geometry
date: 2026-07-25
changed files:
  - src/geometry/static_occluders.py
  - src/geometry/__init__.py
  - tests/test_static_occluders.py
  - tests/test_rasterization.py
  - tests/test_collision.py
  - docs/superpowers/plans/2026-07-25-sop05r-lightweight-teb/state.md
focused command:
  .conda-envs/sop4-risk/bin/python -m pytest -q
  tests/test_static_occluders.py tests/test_rasterization.py tests/test_collision.py
result: 74 passed in 0.58s
semantic decisions changed: none
remaining blocker: M3 lightweight TEB planner does not yet exist
next milestone: M3
```

## 🧾 Evidence: M1 complete

This evidence proves the original M1 implementation only. Its planner identity and
planner-horizon digest are superseded by the pre-publication dual-horizon decision above;
M1 is not complete again until the corrected suite passes and a replacement digest is
recorded.

```text
milestone: M1 — Freeze versions and configuration
date: 2026-07-25
changed files:
  - configs/generator_obstacle_first_teb_train.yaml
  - configs/generator_obstacle_first_teb_test.yaml
  - src/generation/sop05r_contracts.py
  - scripts/05_generate_events.py
  - tests/test_sop05r_contracts.py
  - tests/test_sop05r_teb_contracts.py
  - tests/test_05_generate_events_cli.py
  - docs/superpowers/plans/2026-07-25-sop05r-lightweight-teb/acceptance.md
  - docs/superpowers/plans/2026-07-25-sop05r-lightweight-teb/state.md
focused command:
  .conda-envs/sop4-risk/bin/python -m pytest -q
  tests/test_sop05r_contracts.py tests/test_sop05r_teb_contracts.py
  tests/test_05_generate_events_cli.py
result: 78 passed in 0.73s
semantic decisions changed: none
remaining blocker: M2 geometry authority does not yet exist
next milestone: M2
```

## 🧾 Evidence: M1 corrected and shallow-obstruction contract frozen

```text
milestone: M1 pre-publication correction
date: 2026-07-25
changed files:
  - configs/generator_obstacle_first_teb_train.yaml
  - configs/generator_obstacle_first_teb_test.yaml
  - src/generation/sop05r_contracts.py
  - tests/test_sop05r_teb_contracts.py
  - docs/superpowers/plans/2026-07-25-sop05r-lightweight-teb/{contracts,full-spec,milestones,acceptance,state}.md
focused command:
  .conda-envs/sop4-risk/bin/python -m pytest -q
  tests/test_sop05r_contracts.py tests/test_sop05r_teb_contracts.py
  tests/test_05_generate_events_cli.py
result: 80 passed in 0.73s
train config digest: ec762bf92a3a296adbd857cc36f88b6c7165c6e070cf2787c9e6c20726068758
test config digest: 5c96c0725511c4f4e8d0a4940868724ed7d19b56f8f4673512780e2a65b0bcff
semantic decisions changed:
  - direct-corridor intrusion is frozen to [0.05, 0.15] m
  - template occluders use nonzero lateral offsets, not centered placements
  - initialization_ids is frozen to [straight, bypass_left, bypass_right]
remaining blocker: M3 full optimizer acceptance is incomplete
next milestone: M3
```

## 🧾 Evidence: M3 objective diagnostics and sampled-safety scope

```text
milestone: M3 in-progress refinement
date: 2026-07-25
changed files:
  - src/planning/lightweight_teb.py
  - tests/test_lightweight_teb.py
  - docs/superpowers/plans/2026-07-25-sop05r-lightweight-teb/{contracts,full-spec,milestones,acceptance,state}.md
focused command:
  .conda-envs/sop4-risk/bin/python -m pytest -q
  tests/test_lightweight_teb.py tests/test_replanning.py
  tests/test_input_oracle_isolation.py tests/test_query_maps.py
result: 36 passed in 11.12s
semantic decisions changed:
  - all frozen TEB objective terms are published in per-candidate diagnostics
  - dense sampled collision checks plus clearance margin replace an analytic continuous-time requirement
  - post-arrival stationary holding is permitted but not an M3 acceptance gate
remaining blocker: M3 broader route and integration evidence is incomplete
next milestone: M3
```

## 🧾 Evidence: reduced M3 clearance and enlarged templates

```text
milestone: M3 in-progress clearance retuning
date: 2026-07-25
changed files:
  - configs/generator_obstacle_first_teb_{train,test}.yaml
  - src/generation/sop05r_contracts.py
  - src/planning/lightweight_teb.py
  - tests/test_{sop05r_teb_contracts,lightweight_teb}.py
  - docs/superpowers/plans/2026-07-25-sop05r-lightweight-teb/{contracts,milestones,acceptance,state}.md
focused commands:
  .conda-envs/sop4-risk/bin/python -m pytest -q
  tests/test_sop05r_contracts.py tests/test_sop05r_teb_contracts.py
  tests/test_05_generate_events_cli.py
  .conda-envs/sop4-risk/bin/python -m pytest -q
  tests/test_lightweight_teb.py tests/test_replanning.py
  tests/test_input_oracle_isolation.py tests/test_query_maps.py
result: 80 passed; 36 passed
train config digest: 1e30c2eb9b98e4e2c51dd4c06c3d7a70e54e2ecc641ee1309fc42decc7900897
test config digest: c854523ecedbd53d33a38f0775340f78b1c3ced701ce48b349a9fb98cd7d6c15
semantic decisions changed:
  - minimum represented-obstacle clearance is 0.15 m
  - a 0.08 m seed-only tracking allowance replaces the prior 0.20 m margin
  - rectangle and circle templates are enlarged
remaining blocker: M3 broader route and integration evidence is incomplete
next milestone: M3
```

## 🧾 Historical evidence: pre-long40 M4–M6 decision-local mother construction

```text
milestones: M4, M5, M6 focused implementation
date: 2026-07-25
changed files:
  - src/generation/sop05r_teb_templates.py
  - src/generation/anchored_human_placement.py
  - src/generation/history_visibility.py
  - src/generation/sop05r_teb_decision_state.py
  - src/generation/sop05r_teb_event_sampler.py
  - tests/test_{sop05r_teb_templates,anchored_human_placement,sop05r_history_visibility,sop05r_teb_decision_state,sop05r_teb_event_sampler}.py
focused commands:
  .conda-envs/sop4-risk/bin/python -m pytest -q
  tests/test_sop05r_teb_contracts.py tests/test_sop05r_teb_templates.py
  tests/test_sop05r_history_visibility.py tests/test_anchored_human_placement.py
  tests/test_sop05r_teb_decision_state.py tests/test_sop05r_teb_event_sampler.py
  .conda-envs/sop4-risk/bin/python -m pytest -q
  tests/test_collision.py tests/test_query_maps.py
result: 43 passed; 51 passed
train config digest: 0afb2f0da4e9343fe181b4afd3b2410ba036d8164b6f9a26df19b7c9d02410a7
test config digest: 5d35ba76f103e17aa9efdaf623f9adb0aec27fdcbab3d351757341f583d265ca
semantic decisions changed:
  - human snippet index 7 is the hidden decision frame
  - at least four of eight history frames are visible; transition index is not fixed
  - collision margin is 1.2 s for verification action plus braking, excluding replanning
  - source history and route prefix are joined, then all event products are rebased into
    one decision-local frame
  - M6 published one full route and one exact 15-step suffix with no alternative route
remaining blocker: this evidence is superseded by v3 long40 contracts and must be rerun
next milestone: migrate M1-M8 to v3 before any release gate
```

## 📝 State update format

After each milestone, replace the relevant status row and append one evidence entry:

```text
milestone:
date:
changed files:
focused command:
result:
semantic decisions changed:
remaining blocker:
next milestone:
```

An evidence entry must contain actual command output or artifact identity; intention is
not progress.
