# SOP05R lightweight TEB implementation state

_Current status, frozen decisions, evidence, risks, and next action as of 2026-07-25._

---

## 📚 Document set

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
| v2 method decisions | complete | dual-horizon correction in this document set |
| v2 immutable contracts | M1 corrected | 80 focused contract/CLI tests pass; digests recorded below |
| v2 code | M2 complete; M3 in progress | one straight-band candidate, fixed updates, full objective-cost diagnostics, and focused M3 suite pass; broader integration remains pending |
| current SOP05R v1 code | substantial uncommitted implementation | current working tree |
| v1 focused tests | not re-run in this documentation task | prior implementation files only |
| v1 real smoke | failed scientific gate | 12-base diagnostic below |
| v2 10-template gate | not run | depends on M1–M6 |
| v2 100-event audit | not run | requires user approval after fixture gate |
| v2 1,000-event smoke | not run | requires user approval after visual gate |

No v2 completion claim is valid at this state.

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

It may be deleted after the v2 implementation has permanent regression coverage. It is
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
12. Initial visibility means the start centerline is clear.
13. One intermediate blocked synchronized sample is sufficient to establish occlusion;
    consecutive hidden frames are not required.
14. Small centerline-versus-footprint visibility disagreement is acceptable but must be
    measured.
15. A verification decision must use a blocked witness with enough pre-collision action,
    braking, and replanning margin.
16. Verification actions re-use the same lightweight TEB and the same world-frame goal.
17. Post-action replanning may use a target only when that target has become
    deployment-observable; Oracle future remains forbidden.
18. Active revealability must arise from geometry and executable actions, not relaxed
    label thresholds.
19. Use a dual horizon: a bounded five-second full planner route for reachability,
    collision anchoring, and provenance, while retaining the existing three-second,
    15-step model/data/label horizon.
20. M3 returns `PlannedTebRoute`; M6 alone derives the decision-relative
    `LocalTrajectory` and query maps.
21. The three-second decision suffix is not required to reach the goal. Goal reachability
    is an invariant of the full route. A post-arrival stationary hold is allowed but is
    not a separately required M3 acceptance condition.
22. The accepted first collision must remain inside the decision-relative three-second
    suffix. The longer route extends task completion after collision; it does not extend
    the model or collision-label horizon.
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

## 🔁 Supersession boundary

After M1 is approved:

- `docs/superpowers/plans/2026-07-24-sop05r-obstacle-first.md` remains historical v1
  implementation context
- `docs/sop05r_obstacle_first_event_generation.md` remains the earlier complete SOP05R
  design record
- this five-file directory becomes the v2 implementation authority
- legacy SOP05 behavior remains governed by its existing documents and code

No old document should be deleted as part of v2 implementation.

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
are expressed in a decision-relative three-second timeline. M6 must sample exactly 15
future endpoints after the selected witness, transform them into the decision frame, and
authenticate their relationship to the full route. The first collision must be inside
this suffix, but requiring the suffix also to reach the goal would recreate the original
reachability conflict.

### Circle visibility approximation

A centerline can intersect a tree trunk while part of the human footprint remains
visible. This disagreement is accepted by the v2 generation contract, but M10 must expose
its rate and examples.

### Current dirty working tree

The branch contains extensive pre-existing uncommitted SOP05R and downstream changes,
plus unrelated documentation changes/deletions. Implementation must:

- preserve unrelated changes
- avoid broad formatting or cleanup
- inspect `git status --short` before and after each milestone
- stage or commit nothing without explicit user instruction

## ▶️ Next action

Continue with M5 anchored human placement. M4 now produces target-blind, deterministic
goal/occluder templates and valid nominal routes; downstream placement must consume only
those accepted templates and may not alter their frozen goal or static geometry.

## 📝 Decision update: dual horizon

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
  - initialization_ids is frozen to [straight]
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
