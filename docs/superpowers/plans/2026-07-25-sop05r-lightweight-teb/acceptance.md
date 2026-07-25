# SOP05R lightweight TEB acceptance gates

_Test commands, quantitative thresholds, integrity requirements, and release ladder for SOP05R v2._

---

## 📚 Document set

- [Full specification](./full-spec.md)
- [Contracts](./contracts.md)
- [Milestones](./milestones.md)
- [Acceptance](./acceptance.md)
- [State](./state.md)

Commands in this file describe the required post-implementation interface. A command is
not evidence until its exit code and structured output have been inspected.

## ✅ Definition of done

SOP05R v2 is complete only when:

- every contract in [contracts.md](./contracts.md) has code and test evidence
- all focused unit and integration suites pass
- legacy and SOP05R v1 focused regressions still pass
- the 10-template deterministic gate passes
- the user-approved 100-event visual audit passes
- the user-approved 1,000-event statistical smoke passes every numeric threshold
- strict reload and checksum verification pass
- deterministic rerun identity is demonstrated
- no target/oracle information enters the initial planner
- no target-scale generation has begun before the previous gates pass

Implementation completion is not inferred from file presence, nonempty output, or a zero
exit code alone.

## 🧪 Milestone test commands

Use the verified project interpreter:

```bash
PY=.conda-envs/sop4-risk/bin/python
```

### M1 — Contracts and CLI

```bash
$PY -m pytest -q \
  tests/test_sop05r_contracts.py \
  tests/test_sop05r_teb_contracts.py \
  tests/test_05_generate_events_cli.py
```

Required result:

- all tests pass
- planner version is exactly `lightweight_teb_planner_v2`
- normalized planner config freezes 20 band nodes, `0.25 s` initial interval,
  `[0.1, 0.4] s` interval bounds, `5.0 s` maximum route time, and `0.2 s`
  route sampling
- normalized planner config freezes a `0.15 m` minimum represented-obstacle
  clearance and `0.08 m` bypass tracking allowance
- normalized planner config freezes `initialization_ids = [straight]`
- normalized template config freezes `rectangle=0.4`, `l_shape=0.4`, and
  `circle=0.2`; an L shape is represented as two rectangle components
- maximum route time is representable by the interval bounds and produces exactly 25
  sampled future endpoints
- normalized generation config freezes direct-corridor intrusion to `[0.05, 0.15] m`
- base/model horizon remains exactly 15 steps over `3.0 s`; Schema remains `3.0.0`
- `obstacle_first_teb` rejects missing or v1 config versions
- `legacy` and `obstacle_first` argument behavior remains unchanged

### M2 — Static occluders

```bash
$PY -m pytest -q \
  tests/test_static_occluders.py \
  tests/test_rasterization.py \
  tests/test_collision.py
```

Required result:

- exact circle/segment and oriented-rectangle/segment fixtures pass
- tangency behavior matches configured epsilon
- vectorized and scalar results are identical
- raster masks are deterministic and bounded
- L-shape planning inputs are checked component-wise through the existing rectangle
  geometry authority

### M3 — Lightweight TEB

```bash
$PY -m pytest -q \
  tests/test_lightweight_teb.py \
  tests/test_replanning.py \
  tests/test_input_oracle_isolation.py \
  tests/test_query_maps.py
```

Required result for every accepted route:

- band has exactly 20 immutable poses and 19 bounded interval durations
- exact request start is the implicit `t=0` state and first band pose
- uniform route has exactly 25 endpoint poses/controls at `[0.2, ..., 5.0] s`
- terminal goal error is within frozen tolerance
- goal arrival is no later than `5.0 s`
- first control is continuous with request initial control
- velocity, acceleration, angular-rate, curvature, and nonholonomic limits pass
- represented-occluder clearance lies in the configured band
- rectangle, composite L-shape, and circle bypass fixtures each preserve the configured
  clearance for every represented primitive component
- dense static-occupancy and represented-occluder checks pass at the frozen route
  sample spacing; the required clearance margin supplies conservatism between samples
- diagnostic cost publishes every frozen term: path length, time, smoothness, obstacle
  hinge, nonholonomic, velocity, acceleration, goal heading, and initial control
- repeated request bytes produce identical route and diagnostics
- exactly one straight-initialized candidate executes the frozen optimizer update count
- no `LocalTrajectory` or query map is constructed by M3

### M4 — Goal/occluder templates

```bash
$PY -m pytest -q tests/test_sop05r_teb_templates.py
```

Required result:

- no target/snippet is read before planner completion
- rectangle, L-shape, and circle templates are represented
- every rectangle and L shape has a deterministic signed orientation within
  `±[15°, 45°]` of the start-goal direction
- every occluder has a nonzero signed lateral offset from the direct start-goal line
- direct-corridor intrusion is in the frozen `[0.05, 0.15] m` range
- accepted TEB route reaches the same goal without static collision
- source BaseState and OracleContext digests do not change

### M5 — Anchored placement and centerline visibility

```bash
$PY -m pytest -q \
  tests/test_anchored_human_placement.py \
  tests/test_sop05r_history_visibility.py
```

Required result:

- every transformed anchor remains within `1e-6 m` of the world collision point
- spatial scale is exactly `1.0`
- rigid pairwise distances, speed magnitudes, and acceleration magnitudes are preserved
- headings and velocity vectors rotate by the same angle
- initial-visible fixture has no centerline intersection at start
- it has at least one synchronized blocked sample before collision
- one blocked sample is sufficient
- non-synchronized cross-time point pairs cannot satisfy the predicate
- rectangle and circle fixtures both pass

### M6 — Decision state and mother event

```bash
$PY -m pytest -q \
  tests/test_sop05r_teb_decision_state.py \
  tests/test_sop05r_teb_event_sampler.py \
  tests/test_collision.py
```

Required result:

- selected decision time equals a persisted centerline-occlusion witness
- action, braking, and replanning margin remains before collision
- continuous first collision exists and is not endpoint-only
- full route reaches the exact shared goal within `5.0 s`
- decision-relative nominal suffix has exactly 15 endpoints at `0.2 s` spacing
- suffix query maps recompute exactly under the existing Schema `3.0.0` authority
- continuous first collision occurs after the decision and within the three-second suffix
- suffix may end before goal arrival without invalidating full-route reachability
- event has exactly one full route, one nominal suffix, and no alternative-route requirement
- event, world, target, route, anchor, witness, and source IDs round-trip
- event identity is deterministic

### M7 — Publication and loading

```bash
$PY -m pytest -q \
  tests/test_sop05r_teb_trajectory_store.py \
  tests/test_sop05r_teb_run.py \
  tests/test_sop05r_teb_output_loader.py \
  tests/test_05_generate_events_cli.py
```

Required result:

- canonical JSON and deterministic NPZ round-trip
- unknown, missing, extra, or tampered arrays fail closed
- full-route band arrays, uniform five-second arrays, and three-second suffix arrays have
  distinct authenticated shape/dtype metadata
- outer checksums and semantic digests are verified
- v1/v2 mixed artifacts are rejected
- partial quota output has no completion marker
- complete output self-reloads before publication
- one-worker and multi-worker semantic digests match

### M8 — Verification and replanning

```bash
$PY -m pytest -q \
  tests/test_sop05r_teb_revealability.py \
  tests/test_verification_policy.py \
  tests/test_counterfactual_verify.py \
  tests/test_verification_actions.py
```

Required result:

- target-hidden actions use static-only replanning
- target-revealed actions use only observation-derived dynamic state
- hidden Oracle future never enters planner requests
- every replan uses the exact same world-frame goal, five-second route bound, and
  task-cost authority
- verification risk remains evaluated over the unchanged three-second post-action window
- at least one deterministic fixture has a moving action that beats matched wait
- stop is never counted as an active moving action

### M9 — SOP06 handoff

```bash
$PY -m pytest -q \
  tests/test_pair_variants.py \
  tests/test_sop06_pipeline.py
```

Required result:

- v2 target-present and target-removed variants share source, decision, static scene,
  full route, nominal suffix, and goal
- only target semantics change across the pair
- no alternative trajectory lookup occurs in the v2 branch
- legacy and v1 focused fixtures remain unchanged

### M10 — Audit producers

```bash
$PY -m pytest -q \
  tests/test_sop05r_teb_audit.py \
  tests/test_sop05r_teb_visuals.py \
  tests/test_sop05r_teb_release_gate.py \
  tests/test_05_render_sop05r_audit_cli.py
```

Required result:

- audit metrics recompute source evidence
- figures are deterministic, fixed-scale, and nonblank
- visual layers include goal, occluder shape, full route, three-second nominal suffix,
  human motion, collision anchor, occlusion witness, decision pose, and verification
  traces
- audit completion fails for incomplete or tampered source collections

## 🔬 Deterministic 10-template gate

The release-gate test uses production v2 APIs with deterministic synthetic BaseState,
rectangle/circle occluders, and real `MotionSnippet` contracts:

```bash
$PY -m pytest -q \
  tests/test_sop05r_teb_release_gate.py::test_ten_template_gate
```

All 10 fixtures must satisfy:

```text
source start unchanged                         10/10
direct path blocked by represented occluder   10/10
TEB route dynamically valid                   10/10
route reaches exact shared goal               10/10
decision suffix has 15 future endpoints       10/10
suffix query maps reproduce exactly            10/10
human anchor invariant                        10/10
required start-visibility stratum valid        10/10
centerline occlusion witness exists            10/10
continuous non-endpoint collision              10/10
collision lies inside 3-second suffix          10/10
single nominal trajectory record               10/10
strict serialization reload                    10/10
```

At least four fixtures use circles and at least four use rectangles. At least one fixture
must fail before the rotation solver and pass after anchored rotation, proving that the
solver—not a hard-coded target—is responsible for placement.

Any failure requires a minimal RED regression test before implementation changes.

## 👁️ Real 100-event visual audit

This stage requires user approval because it writes a versioned real-data audit output.
Use authenticated SOP03 outputs, not direct reads from `data/`.

```bash
PY=.conda-envs/sop4-risk/bin/python
SOP03_ROOT=outputs/sop03_thor_motion_snippet_v2_schema3_47b3acd_v1
OUT=outputs/sop05r_teb_train_visual_audit_100_v1
AUDIT=outputs/sop05r_teb_train_visual_audit_100_v1_audit

$PY scripts/05_generate_events.py \
  --generator-mode obstacle_first_teb \
  --sop03-root "$SOP03_ROOT" \
  --split train \
  --generator-config configs/generator_obstacle_first_teb_train.yaml \
  --verification-action-config configs/verification_actions.yaml \
  --output-dir "$OUT" \
  --seed 310725 \
  --accepted-quota 100 \
  --max-base-states 400 \
  --checksum-workers 8 \
  --workers 4 \
  --git-executable "$HOME/.local/git/bin/git"

$PY scripts/05_render_sop05r_audit.py \
  --generator-mode obstacle_first_teb \
  --source-root "$OUT" \
  --output-dir "$AUDIT" \
  --sample-count 100 \
  --seed 310725
```

Required review:

- strict loader verifies all 100 events
- every event is included in the visual bundle; no failed sample is skipped
- occluder shape and raster agree visually
- full route is smooth, reaches the goal, and has no static clipping
- decision-relative three-second suffix agrees with the full route or deterministic goal
  hold
- anchored human trajectory does not pass through the occluder
- collision anchor is internal to both motions
- initial-visible and initially-hidden strata match metadata
- selected decision pose is at the persisted centerline witness
- verification-action traces are physically plausible
- every unexplained artifact creates a regression test and a new versioned audit output

Centerline/footprint-raycast disagreements are counted and visualized. They are not an
automatic rejection because centerline intersection is the frozen v2 authority, but any
systematic disagreement must be reported before the 1,000-event gate.

## 📊 Real 1,000-event statistical smoke

This stage also requires user approval.

```bash
PY=.conda-envs/sop4-risk/bin/python
SOP03_ROOT=outputs/sop03_thor_motion_snippet_v2_schema3_47b3acd_v1
OUT=outputs/sop05r_teb_train_1k_v1

$PY scripts/05_generate_events.py \
  --generator-mode obstacle_first_teb \
  --sop03-root "$SOP03_ROOT" \
  --split train \
  --generator-config configs/generator_obstacle_first_teb_train.yaml \
  --verification-action-config configs/verification_actions.yaml \
  --output-dir "$OUT" \
  --seed 310725 \
  --accepted-quota 1000 \
  --max-base-states 4000 \
  --checksum-workers 8 \
  --workers 8 \
  --git-executable "$HOME/.local/git/bin/git"
```

The summary must define and expose every denominator used below.

### Hard integrity thresholds

```text
source BaseState mutation count                         == 0
initial planner target/oracle input count               == 0
published events with route count != 1                  == 0
published full routes with sample count != 25           == 0
published nominal suffixes with future steps != 15      == 0
published nominal suffixes with horizon != 3.0 s        == 0
published collisions outside nominal suffix             == 0
published events with changed world-frame goal          == 0
published events without an occlusion witness           == 0
published events without continuous collision           == 0
published endpoint-only collisions                      == 0
published spatial scales != 1.0                         == 0
NaN/Inf count in structured artifacts                   == 0
strict-reload failures                                  == 0
checksum/digest failures                                == 0
unknown rejection reasons                              == 0
```

### Scientific and efficiency thresholds

```text
abs(seen_then_occluded_fraction - 0.80)               <= 0.03
abs(initially_hidden_fraction - 0.20)                 <= 0.03
geometry_eligible_teb_success                          >= 0.80
TEB-success-to-continuous-collision-mother rate        >= 0.50
end-to-end selected quota                              == 1000
training active-revealable fraction                    >= 0.70
median goal-occluder templates per accepted mother     <= 8
median refined rotations per accepted mother           <= 8
```

Definitions:

```text
geometry_eligible_teb_success =
    successful TEB templates / geometry-eligible templates

TEB-success-to-continuous-collision-mother rate =
    accepted continuous-collision mothers / successful TEB templates

training active-revealable fraction =
    selected mothers with >=1 active moving action / selected mothers
```

The centerline/footprint-raycast disagreement rate is mandatory audit output but has no
release threshold in v2 because the user explicitly accepts small centerline
approximation error. It must not be omitted or redefined after seeing the result.

### Deterministic prefix rerun

Re-run a frozen 20-base prefix with identical seed and one worker, then four workers:

```bash
$PY -m pytest -q \
  tests/test_sop05r_teb_run.py::test_worker_count_preserves_semantic_digest
```

Required result:

- identical schedule digest
- identical selected event IDs and order
- identical event/world/full-route/suffix semantic digests
- identical aggregate counters

## 🔁 Focused compatibility regression

After the 1,000-event smoke, run the new suites plus directly affected existing suites:

```bash
$PY -m pytest -q \
  tests/test_sop05r_contracts.py \
  tests/test_static_occluders.py \
  tests/test_lightweight_teb.py \
  tests/test_sop05r_teb_templates.py \
  tests/test_anchored_human_placement.py \
  tests/test_sop05r_teb_decision_state.py \
  tests/test_sop05r_teb_event_sampler.py \
  tests/test_sop05r_teb_trajectory_store.py \
  tests/test_sop05r_teb_run.py \
  tests/test_sop05r_teb_output_loader.py \
  tests/test_sop05r_teb_revealability.py \
  tests/test_sop05r_teb_audit.py \
  tests/test_sop05r_teb_visuals.py \
  tests/test_05_generate_events_cli.py \
  tests/test_pair_variants.py \
  tests/test_sop06_pipeline.py \
  tests/test_verification_actions.py \
  tests/test_counterfactual_verify.py \
  tests/test_input_oracle_isolation.py \
  tests/test_collision.py \
  tests/test_query_maps.py
```

Do not substitute a repository-wide test run for this focused evidence. Additional suites
may be run only when a focused failure identifies a broader dependency.

## 🔐 Publication and failure policy

- every run uses a new no-overwrite output directory
- partial runs publish diagnostics without a completion marker
- quota deficits remain explicit and fail the command with the frozen nonzero exit code
- no sample is skipped from strict reload or visual audit
- no threshold is lowered after observing results without a new version and documented
  scientific decision
- no target-scale generation begins while any hard integrity or numeric gate is unmet
- completion review includes `git diff --check` and `git status --short`
