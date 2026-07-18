# SOP05 Blind-Reachability-First Design

## Status and decision

This design replaces the SOP05 `joint_occluder_first_v4` production proposal
with a new `blind_reachability_first_v1` proposal. The primary objective is to
produce enough physically valid collision-under-physical-occlusion events for
downstream training. Every primary mother has one causal environment
occluder. A complete six-variant group is not a publication requirement; it is
an optional audit artifact.

The design keeps the frozen SOP03/SOP04 inputs, 23-point real motion snippets,
SE(2)-only transplantation, `time_scale=1.0`, and the no-extrapolation rule. It
does not change `src/contracts.py`, `configs/base.yaml`, or the generated-event
target-motion array schema.

## Motivation and evidence

The current v4 proposal first selects a trajectory conflict point and places a
programmatic occluder at a normal offset from that point. After a real snippet
is transplanted, it realigns the occluder to the target line of sight. This
couples the occluder, conflict point, and target into the repeated pattern of a
front-facing obstacle with the target behind it.

The formal v4 run attempted 3,836 complete joint candidates, accepted 24, and
published 10. All 10 selected events used one trajectory ID; nine were
structural blind-sector events and only one used an environment occluder. A
separate five-group complete-sixpack audit also selected one trajectory ID,
one programmatic occluder per group, a fixed collision time, and no base static
occupancy. These results show both low proposal efficiency and geometry mode
collapse.

Increasing the retry count or tuning v4 quantiles would retain the same
coupling. Changing the proposal inside v4 without changing its identity would
silently alter a frozen scientific contract. A new algorithm identity and new
publication are therefore required.

## Goals

1. Maximize accepted, physically valid collision-under-occlusion mother events
   per CPU-hour.
2. Use real split-local SOP03 motion snippets without time scaling or
   extrapolation.
3. Decouple obstacle placement from one preselected conflict-point normal so
   occlusion geometry can occur at different bearings, ranges, orientations,
   and approach sides.
4. Turn random combination retries into a staged feasibility search with cheap
   broad-phase filters before exact 23-frame validation.
5. Preserve deterministic output and auditable counts at every rejection
   stage.
6. Keep paired negatives available for training and analysis without requiring
   every mother event to form a complete sixpack.

## Non-goals

- Generating a synthetic straight or idealized target trajectory.
- Changing SOP03 snippet extraction, SOP04 candidate-trajectory contracts, the
  base BEV grid, or object footprints.
- Supporting v4 and v5 production artifacts through one permissive loader.
- Enforcing hard environment/structural/mixed publication proportions.
- Requiring two causal occluders in the first production version.

Structural-FOV and mixed events may be generated as separately reported
auxiliary publications, but they cannot fill the primary physical-occluder
quota. No fixed ratio is imposed among auxiliary event kinds.

## Required SOP04 time contract

SOP05 v5 accepts only the corrected SOP04 future-endpoint layout:

```text
trajectory.poses[i]       time = (i + 1) * dt
target.future_poses[i]    time = (i + 1) * dt
controls[i]               interval = [i * dt, (i + 1) * dt]
```

For `dt=0.2` and 15 steps, both pose arrays cover `0.2-3.0 s` and pair by the
same index. The SOP04 input adapter must require the new
`future_endpoints_dt_to_horizon_v1` layout metadata and reject the old
`0.0-2.8 s` bank. SOP05 integration and publication remain blocked until the
separately owned SOP04 correction supplies its commit, artifact root, checksum
digest, and audit evidence. SOP05 must not implement a shifted-index fallback.

## Production data flow

### 1. Select the base and robot trajectory

Select a split-local `BaseState`, `OracleContext`, and collision-free
`LocalTrajectory` by a stable seeded key. The candidate pool must include every
eligible SOP04 trajectory rather than stopping at the first trajectory that
produces an event.

Construct continuous collision sweeps for robot history plus future and all
context-object history plus future. Inflate them by the configured footprints.
These sweeps define where a new static obstacle may not be placed.

### 2. Propose one causal environment occluder

Sample one wall, shelf, or pillar from free space in a configured interaction
band around the robot trajectory. The obstacle proposal is stratified over
relative bearing, range, orientation, type, and dimensions, but it is not
anchored to `conflict_point + normal_offset`.

Reject the proposal before target work if it:

- overlaps base static occupancy;
- overlaps any robot or context continuous sweep;
- lies outside the BEV;
- blocks the robot candidate path; or
- produces no useful line-of-sight shadow in the interaction region.

The first production version has exactly one causal programmatic occluder.
Optional background clutter is a later, best-effort augmentation: failure to
place it must not discard a valid mother event, and any accepted clutter still
requires complete scene revalidation.

### 3. Build footprint-safe blind-region masks

Ray-cast the current visibility mask using exactly the formal renderer's
current-frame inputs: base static occupancy, the causal programmatic obstacle,
current visible context occupancy, sensor pose, FOV, and range. Oracle future
must not participate. The proposal and renderer call the same visibility
kernel and must produce byte-identical masks for the same persisted inputs.

Remove occupied cells from the unobservable mask. For each allowed target
footprint digest and orientation bin, derive a center mask whose membership
means the complete target footprint is currently hidden and collision-free. A
point-center or footprint-class-only test is insufficient, especially for
rotated rectangles. Exact polygon visibility remains mandatory after the
candidate's continuous yaw is known.

Persist the causal occluder identity, visibility algorithm version, raw blind
mask digest, footprint-safe center-mask digest, and valid-cell count.

### 4. Enumerate conflict anchors

Enumerate every discrete trajectory anchor in the configured `1.0-2.2 s`
window that satisfies curvature, bounds, and local-free-space checks. Do not
commit to one randomly selected anchor before testing reachability.

For each anchor, retain the exact robot pose, conflict point, tangent, normal,
and timestamp. The target source anchor remains
`current_index 7 + conflict_time_s / dt`; no source index may leave the frozen
23-point snippet.

### 5. Construct snippet-specific reachable arcs

Pre-index eligible snippets by split, target type, footprint, anchor index,
source displacement length, speed/acceleration validity, and chord-deviation
diagnostics. Invalid-dynamics snippets are excluded once during index
construction rather than rediscovered in every complete attempt.

For snippet `s` and conflict anchor `p*`, let

```text
delta_s = source_position(anchor) - source_position(current_index=7)
```

For an allowed SE(2) rotation `R(theta)`, the only current center that maps the
real source anchor to the conflict point is

```text
x0 = p* - R(theta) @ delta_s
```

For one snippet and anchor, allowed crossing angles therefore form one or two
arcs around `p*`, not an arbitrary two-dimensional fan. The union across
snippet displacement lengths and allowed angles is an annular sector. The
configuration freezes the finite angle schedule and arc-query resolution.
Query the footprint-safe blind-region mask with the exact continuous `x0` for
each scheduled `theta`; never snap `x0` to a grid-cell center. The grid is a
broad-phase lookup only.

This construction must not modify snippet time, speed, curvature, history, or
future. It only proposes an SE(2) transform.

### 6. Apply a cheap straight-chord broad phase

Most snippets are close to straight over the short current-to-conflict window.
Use the chord from `x0` to `p*`, expanded by the target footprint and the
snippet's recorded lateral chord-deviation bound, to triage proposals cheaply.
If this conservative tube is disjoint from occupied space, the interval is a
high-priority coarse pass. Intersection means unresolved rather than proven
collision, so the candidate enters a lower-priority exact-validation fallback.
Only an occupied current/anchor footprint or another independently proved
collision may be rejected at this stage.

This broad phase is not scientific ground truth and never replaces the source
trajectory. Every coarse outcome is counted separately. Curved snippets are
not declared physically invalid merely because the chord approximation is
poor.

### 7. Run exact 23-frame validation

Apply the candidate SE(2) transform to the original 8 history and 15 future
poses, then require all of the following:

- `history_poses` and `future_poses` retain their frozen shapes, `float32`
  dtype, finite values, and exact current/history seam;
- target history/current/future footprints do not intersect base static
  occupancy or any programmatic obstacle;
- target footprints do not collide with context-object footprints;
- the target does not overlap the current robot;
- the complete target footprint is currently unobservable under the fixed
  mask, and at least one future footprint continuously becomes visible;
- target and robot footprints collide at the selected future anchor under the
  shared timestamp contract;
- configured target speed and acceleration bounds remain satisfied; and
- no time scaling, interpolation beyond the contract, or extrapolation occurs.

Static/programmatic-obstacle and dynamic footprint validation reuses the
frozen continuous-sweep certificate: densify SE(2) motion to at most half a
BEV cell of translation and `5 degrees` of yaw, compute real-footprint signed
clearance, certify each interval with the conservative motion bound, and
recursively inspect unresolved midpoints. Discrete 23-frame overlap alone is
not a final collision test.

Only this exact continuous stage decides physical acceptance. The straight
approximation cannot certify the final event.

### 8. Select accepted mothers without mode collapse

Evaluate a deterministic, configuration-bounded candidate schedule for the
assigned base/trajectory work item and retain every feasible result in that
schedule. Select by a stable seed-derived key. Never accept the first feasible
candidate as an implicit priority rule.

The canonical reducer may use soft inverse-frequency weights for already
common trajectory IDs, occluder-bearing bins, target approach sides, and
conflict-time bins. It applies the deterministic greedy ranking to the
canonical candidate order with stable tie breaks, so worker completion order
cannot affect selection. These weights are not hard publication quotas and
cannot make an otherwise sufficient total output incomplete.

## Mother-event publication contract

A mother event is eligible for the main SOP05 publication when it has:

1. one causal physical environment occluder outside the robot/context sweeps;
2. one real, split-local target snippet with complete provenance;
3. a physical collision with the selected robot trajectory;
4. complete target-footprint occlusion at the current frame;
5. continuous future emergence;
6. exact 23-frame physical validity; and
7. deterministic, loader-verifiable event/world identity.

Publication completion depends only on the requested total number of eligible
physical-occluder mother events. Obstacle type, trajectory ID, and
paired-variant coverage are diagnostic distributions, not completion quotas.

## Paired variants and sixpack scope

Paired generation is decoupled from mother publication:

- Each collision mother may independently attempt `near_miss`,
  `temporal_safe`, `spatial_safe`, `irrelevant_hidden`, and
  `empty_blind_spot` variants.
- Every valid variant is retained with an explicit coverage mask; one failed
  variant does not delete the mother or other valid variants.
- Training may consume unpaired mothers, partial pairs, and independently
  generated valid negatives according to its configured sampler.
- A complete sixpack is generated only for a small, versioned audit subset used
  for visualization, shortcut checks, and controlled ablations.
- Sixpack success rate is reported independently and is never substituted for
  mother-event throughput.

Advance the paired producer to `independent_partial_pairs_v1` and the persisted
group contract to `sop06_partial_pair_group_v1`. `collision` is the required
mother position; every other position is optional and represented by the
existing six-position coverage mask plus an enumerated missing reason. A
singleton collision group, a partial group, and a complete audit group retain
the same mother-derived `pair_group_id`, but the training sampler explicitly
selects which coverage patterns it accepts. SOP06 v5 rejects old joint-pair
versions and exposes distinct entry points for mother rendering, partial-pair
rendering, and complete-sixpack audit rendering.

## Module boundaries

The implementation should keep the existing public contracts and introduce
small generation units with explicit responsibilities:

- obstacle free-space proposal and causal-occluder validation;
- visibility and footprint-safe blind-center-mask construction;
- snippet reachability indexing and arc/annular-sector queries;
- chord broad-phase scoring;
- exact transformed-snippet validation;
- deterministic feasible-pool selection; and
- stage-wise structured reporting.

`event_sampler.py` remains orchestration rather than accumulating all geometry
logic. Importing any module must not generate events or write outputs.

## Versioning and migration

- Set the formal generator algorithm identity to
  `blind_reachability_first_v1`.
- Advance the formal tokens to `sop05_generation_run_v5`,
  `sop05_run_manifest_v3`, `sop05_generation_summary_v3`,
  `sop05_pair_generation_report_v3`, `sop05_producer_complete_v3`, and
  `sop05_publication_semantic_digest_v2`.
- Advance the SOP04 input lock to `sop05_input_lock_v2` and require
  `future_endpoints_dt_to_horizon_v1`.
- Advance paired production to `independent_partial_pairs_v1` and
  `sop06_partial_pair_group_v1`; retire `joint_environment_pair_v2` from formal
  v5 consumers.
- New formal consumers accept only the new producer and generator identity.
  They must not reinterpret or upgrade v4 artifacts.
- Existing v4 directories remain immutable reference outputs.
- Publish v5 into a new, atomically exposed output directory with a new
  external semantic digest.
- Regenerate SOP06/SOP07 artifacts whose lineage depends on v4 events.
- Treat checkpoints, calibration statistics, metrics, figures, and evaluation
  artifacts trained or derived from v4 events as invalid for v5 claims.
- SOP03/SOP04 publications remain valid inputs and are not regenerated solely
  for this proposal change.
- No dependency, `src/contracts.py`, `configs/base.yaml`, `DECISIONS.md`, or
  `STATUS.md` change is authorized by this design.

## Determinism and parallel execution

Candidate identity and ordering derive only from stable digests over the
scientific seed and persisted IDs/parameters; Python `hash()` and worker order
are forbidden. Base/trajectory work items are independent and may run through
Slurm CPU arrays or process workers. The final reducer canonicalizes accepted
entries before total-quota selection and atomic publication.

Repeated runs with the same inputs, configuration, code identity, and seed
must publish the same selected event IDs and semantic digest regardless of
worker count or completion order.

## Reporting and failure handling

Structured summaries report, at minimum:

- obstacle proposals, free-space passes, and useful-shadow passes;
- conflict anchors considered;
- snippet/anchor reachable-arc queries and blind-mask hits;
- chord broad-phase passes and rejections;
- exact 23-frame validations and rejection reasons;
- accepted mother events and accepted events per CPU-hour;
- unique base, trajectory, snippet, obstacle-type, bearing-bin,
  approach-side, and conflict-time counts; and
- per-variant attempts/accepts plus complete-sixpack count for the audit subset.

Every obstacle proposal, arc query, transform candidate, and exact validation
has a stable identity derived from persisted inputs and scheduled parameters.
Stage counts obey explicit conservation equations, so changing batching or
candidate granularity requires a new report version. CPU-hour includes index
construction, rejected and failed work items, exact validation, reducer work,
and allocated CPU wall time rather than only successful worker time.

Coarse candidates remain part of the evidence chain. The producer must not
raise its reported success rate by starting the denominator after broad-phase
filtering. It reports both root yield (`accepted mothers / obstacle proposals`)
and exact-candidate acceptance (`accepted mothers / exact validations`) beside
all intermediate conversion rates. Unexpected exceptions abort the work item;
only enumerated physical rejection reasons may continue sampling.

## Verification and acceptance

Implementation follows test-driven development and must include:

1. Unit fixtures for reachable-arc geometry, footprint-safe blind masks,
   crossing-angle constraints, and stable candidate identity.
2. Toy scenes proving a straight snippet can be placed behind left-, right-,
   and oblique-bearing occluders without intersecting them.
3. Curved-snippet cases where chord broad phase is insufficient and exact
   validation is authoritative.
4. Regression cases for obstacle/robot sweep overlap, target/obstacle contact,
   current partial visibility, non-emergence, context collision, and invalid
   time anchors.
5. Shape, dtype, NaN/Inf, current/history seam, time-scale, no-extrapolation,
   collision, and visibility invariants.
6. Determinism across repeated serial and parallel runs.
7. A Slurm real-data smoke benchmark over 10-100 outputs using the same trusted
   SOP03/SOP04 inputs as v4.
8. A same-budget v4/v5 comparison reporting accepted mothers per CPU-hour and
   every stage conversion rate. Full-scale generation does not start unless v5
   improves accepted-event throughput by at least 2x without violating any
   scientific invariant; 5x or greater is the performance target.
9. Five deterministic-random mother events sampled before pair-success
   conditioning, showing robot state, candidate trajectory, causal obstacle,
   blind/visible regions, source-derived target history/future, and collision
   timing. A separate complete-sixpack audit must be labeled as a conditional
   sample and cannot replace the random-mother review.

The target-scale run uses multiple Slurm CPUs, writes only to a new staging
directory, resumes only from verified checkpoints, and exposes output
atomically after formal-loader validation.

## Authoritative documentation changes after review

After this design is approved, update the directly conflicting sections in:

- `docs/event_centered_blind_spot_implementation_spec.md`;
- `docs/event_centered_blind_spot_agent_sops.md`; and
- `docs/parallel_acceleration_implementation_plan.md`.

Those updates must replace the v4 production semantics rather than describing
both algorithms as simultaneously valid. Downstream SOP06 sections must also
remove complete-sixpack coverage as a main-publication prerequisite while
retaining sparse paired coverage and the audit-sixpack protocol.
