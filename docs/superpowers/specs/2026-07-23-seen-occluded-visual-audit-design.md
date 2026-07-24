# Real Seen-Then-Occluded Visual Audit Design

## Status

Approved for a three-sample, real-data visual audit. This is an audit-only
producer, not a training-data or production SOP05/SOP06 publication.

## Goal

Generate exactly three auditable THOR-derived collision mothers whose target
history regime is `seen_then_occluded`. Every selected mother must support a
complete six-position SOP06 paired group and produce:

- `event_replay.gif`;
- `paired_events.png`;
- `audit.json`.

The collection must make it possible to inspect whether the hidden target was
really visible earlier, is hidden now, emerges plausibly, collides as labeled,
and has not crossed any represented obstacle geometry.

## Empirical Search Correction

The initial implementation required a complete sixpack from the formal
independent partial-pair producer. A full schema-3 audit search over 512 stable
base/trajectory pairs and 199 real `seen_then_occluded` mothers found zero
complete groups. Only one mother had a valid `temporal_safe` variant, and all
of its spatial-safe candidates intersected the accepted occluder. Restoring the
formal 0.025 m / 3.0 m spatial grid and inspecting the full ranked candidate
set did not change that result.

This is a conditioning mismatch, not permission to relax labels. Formal SOP06
is intentionally allowed to publish partial groups, while this visual audit
requires complete groups. The audit therefore adds an explicitly audit-only
joint occluder search after the formal mother is generated:

1. retain the real schema-3 mother target motion, base state, candidate
   trajectory, source snippet, context actors, label thresholds, and temporal
   offset schedule;
2. consider mothers that already provide collision, near miss, irrelevant
   hidden, and empty-blind-spot variants; allow either a temporal-only gap or
   a temporal-plus-spatial gap, but require temporal failure to be caused by
   current visibility/original-occluder collision and any spatial gap to be
   exactly `target_occluder_collision`;
3. construct a synchronously safe temporal target from the same real snippet;
4. jointly place one replacement occluder that clears the complete robot,
   context, collision-target, and temporal-target sweeps;
5. recompute all eight history visibility frames and require the rebound mother
   and every non-empty paired variant to remain `seen_then_occluded`;
6. pass the rebound mother through the unchanged formal SOP06 paired generator,
   renderer, independent audit, checksum envelope, and atomic publisher.

The v4 implementation treats LOS normal coordinates in the robot frame. The
earlier prototype incorrectly added the conflict-point normal component a
second time, which made valid placements depend on whether that component was
near zero. Candidate centers first follow the original compact schedule. Only
after an unshifted center passes static, continuous-sweep, and visibility gates
does v4 refine that center along the occluder's long-axis interval that still
covers both current LOS intersections. Continuous collision checks retain the
same exact recursion; a conservative center-radius lower bound only skips exact
endpoint work when it already proves the complete interval safe.

The authenticated 512-pair prefix yielded two distinct complete base states.
The stable prefix was therefore extended without changing inputs or protocol;
rank 903 supplied a third, naturally complete SOP06 group. The formal audit run
uses `max_pairs=904` and still stops immediately after three distinct accepted
base states.

The replacement world records the source world ID, joint-audit algorithm and
config digests, selected temporal offset, complete rejection counts, and
`audit_only=true`. It is never a SOP05 training shard and must not be consumed
as production data.

## Non-Goals

- Do not generate a production or training shard.
- Do not reuse old visual-audit samples as current evidence.
- Do not use toy or synthetic motion in the final three samples.
- Do not add or train an occupancy model.
- Do not claim collision checks against geometry absent from THOR-MAGNI.

## Authoritative Inputs

The producer uses the current schema-3 contract and these read-only inputs:

- SOP03 root:
  `outputs/sop03_thor_motion_snippet_v2_schema3_47b3acd_v1`;
- SOP04 root:
  `outputs/sop04_canonical_trajectory_bank_9bf19d5_v2_schema3`;
- trusted SOP04 handoff digest:
  `a9a71835f9b87487b0f8051d0850d86ad9008d9348c96ba2af9ae4bc373a33ce`;
- split: `train`;
- base config: `configs/base.yaml`;
- generator config:
  `configs/generator_seen_occluded_visual_audit.yaml`;
- paired config: `configs/paired_variants_visual_audit.yaml`;
- root seed: `42`.

The producer validates the complete SOP03 and SOP04 checksum envelopes through
the existing input adapters before searching. Runtime manifests record the
actual config hashes, producer source identity, input evidence, and semantic
digests rather than trusting the names above.

`outputs/sop03_thor_motion_snippet_v2_4d164c3_v1` is explicitly ineligible: its
base-state publication is schema `2.0.0`. Likewise, the non-schema-3 SOP04 bank
is not accepted. The old visual collections may guide presentation style but
must never be referenced as source evidence because they predate the frozen
history-stratified generator semantics.

## Search Protocol

The audit producer reuses the existing formal boundaries instead of copying
generator logic:

1. `prepare_sop05_run()` authenticates inputs and builds the stable
   base-state/trajectory schedule.
2. Schedule entries are visited in rank order. `generate_events()` runs with
   an audit-only `100/0` history composition so it searches only the requested
   `seen_then_occluded` stratum. All physical and acceptance parameters are
   byte-equivalent to the normal generator config; the audit-only proposal
   prefix is bounded at 8 obstacles by 8 snippets rather than the production
   64 by 64.
3. Only mothers independently reclassified as `seen_then_occluded` are passed
   to paired generation. Their exact eight-frame vector must contain at least
   one visible frame and at least two trailing hidden frames.
4. The source snippet is joined by its authenticated snippet ID.
   `generate_paired_variants()` creates the six positions with a seed derived
   from the root seed and generated event ID.
5. A sample is selectable only when coverage is exactly
   `(True, True, True, True, True, True)`, strict-pair eligibility is true, and
   `render_sop06_complete_audit_group()` validates all six observations.
6. Selected event IDs and base-state IDs must be unique. Search stops as soon
   as the third complete group passes the independent audit checks.

The default hard bounds are 512 schedule pairs, one requested mother per pair,
512 assessed `seen_then_occluded` mothers, and eight rank-ordered CPU workers.
These are search bounds, not
publication quotas. The command retains only three complete groups and compact
attempt records; it does not serialize rejected worlds or a risk dataset.

Every visited schedule pair receives an attempt row. Every assessed mother
receives either `accepted` or one stable reason, including generator stratum
deficit, duplicate identity, partial paired coverage, paired-generation error,
formal-render error, or independent-audit failure. Missing variants preserve
the exact reason returned by SOP06. Candidates are never silently skipped.

The audit paired config keeps all six label thresholds, temporal offsets,
version tokens, and strict-completeness gates unchanged. It uses a bounded
0.1 m / 2.0 m spatial proposal grid instead of the production 0.025 m / 3.0 m
grid. This changes search coverage only; every accepted variant still passes
the same continuous geometry and label predicates.

## Components

### Audit Module

Add `src/evaluation/seen_occluded_visual_audit.py` with four focused layers:

- authenticated, bounded search over formal SOP05 inputs;
- independent reconstruction and checking of audit geometry;
- deterministic GIF and paired-panel rendering;
- atomic collection serialization and checksum verification.

The public layer accepts paths and scalar limits through an immutable request
object. Search, validation, rendering, and publication remain separately
testable. It may call existing public generator and renderer APIs, but it must
not depend on old audit artifacts or private temporary scripts.

### Command-Line Entry

Add `scripts/05_render_seen_occluded_visual_audit.py`. The CLI exposes all
input roots, trusted digest, configs, output path, seed, sample count, search
bounds, checksum workers, and Git executable. The sample count is fixed to
three for this collection version. Expected contract and insufficient-search
failures return a nonzero status with a structured JSON summary.

### Optional Visualization Dependencies

Rendering uses the already available Matplotlib and Pillow packages. Add a
`visual-audit` optional dependency group with their verified versions; do not
add ImageIO or alter training/runtime dependency groups. Pillow writes GIFs
directly, so no extra image stack is needed.

## `event_replay.gif`

The GIF has exactly 23 frames: eight real history frames from `-1.4 s` through
`0.0 s`, followed by fifteen oracle replay frames from `+0.2 s` through
`+3.0 s`. Dimensions, duration, loop behavior, and palette conversion are
fixed by the producer and recorded in `audit.json`.

All frames use the robot-current local metric frame, equal axis scaling, fixed
limits, and a stable legend. They show:

- the current robot footprint and its state at the replayed time;
- the candidate path as a thick teal line;
- represented static occupancy and the causal occluder in dark gray;
- visible space with a translucent cool layer;
- unobservable space with a translucent red layer;
- visible dynamic content with solid markers or lines;
- the transplanted target current pose;
- the target oracle future as a dashed magenta path.

For the first eight frames, visibility is recomputed at that frame's real robot
pose with the matching dynamic history and accepted occluder. This is what
makes the earlier visible frames and trailing hidden frames directly
inspectable. From `t=0` onward, the `t=0` visibility layer is frozen and the
frame is explicitly titled `oracle replay`; future visibility is never
presented as model input. Target states that were visible in history are solid,
while hidden states and oracle future motion use hollow markers or dashed
lines. The current candidate path remains thick in every frame.

## `paired_events.png`

The PNG is a fixed 2-by-3 panel in this order:

1. collision;
2. near miss;
3. temporal safe;
4. spatial safe;
5. irrelevant hidden object;
6. empty blind spot.

Every panel uses identical metric axes, visibility layer, static occupancy,
occluder geometry, robot state, and candidate path. Non-empty panels show the
variant target pose and dashed oracle future. Titles include the semantic label
and minimum signed clearance; temporal-safe also reports its time offset.
`empty_blind_spot` is labeled as target removal and is not described as an
identical-input counterfactual for a previously visible target.

The figure-level footer states that the skeleton is shared and that target
placement/timing/removal is the controlled difference. This allows a reviewer
to distinguish real risk geometry from a shortcut based only on occluder
shape.

## Independent Audit Record

Each `audit.json` contains canonical, finite JSON with at least:

- SOP03/SOP04 evidence identities and digests;
- config paths, hashes, normalized semantic digests, and producer source
  identity;
- schedule rank, pair seed, event ID, pair-group ID, base-state ID,
  trajectory ID, source snippet/object/recording/session IDs;
- the exact eight booleans, last-visible index, trailing-hidden count,
  recomputed regime, and policy digest;
- current-hidden, future continuous-emergence, and final-visibility results;
- represented static/occluder/context collision checks and target kinematics;
- all six variant transforms, signed-clearance sequences, minima, times of
  minima, visibility sequences, and label predicates;
- shared static occupancy, occluder, blind-spot, axes, and candidate-trajectory
  checks across panels;
- GIF frame count/timing and image format, dimensions, byte count, and SHA-256
  for both visual artifacts.

The audit recomputes checks from arrays and geometry; it does not merely copy
world metadata. A mismatch between stored metadata and recomputed facts is a
hard rejection.

THOR-MAGNI has no aligned facility occupancy map in this SOP03 artifact.
Therefore `audit.json` distinguishes `represented_geometry_passed` from
`unmodeled_floorplan_unknown`. Passing the audit proves no collision with the
accepted occluder, represented occupancy, or represented context actors; it
does not prove that an absent real wall or shelf was respected.

## Collection Publication

The requested output is:

`outputs/sop05_seen_then_occluded_visual_audit_20260723_v1/`

Its payload is:

```text
visual_audit_manifest.json
search_attempts.jsonl
artifact_checksums.sha256
sample_01_<event-prefix>/
  audit.json
  event_replay.gif
  paired_events.png
sample_02_<event-prefix>/
  audit.json
  event_replay.gif
  paired_events.png
sample_03_<event-prefix>/
  audit.json
  event_replay.gif
  paired_events.png
.audit-complete
```

The manifest binds the ordered sample IDs, attempt-log digest, all input and
protocol digests, diversity summary, per-sample artifact hashes, and status.
`artifact_checksums.sha256` then covers the manifest, attempt log, every audit,
and every image while excluding itself and the completion marker. Finally,
`.audit-complete` binds the SHA-256 of both the manifest and checksum file. This
ordering avoids checksum self-reference. Files are written to a sibling staging
directory, reloaded and verified, then atomically renamed. The producer refuses
to overwrite any existing output path.

`.audit-complete` is written only when exactly three samples pass every check
and all payload checksums verify. If the bounded search finds fewer than three,
the producer atomically publishes a status `insufficient_complete_samples`
diagnostic collection without the completion marker. It includes all stable
attempt reasons and any already completed sample audits, returns nonzero, and
cannot be mistaken for successful evidence.

## Verification

Add `tests/test_seen_occluded_visual_audit.py` and verify:

1. exact 23-frame history/future timing and frozen future visibility;
2. deterministic 2-by-3 ordering, common axes, and shared-skeleton checks;
3. rejection of malformed history, metadata drift, incomplete sixpacks,
   geometry overlap, wrong input schema, and output overwrite;
4. stable attempt ordering and reason codes under repeated bounded searches;
5. GIF/PNG dimensions, frame count, nonblank pixel content, and checksum
   binding rather than file-existence-only assertions;
6. absence of a completion marker for insufficient search;
7. one authenticated real-input smoke followed by the exact three-sample
   command;
8. independent reload of the final manifest, attempts, audits, images, and
   checksums.

After generation, inspect all three paired PNGs and representative history,
current, and future GIF frames. The manual review records, per sample, whether
the target is hidden now, emerges from a plausible represented location, has
clear collision/near-miss/safe differences, avoids represented geometry, and
looks mechanically repetitive. Visual impressions are reported separately
from the automated pass/fail fields.
