# SOP05 Result-Equivalent Throughput Optimization Design

## Objective

Make the frozen SOP05 producer capable of generating enough valid hidden-event
mothers for downstream SOP06/SOP07 work without changing the scientific sample
definition. The optimized producer must preserve the same pair schedule,
accepted event identities, collision and visibility checks, real SOP03 snippet
provenance, and global `sop05_diversity_total_selection_v1` result.

## Observed failure

The current worker expands a Cartesian product for every state/trajectory pair:

```text
64 occluder proposals
  × all split-matched human snippets
  × 7 conflict indices
  × 30 crossing directions
```

For the train split this can mean millions of Python-level transform candidates
for one accepted occluder. Three implementation details then make the search
unbounded in memory:

1. circular human footprints are yaw invariant, but the centre-mask cache uses
   the raw yaw and stores many identical 160×160 masks;
2. every candidate ID and every exact-accepted heavy payload is retained until
   the pair finishes;
3. `ProcessPoolExecutor.map` submits the complete pair schedule and the parent
   retains all restored reports and events before selection.

This explains the measured 19.7–26.5 GB peak RSS and 18–22 minute cancelled
5/20/100-pair jobs. Adding CPUs before bounding these structures would multiply
the memory pressure.

## Frozen scientific behavior

The first optimization stage is result-equivalent. It must not change:

- schema `3.0.0`, the 23-point snippet time axis, or the no-extrapolation rule;
- the SOP03 recording/split provenance or SOP04 canonical trajectory bank;
- occluder collision, line-of-sight, full-footprint hiding, emergence, target
  physics, same-index collision, or history-seam validation;
- deterministic seeds, proposal order, candidate identities, event/world IDs,
  event ordering, pair schedule, or the global diversity selector;
- shape, dtype, finite-value, occupancy, and stage-count conservation checks.

Runtime worker count and completion order remain non-scientific and must not
change selected records or publication digests.

## Result-equivalent architecture

### 1. Hoist immutable work

Compute trajectory tangent/normal and the ordered crossing directions once for
each eligible conflict index, rather than once per snippet and proposal. Compute
the canonical footprint specification/digest and fixed source anchors once per
snippet. Invalid source deltas are rejected before entering the 30-angle loop,
while their counters remain conserved.

### 2. Bound centre-mask memory

Use canonical yaw `0.0` for circle-footprint cache keys, matching
`FootprintCenterMask` itself. Scope the cache to one blind region/proposal so a
region that can never be reused does not survive into later proposals. Query
the candidate centre against `blind_free_mask` before constructing a full
footprint centre mask; failure is a necessary rejection and cannot remove a
valid event.

### 3. Bound accepted heavy payloads

Maintain only the lexicographically smallest `event_count` accepted candidates
under the existing key
`(proposal_id, candidate_id, transform_id)`. This is exactly equivalent to
sorting all accepted candidates and slicing. Construct `world_occupancy` only
when materializing those retained candidates.

### 4. Bound process scheduling

Replace eager ordered `executor.map` with an explicit bounded-in-flight
scheduler. At most `2 × workers` pair futures may be outstanding. Completed
items are buffered by rank and committed in canonical schedule order. The full
schedule is still processed, so the frozen global selection remains exact.
On error, outstanding futures are cancelled and the exception propagates.

### 5. Preserve auditable evidence

Stage counts and rejection counts remain exhaustive and conserved. Explicit
IDs required to validate accepted events remain present. If exhaustive rejected
candidate ID tuples are later replaced by count plus streaming digest, that is
a separately versioned report-format change; it is not silently folded into
this stage.

## Verification gates

Development follows TDD. Unit tests must first fail for:

- circular yaw variants constructing more than one centre mask per region;
- trajectory geometry being recomputed more than once per eligible index;
- accepted-candidate storage exceeding `event_count` or differing from full
  sort-and-slice;
- more than `2 × workers` pair futures outstanding;
- worker count/completion order changing selected IDs or scientific summaries.

After unit and toy-fixture tests, real-data jobs run only through Slurm in the
order 1 pair, 5 pairs, 20 pairs, then a 10–100 sample smoke target. Each run
records wall time, allocated CPU time, accepted count, rejection stages, and
maximum RSS. Expansion stops if shape/dtype/NaN/Inf, determinism, collision,
hidden-current/emergence, provenance, or selection invariants fail.

## Escalation boundary

Stopping once quota is reached, sampling only a snippet subset, coarsening
angles, or lowering proposal budgets would change the candidate population and
therefore can change the frozen hash/diversity selection. If result-equivalent
optimization is still too slow, the next stage must introduce an explicit
quota-first selection/producer version and update the authoritative SOP05
contract before use. It must not masquerade as a runtime-only optimization.

## Escalation decision after the real-data gate

The result-equivalent implementation reduced the real one-pair peak RSS from
roughly 20 GB to about 340 MB, but Slurm job 35394 still consumed 10:46 CPU over
6:17 elapsed time without finishing one pair. This fails the throughput goal,
so the versioned escalation is activated:

- generator algorithm: `blind_reachability_quota_first_v3`;
- producer: `sop05_generation_run_v6`;
- pair report: `sop05_pair_generation_report_v4`;
- deterministic pair-seeded ordering for conflict/snippet/side/angle;
- rotating `snippet_candidates_per_proposal=64` real-snippet window;
- stop the pair immediately after `requested_event_count` exact-valid events.

All physical validators and schema 3 tensors remain unchanged. The previous v2
producer is rejected rather than interpreted as v3.

## Measured result

The completed implementation also batches closed-cell footprint rasterization
with an exact scalar fallback at numerical boundaries, adds an endpoint
collision short-circuit, uses a conservative swept-AABB broadphase, and bounds
the sum of running plus rank-buffered pair reports. The rectangle raster path
matched the previous scalar authority on 149 seeded random/axis/boundary cases
and was approximately 131x faster in that microbenchmark.

On the same real train input and seed, one accepted pair fell from about 141 s
wall / 120 CPU-s to about 59 s wall / 34.7 CPU-s. Roughly 29--31 s is the
one-time input checksum/load cost. A 100-pair Slurm smoke with 16 workers
finished in 3:31.87, generated 95 valid events, and selected the requested 90.
The production scheduler should therefore reserve 5--10% extra pairs and must
retain quota-unmet reports for the residual failures.
