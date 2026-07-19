# SOP05 Global Total Selection Design

## Context

SOP05 currently treats the configured `60/30/10` event-type weights as a
hard publication quota. A real eight-pair smoke run generated 24 physically
valid events, but publication failed because only five matched the exact
`6 environment / 3 structural / 1 mixed` quota. The research priority is
instead to publish enough events that satisfy the collision, current
occlusion, future emergence, and physical-validity contracts. Event-type
diversity remains useful diagnostic information but is not an acceptance
condition.

## Goals and non-goals

The publication succeeds when it deterministically selects the requested
total number of valid events. Event type does not affect eligibility, ranking,
or success. Generated and selected counts by type remain in structured
reports.

The global-selection change itself does not alter event geometry, target
transplantation, visibility, generator proposal weights, or upstream
SOP03/SOP04 data. During final review, separately identified correctness bugs
required a frozen v4 generator boundary: programmatic occluders now use the
same continuous signed-clearance certification for robot, context, and target
full-motion sweeps; the complete producer also proves that the formal consumer
can load staging before publication. Adaptive proposal weighting and
early-stop generation remain out of scope.

## Frozen selection contract

The selector receives a seed, a positive `accepted_quota`, and all accepted
event entries. Every entry has exactly `generated_event_id` and `event_kind`.
It rejects duplicate IDs and unknown event kinds, ranks every entry by the
existing stable selection key derived from `seed + generated_event_id`, and
returns the first `accepted_quota` entries. The persisted shard uses canonical
lexicographic event-ID order, so the loader compares the same selected set in
that canonical order.

The selector has a new version token. There are no per-kind required counts,
minimums, fallbacks, or substitutions. If fewer than `accepted_quota` accepted
events exist, the run is `quota_unmet`; otherwise it is `complete`.

## Producer and evidence changes

The SOP05 producer and run identity advance to v4. The scientific
request records the total quota and the new selector version, but no
`required_event_kind_counts`. Pair reports keep their exact v2 accepted-entry
schema. The global summary records total generated and selected counts plus
generated/selected counts by event kind for audit only. It must not contain
per-kind deficits or imply that type counts determine completion.

The producer recomputes the selected IDs immediately before publication and
requires an exact match with the shard. Output remains atomic and tamper
evident. The configured `event_type_weights` continue to control proposal
generation only and remain bound into the generator configuration digest.

For a complete run, the producer writes the completion marker in staging and
then calls the same strict loader used by SOP06 with the newly computed
external digest and trusted run ID. Only a successful full consumer
round-trip may be atomically exposed; any failure removes staging.

A complete run also returns a versioned `publication_semantic_digest` through
the CLI/result handoff. This digest binds the run-manifest SHA-256, checksum-
manifest SHA-256, target-motion manifest digest, and full target-motion payload
semantic digest. The handoff value is a trust anchor outside the output
directory: downstream code must retain it separately and must not obtain the
expected value from the directory it is validating. In-directory checksums
prove internal consistency, but cannot by themselves prove authenticity after
an entire directory has been coherently rewritten.

## Loader and migration

The loader accepts only the new producer, run, selection, and summary contract.
It reconstructs every accepted entry from canonical pair reports, recomputes
the global top-N selection from the trusted scientific seed and total quota,
and compares event IDs, event kinds, pair identities, shard order, and global
counts. It also requires the externally handed-off
`publication_semantic_digest`, recomputes generated-event and world identity
from persisted lineage and target motion, and compares the full payload digest
transitively bound by that trusted value. Therefore recomputing checksums or
the completion marker cannot legitimize a different selected set when the
original handoff digest is retained.

The generated-event metadata persists `event_slot_index` so the event lineage
ID can be independently recomputed. The world ID continues to identify lineage
plus exact target motion; complete `OracleWorld` content, including static
occupancy, context objects, and occluders, is bound by the shard world/payload
semantic digests and the external publication anchor. A second self-referential
full-content world ID is deliberately not introduced.

Old hard-quota publications are rejected. The implementation will not support
both contracts or reinterpret an existing `quota_unmet` directory as complete;
a new output directory must be generated.

## Documentation changes

The authoritative implementation specification and SOP05 agent checklist will
state that `60/30/10` is the default proposal distribution, not a publication
quota. Publication acceptance is based only on the requested total count and
the unchanged scientific event invariants.

## Verification

TDD coverage will include:

- only structural events filling the total quota;
- two or three available event kinds without per-kind minimums;
- fewer total events producing `quota_unmet`;
- deterministic ranking independent of input and worker order;
- duplicate, unknown-kind, reordered, fake-ID, and fully resealed tampering;
- producer/loader run-identity and exact-schema drift;
- old generator versions, old paired-generator versions, and old multi-LOS
  placement versions being rejected even when internally resealed;
- continuous occluder clearance over all robot, context-object, and target
  history/future sweeps, including narrow between-frame contacts;
- expected temporal-transplant candidate rejections continuing the frozen
  offset schedule with low-cardinality audit reasons;
- a complete producer refusing to expose staging that the formal loader
  rejects;
- a real SOP03/SOP04 smoke publication of 10 events;
- shape, dtype, NaN/Inf, visibility, collision, lineage, and repeated-load
  determinism checks on the published shard.
