# SOP05 Target History-Visibility Strata Design

## Status

Approved from the paper-level requirement that hidden risk must cover both
stale observations and targets unseen in the finite model history. This design
replaces the ambiguous requirement that a target remain invisible throughout
the whole history window.

## Goal

Extend formal SOP05 environment collision-mother generation so one immutable
run can intentionally generate a controlled mixture of:

- `seen_then_occluded`: the target was visible in model history and is
  continuously hidden immediately before and at the current frame;
- `unseen_in_history_window`: the target is invisible in all eight model
  history frames.

The change must not generate a production-scale dataset. Unit fixtures and a
small deterministic event-generation smoke are sufficient for this change.

## Scientific Semantics

The history window is eight frames spanning `-1.4 ... 0.0 s`. Therefore the
second stratum is named `unseen_in_history_window`, not `never_seen`: the
generator cannot make a claim about observations before the serialized model
history.

For a boolean target visibility vector `v[0:8]`:

- `unseen_in_history_window` iff every value is false;
- `seen_then_occluded` iff at least one value is true and the configured final
  `min_trailing_hidden_frames` values are all false;
- every other pattern is `ineligible` for this formal producer version.

The production default is:

```yaml
target_history_visibility:
  policy_version: target_history_visibility_policy_v1
  min_trailing_hidden_frames: 2
  weights:
    seen_then_occluded: 0.8
    unseen_in_history_window: 0.2
```

Weights are strict requested composition, not soft sampling hints. Either
weight may be zero for focused diagnostics, but at least one must be positive.

## Generation Architecture

Add a focused `src/generation/history_visibility.py` module that owns:

- strict policy normalization and canonical serialization;
- visibility-vector classification;
- deterministic allocation of an integer event request across strata;
- count/deficit helpers used by pair and run summaries.

SOP05 continues to compute visibility from the real moving sensor, context
objects, target footprint, and accepted occluder. Classification happens only
after the existing physical, collision, current-hidden, and future-emergence
checks pass.

Each pair receives deterministic requested stratum counts derived from
`events_per_pair`, the policy weights, and the pair seed. Exact-valid candidates
are stored in separate bounded deterministic prefixes. Search stops only when
all pair-local stratum requests are met or the frozen proposal budget is
exhausted. A candidate from a stratum whose local quota is full does not fill a
different stratum.

The run-level selector partitions candidates by stratum and selects the exact
counts derived from `accepted_quota`. A short stratum is not backfilled from the
other stratum. Publication remains `quota_unmet` until both the total and every
requested stratum count are satisfied.

## Determinism

Integer allocation uses a deterministic stratified unit interval with a
stable digest-derived offset. It keeps allocation error within one event per
stratum, gives exact `8/2` for a ten-event `80/20` request, and allows
single-event pairs to distribute across strata over different pair seeds.

Candidate identity and ordering remain independent of worker completion order.
The normalized visibility policy is included in the generator semantic digest,
so changing weights or the trailing-hidden requirement changes run and event
identity.

## Provenance And Publication

Accepted event metadata records:

- `target_history_visibility_policy` and its digest;
- `target_history_visibility_regime`;
- `target_history_last_visible_index`, null for unseen targets;
- `target_history_trailing_hidden_frames`;
- `target_history_visibility_policy_version`.

Pair summaries record requested, exact-valid, accepted, and deficit counts for
both strata. Run summaries independently recompute generated, selected,
required, and deficit counts. Loaders recompute classification from the stored
eight-frame visibility vector and reject metadata drift.

This changes generator and publication semantics while preserving
`schema_version=3.0.0`: model inputs, labels, coordinate systems, and array
layouts are unchanged. The SOP05 generator/producer/report version tokens must
advance, and old publications must not be interpreted under the new policy.

## Paired Variants

For every non-empty SOP06 variant, target-history visibility is recomputed by
the existing renderer and must remain in the collision mother's stratum.
Candidates that cross strata are rejected with a stable reason. The existing
`empty_blind_spot` variant remains a target-removal negative but is not claimed
to preserve identical observed history for `seen_then_occluded`; this
limitation is recorded for subsequent counterfactual redesign.

This acceptance change advances the paired producer to
`independent_partial_pairs_v2` and the persisted group contract to
`sop06_partial_pair_group_v2`. SOP06 reads the normalized policy and digest
from the authenticated mother metadata rather than hard-coding the trailing
hidden threshold.

## Verification

Targeted tests must prove:

1. classification of unseen, seen-then-occluded, too-recent, malformed, and
   flickering histories;
2. deterministic `80/20` allocation and single-event seed diversity;
3. pair generation does not substitute unseen candidates for a requested seen
   quota;
4. a controlled fixture produces at least one real `seen_then_occluded`
   collision mother through the normal geometry/renderer boundary;
5. selector and run publication fail closed on a stratum deficit;
6. serial and parallel selection remain byte-identical;
7. loaders reject tampered regime metadata;
8. existing current-hidden, future-emergence, collision, geometry, and oracle
   isolation tests remain green.
