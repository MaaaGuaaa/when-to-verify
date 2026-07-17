# Recording-Generalization Provenance Design

## Status

Approved in conversation on 2026-07-17 for implementation planning. The
scientific evaluation target is unseen recordings within known THÖR-MAGNI
recording-day sessions; it is not unseen-session or unseen-participant
generalization.

## Problem

The existing frozen THÖR split assigns 52 recordings to
train/calibration/validation/test as 37/5/5/5. It contains recording IDs but no
session IDs. SOP-03 later derives the recording-day session from the official
recording identifier, after the split has already been frozen. As a result, the
five recording-day sessions occur in more than one split, while the old split
audit reports zero session overlap because there were no session values to
audit.

The numerical SOP-03 conversion is otherwise usable: recording trajectories,
typed dynamic objects, footprints, base states, oracle contexts, and snippets
passed their existing numerical and schema checks. The defect is the ambiguity
and incomplete reporting of the evaluation scope and split provenance.

## Scientific Decision

The formal evaluation scope is:

```text
evaluation_scope: unseen_recording_within_known_sessions
grouping_unit: recording_id
recording_overlap_policy: forbidden
session_overlap_policy: allowed_reported
participant_overlap_policy: unavailable
```

The current 37/5/5/5 recording assignment remains frozen. A recording may
occur in exactly one split. A recording-day session may occur in multiple
splits, but every such overlap must be enumerated in the audit. Stable
cross-recording participant identities are not available; reused helmet labels
must not be represented as stable participant IDs.

Paper claims and handoffs must describe the test split as unseen recordings
from known recording-day sessions. They must not claim generalization to unseen
days, sessions, or participants.

## Contract Changes

The default generic split behavior remains strict: recording, session, and
available participant overlaps are rejected. The THÖR production split opts
into the recording-generalization policy explicitly. This avoids weakening
other datasets or silently changing existing callers.

The split audit distinguishes three concepts:

1. field coverage: whether each required provenance field is populated;
2. detected overlap: every value present in more than one split;
3. disallowed overlap: the subset that violates the selected policy.

For the approved THÖR policy, missing `recording_id` or `session_id` is an
error. Recording overlap is an error. Session overlap is detected and reported
but is not a policy violation. Participant coverage is explicitly unavailable
and does not receive fabricated identifiers.

A passing report therefore contains the five detected session overlaps and
zero disallowed overlaps. An empty set of session values can never be reported
as a successful session audit.

## Components and Data Flow

### Metadata index

A lightweight THÖR metadata-index step reads only official recording metadata
needed before splitting: `recording_id`, recording-day `session_id`, and source
path. It does not parse or transform trajectory samples. The recording-day ID
comes from the official FILE_ID/date metadata and is cross-checked against the
filename used by the current adapter.

### Frozen recording assignment

The existing recording-to-split mapping is imported as an immutable assignment
and enriched with the metadata index. Validation rejects missing recordings,
extra recordings, duplicate assignments, unknown split names, source-path
mismatches, and missing sessions. The external name `validation` is normalized
to the internal name `val` before serialization.

The resulting canonical JSONL manifest has 52 rows and preserves the existing
37/5/5/5 assignment. It records the evaluation scope, overlap policy, stable
seed namespace, schema version, and a deterministic manifest digest.

### Split audit

The audit reports recording, session, participant, snippet, pair-group, and
seed-namespace coverage and overlap. Policy is explicit in the report and in
the split summary. The report passes only when all forbidden fields have zero
overlap, all required fields have complete coverage, and every allowed overlap
is enumerated deterministically.

### SOP-03 rebuild

SOP-03 is regenerated into a new versioned output directory without modifying
the old run. All four splits consume the enriched frozen manifest. Recording
indexes, base states, oracle contexts, and typed snippet libraries are rebuilt
with the same numerical configuration and eight Slurm CPUs.

Every SOP-03 summary and manifest records the new split digest and evaluation
scope. Because the recording assignment and numerical configuration remain
unchanged, accepted counts and arrays are expected to remain identical; this
is verified rather than assumed. Any mismatch is reported and investigated
before the new run is accepted.

### SOP-04 provenance

The SOP-04 canonical trajectory bank is independent of SOP-03 split contents.
It is regenerated under the combined code commit so that its code provenance
is current. Serial-versus-parallel determinism, query-map invariants, and
artifact checksums are revalidated. No state-specific split digest is attached
to the canonical bank.

## Files in Scope

Implementation planning may modify or create only the directly owned split,
THÖR metadata, documentation, and tests required by this design. It must not
modify `src/contracts.py`, `configs/base.yaml`, `DECISIONS.md`, `STATUS.md`, raw
`data/`, `legacy/`, dependency files, or another agent's SOP-05/06/07 files.

The authoritative implementation specification, parallel plan, and agent SOP
must be updated consistently to replace the unconditional THÖR session-isolate
claim with the explicit recording-generalization policy. The generic strict
split contract remains documented as the default.

## Failure Handling

- Refuse to overwrite an existing split or SOP output with different bytes.
- Write new artifacts through staging directories and atomic rename.
- Reject incomplete metadata coverage before any trajectory processing starts.
- Reject recording, snippet-source, pair-group, or seed-namespace overlap.
- Preserve and report the five allowed session overlaps.
- Stop on a contract/version mismatch; do not support two implicit formats.
- Record every rejected recording or sample with an explicit reason.

## Verification

Verification follows this order:

1. TDD regression tests demonstrate that a missing-session manifest fails and
   an explicitly allowed session overlap is reported without being rejected.
2. Existing strict-policy leakage tests continue to reject session overlap.
3. Toy metadata and split fixtures prove deterministic, byte-identical output.
4. A 10-recording Slurm smoke run checks schema, shape, dtype, finite values,
   deterministic IDs, policy propagation, and numerical parity.
5. The 52-recording Slurm rebuild checks 52/52 session coverage, 37/5/5/5 split
   counts, zero recording overlap, five reported allowed session overlaps, zero
   disallowed overlap, and a new deterministic split digest.
6. SOP-03 checks base-state/oracle alignment, observed-oracle separation,
   typed footprint semantics, source overlap, NaN/Inf, and accepted counts.
7. SOP-04 checks all trajectory/query-map invariants and serial/8-worker exact
   equality.
8. The repository test suite passes on Slurm before the implementation commit.

## Artifact Policy

The old SOP-03 and SOP-04 outputs remain read-only provenance. New artifacts
use distinct run IDs and include code commit, configuration digest, source
split digest, evaluation scope, policy, Slurm resources, counts, checksums, and
audit results. Generated outputs remain ignored by Git and are not committed.

## Known Limitation

This policy intentionally permits recording-day session overlap. It provides
no evidence for unseen-session/day or unseen-participant generalization. That
limitation must be carried into SOP-05 through SOP-16 manifests, evaluation
reports, and paper wording.
