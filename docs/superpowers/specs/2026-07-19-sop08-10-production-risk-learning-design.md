# SOP08--10 Production Risk Learning Design

## Status

The user approved the production direction on 2026-07-19 after the completed
SOP07 train collection was reported. This document freezes the production
architecture to be implemented after written review. It extends, rather than
redefines, the committed toy-first design.

The trusted upstream train input is:

```text
collection root:
  outputs/sop07_schema3_263efb5_train_from_sop05_20k_v1
schema: 3.0.0
shard layout: risk_shard_npz_jsonl_v2
split: train
shards: 256
samples: 88,089
collection semantic digest:
  e5c7c02aeaba5e0889eff8714c3bdb12ae13b855869162161b66b35588926d51
handoff SHA-256:
  439a5e5a71b5d6245ef6b8988e057a08734d7ad833cf0ee139aeaa18c0bee388
```

The upstream full-run audit is accepted as prior evidence and is not repeated
by this design task.

## Goal and Completion Boundary

Build the production data, training, calibration, and evaluation paths needed
to run SOP08 through SOP10 on real schema-3 risk samples while preserving the
frozen input/label boundary and split policy.

The implementation is intentionally staged:

1. a real train-only shard smoke and 1,000-sample overfit may run once the
   dataset seal, streaming loader, and production trainer pass;
2. formal 50k R0/R1 training requires an independent validation split and a
   proven train/validation identity audit;
3. formal SOP08 B1--B4 requires the occupancy/query sidecar described below;
4. SOP10 calibration and test evaluation require independent calibration and
   test publications plus a four-split leakage audit.

No train subset may be renamed or randomly partitioned into validation,
calibration, or test. A software smoke is not a G2 scientific result.

## Alternatives Considered

### Selected: dataset seal, immutable shards, and companion sidecars

Keep the 256 authenticated SOP07 shards unchanged. Publish a small
dataset-level `risk_dataset_v2` seal that binds the collection and trusted
provenance. Stream one verified shard at a time. Publish future occupancy,
time-indexed robot footprints, and evaluation-only fields in a separate
immutable sidecar collection keyed by `sample_id`.

This preserves upstream evidence, avoids rewriting 1.2 GB of compressed data,
keeps oracle arrays out of model inputs, and lets SOP09 progress before the
SOP08 sidecar replay finishes.

### Rejected: concatenate all shards into one physical dataset

Physical concatenation would duplicate an already authenticated publication,
create a new large failure surface, and encourage full-memory loading. The
uncompressed model inputs are approximately 2.83 MiB per sample, so 50k
samples are approximately 138 GiB before model activations.

### Rejected: let each downstream script interpret shard roots directly

The shard contract proves each local payload, but it does not by itself bind a
dataset split, ordered shard set, G1 split digest, dynamic-object policy, or
target-type policy. Allowing every script to guess those fields would create
multiple incompatible production contracts.

## Repository and Branch Strategy

Implementation occurs in a new linked worktree on
`feat/sop-08-10-production`, created from current `main`. The integration
branch brings in, in order:

1. `feat/sop-05-06-07-v5-integration` at
   `263efb5ea6870a9c40a27abcfe93311efdc1f94e`;
2. `feat/sop-08-10-risk-learning`, including the toy closure and this design.

The known authoritative-document conflict must retain the frozen future
endpoint interpretation `0.2 ... 3.0 s`; it must not restore a current-time
future frame. The implementation branch is not merged to main or pushed until
the user reviews the final handoff.

The implementation does not modify `src/contracts.py`, `configs/base.yaml`,
`DECISIONS.md`, `STATUS.md`, raw data, legacy assets, or the existing SOP07
collection. It adds no dependency and does not modify `pyproject.toml` without
separate approval.

## Production Dataset-Level Seal

### Relationship between layouts

`risk_shard_npz_jsonl_v2` and `risk_dataset_v2` are different hierarchy
levels, not competing schema versions:

- `risk_shard_npz_jsonl_v2` authenticates one shard containing
  `samples.npz`, `metadata.jsonl`, and `summary.json`;
- `risk_dataset_v2` authenticates one ordered collection of those shards for
  exactly one split.

The loader accepts this single hierarchy only. It does not add fallback
support for legacy v1 or ambiguous layouts.

### Seal contents

A dataset seal is a small, immutable publication containing
`dataset_manifest.json`, `checksums.sha256`, and `.producer-complete`. Its
manifest includes:

```text
dataset_layout_version: risk_dataset_v2
schema_version: 3.0.0
split: train | validation | calibration | test
collection_handoff_version: sop07_collection_complete_handoff_v1
collection_handoff_sha256
collection_semantic_digest_sha256
sample_count
shard_count
ordered shard index, sample count, manifest digest, semantic digest,
  payload SHA-256, metadata SHA-256, and summary SHA-256
grid height/width/resolution/history_steps/future_steps/sample_dt_s
ordered input and target channel specifications
g1_split_manifest_digest
dynamic_objects_config_digest
target_type_policy_digest
source SOP03/SOP04/SOP05/SOP07 commits and publication digests
risk_dataset_manifest_digest_sha256
```

The manifest semantic digest uses canonical JSON, excludes its own digest
field and runtime-only information, and is domain-separated. Absolute paths,
timestamps, hostnames, Slurm IDs, and output locations do not affect it.

The seal command receives the seal root and collection root separately. The
seal stores no machine-specific absolute path. At load time the caller again
supplies both roots; the loader requires the supplied collection handoff and
ordered shards to match the seal exactly.

The publisher refuses missing provenance, symlinks, duplicate/missing shard
indices, non-contiguous indices, digest disagreement, non-complete handoffs,
split disagreement, and attempts to overwrite an existing output. Formal seal
publication streams all shards through the already audited SOP07
`load_risk_shard()` implementation before the atomic rename.

### Dataset family seal

A later `risk_dataset_family_v1` seal binds exactly four independently
published `risk_dataset_v2` split digests and a structured cross-split audit.
The frozen THOR policy permits known session/day overlap but requires zero
recording, source recording, source snippet, pair-group, and seed-namespace
overlap wherever those identities are contractually split-bound.

Train-only smoke accepts a single-split seal and records
`global_cross_split_leakage=NOT_PROVEN`. Formal training requires train and
validation evidence. SOP10 requires the complete four-split family seal with
`global_cross_split_leakage=PROVEN`.

## Streaming Production Loader

The loader reuses SOP07 `load_risk_shard()`; it does not implement an
independent permissive NPZ parser. It validates the dataset seal before
returning any sample.

One process retains at most one decompressed shard and one mini-batch. Shards
are consumed in manifest order for evaluation. Training order is a stable,
seeded ordering derived from SHA-256 over dataset digest, seed, epoch, shard
index, and sample ID; Python `hash()` is forbidden.

The initial implementation uses a deterministic single-process iterator.
This is deliberate: multi-worker iterable interleaving can change sample
order. Later prefetch may be added only with an equivalence test that proves
identical ordered sample IDs and numeric results. The handoff resource limits
remain hard bounds, not defaults.

The loader supports:

- exact split selection;
- fixed `max_samples` selection with an ordered sample-ID digest;
- mini-batch collation without retaining prior shards;
- `float32` model inputs and targets, explicitly converting shard
  `risk_severity` and `min_clearance` from float64;
- recursive rejection of oracle/future keys from model-input mappings;
- deterministic epoch ordering and resumable `(epoch, shard, row)` state;
- collision, severity, clearance, and near-miss labels without inferring a
  label from `event_type`.

In particular, the 242 `temporal_safe` rows remain `collision_label=0` and
`near_miss=1` where published. They are not silently converted into ordinary
safe negatives.

## SOP09 Production Training

The existing R0/R1 model and loss definitions remain unchanged unless a
failing production test proves a model-side defect. Production training gets
a separate control path; it does not fall through into toy dataset creation or
the toy full-batch trainer.

The production trainer supports GPU device selection, batch size, epochs,
gradient accumulation, deterministic seed, maximum sample count, checkpoint
interval, exact resume state, and atomic artifacts. Training and all Python
tests run through Slurm. A training job must request an actual GPU resource,
not merely use a partition named `gpu`.

### Training gates

1. **One-shard smoke:** finite forward loss and gradients; output shapes
   `quantiles [B,4]` and `p_collision [B]`; no quantile crossing.
2. **Real 1k overfit:** fixed sample-ID digest; materially decreasing training
   loss; finite metrics; zero quantile crossing; checkpoint reload reproduces
   one CPU batch; changing trajectory query produces a measurable output
   change. This is an engineering gate only.
3. **Formal 50k R0/R1:** requires an independent validation seal and proven
   train/validation isolation. Best checkpoint selection uses validation only.
   Test is never loaded by training or checkpoint selection.

Checkpoints bind schema, ordered channels, model variant, seed, full training
configuration, dataset/family digests, G1 split digest, dynamic-object digest,
target-policy digest, exact training subset digest, and model state digest.
Toy and production checkpoints continue to reject each other.

The current 88,089 train rows satisfy the SOP09 quantity threshold. The first
formal run uses a fixed 50k subset for the required R0/R1 comparison; the full
train set may be used only after that gate is reproducible.

## SOP08 Companion Sidecar

### Required arrays

SOP08 cannot reconstruct oracle future occupancy from the published
`RiskSample`. A new immutable `risk_occupancy_sidecar_v1` publication contains,
per sample:

```text
sample_id
hidden_risk_occupancy: float32 [15,160,160]
robot_future_footprints: float32 [15,160,160]
future_endpoint_times_s: float32 [15] = 0.2 ... 3.0
```

The hidden occupancy includes only hidden dynamic objects participating in the
frozen SOP07 risk target. The robot footprint is trajectory query geometry,
not future sensor information, but it remains outside the main model-input
mapping so the baseline interface is explicit.

Each sidecar shard binds the corresponding risk-shard semantic digest and
ordered sample IDs. A sidecar collection seal binds the base
`risk_dataset_manifest_digest_sha256`, all sidecar shard digests, array layout,
and semantic definition. A join fails on any missing, duplicate, reordered, or
extra sample ID.

### Replay of the accepted train collection

The sidecar producer runs inside the SOP06/07 assembly path while oracle world
state and robot future poses are still available. It returns the public
`RiskSample` and a separate sidecar record; oracle arrays are never written to
`RiskSample.metadata` or model inputs.

To attach sidecars to the accepted 88,089-row collection, replay the exact
SOP05 shard, config, seed, partial-retry count, and deterministic ordering.
Publication is allowed only if every regenerated risk-shard sample ID and
semantic digest equals the already accepted shard. A mismatch fails closed and
leaves the existing collection untouched. Sidecars are written to a new
versioned root through staging and atomic rename.

### SOP08 execution

B1 and B2 consume observed history plus the time-indexed robot footprint. B3
trains ConvGRU occupancy against `hidden_risk_occupancy`; B4 consumes predicted
occupancy and query geometry only. Oracle occupancy is used only by occupancy
loss and offline metrics.

The same streaming, subset, checkpoint, and split rules used by SOP09 apply.
A real 1k occupancy overfit precedes full train diagnostics. Formal baseline
comparison waits for validation/calibration/test and uses exactly the same
dataset family and calibration protocol as the main risk model.

## Production Prediction and Evaluation Metadata

SOP10 prediction tables must not reuse toy manifest keys. A production row is
formed by joining model predictions to authenticated sample identity and a
separate evaluation-only metadata record. The record freezes:

- blind-spot type;
- critical-area fraction;
- age in seconds;
- occupancy-density fraction;
- critical object ID/type;
- target and robot footprint kind/dimensions;
- OOD tag and its rule version;
- pair eligibility and its rule version;
- event type, pair group, recording/session/source identities;
- collision, near-miss, severity, clearance, and first-collision labels.

Fields absent from the current risk metadata must be computed while their
authoritative source is available and published in an immutable companion
record. They must not be filled with constants or inferred differently by the
main model and baseline scripts. `empty_blind_spot` is the canonical no-object
event name; metric normalization must recognize it explicitly.

## SOP10 Production Calibration and Evaluation

Production prediction tables bind method/checkpoint digest, dataset split
digest, family digest, ordered cohort digest, channel specification, all G1 and
configuration provenance, and exact row semantics.

Calibration fits only the calibration split. Validation may select model
checkpoint and fixed hyperparameters but may not fit conformal residuals. Test
is loaded only after the checkpoint and calibration artifact are sealed.

The existing conformal and metric definitions remain frozen. Production mode
adds:

- validation of production prediction rows and provenance;
- calibration/test identity isolation against the family seal;
- global and one-dimension-at-a-time grouped calibration;
- comparison of the risk model and occupancy baseline on identical test
  cohort IDs and calibration protocol;
- structured undefined/fallback reasons for sparse groups;
- a G2 report that records failures without filtering the test distribution.

The scarce temporal-safe subset and 216 complete six-packs do not block the
binary collision smoke. They do limit temporal hard-negative and strict paired
claims. Those claims remain unavailable until supplemental data is produced
and the final split reports contain adequate counts.

The current human/circle target coverage permits a human-target result only.
Cart/carrier or general dynamic-object claims require separately generated,
split-safe target coverage and cannot be inferred from background-object
presence.

## Failure Handling and Atomicity

- Reject missing/mismatched layouts, schema, channels, split, digest, dtype,
  shape, finite values, or provenance before yielding a batch.
- Reject symlinks and absolute machine-specific paths in publications.
- Reject partial collections and overwrite attempts.
- Keep sidecars and evaluation-only fields outside every model-input mapping.
- Stop on sidecar/sample mismatch rather than dropping rows.
- Do not catch broad exceptions and continue with fewer samples.
- Checkpoint/resume records exact dataset, subset, epoch, shard, and row state.
- Write checkpoints and publications through staging followed by atomic rename
  or atomic file replacement.
- Record structured rejection and unavailable-gate reasons.

## Testing and Slurm Verification

Every behavior change follows red-green-refactor. Unit tests use small formal
fixture publications with real NPZ/JSONL loaders; they do not merely mock file
existence.

The verification ladder is:

1. unit tests for manifest canonicalization, digest tampering, layout/split
   rejection, deterministic subset/order, streaming release, float32
   conversion, and oracle isolation;
2. fixture integration for one dataset seal, risk shard, sidecar shard, and
   dataset family audit;
3. one real SOP07 shard loader smoke;
4. GPU forward/backward smoke;
5. real deterministic 1k SOP09 overfit and checkpoint replay;
6. real deterministic 1k SOP08 occupancy overfit after sidecars exist;
7. formal 50k R0/R1 only after validation publication;
8. calibration/test G2 only after the four-split family seal.

All Python tests, generation, and training execute through Slurm using:

```text
/home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI/.conda-envs/sop4-risk/bin/python
```

Tests and real smoke checks verify shape, dtype, finite values, probability
bounds, deterministic sample IDs/digests, gradient finiteness, checkpoint
replay, scientific invariants, input/label isolation, and split/source
isolation. CPU-heavy work is not run on the login/current node.

## Delivery and Commit Boundaries

Implementation is split into independently reviewable local commits:

1. integrate upstream branches and freeze production publication tests;
2. add dataset/family seals and streaming loader;
3. add SOP09 production trainer and real smoke/1k artifacts;
4. add SOP08 sidecar publisher/loader and production baseline path;
5. add SOP10 production prediction/calibration/evaluation path;
6. add final documentation and handoff evidence.

Only exact owned source, test, config, script, and design files are staged;
`git add .` is forbidden. Generated outputs, Slurm logs, caches, and temporary
scripts are not committed. No merge to main and no push occur without explicit
user direction.

## Contract Changes Requested

No conflict exists in `src/contracts.py`; the core schema-3 `RiskSample`
tensors match SOP09. Production implementation requires additive publication
contracts outside `RiskSample`:

1. `risk_dataset_v2` and `risk_dataset_family_v1` seals;
2. `risk_occupancy_sidecar_v1`, bound to the base dataset digest;
3. an evaluation-only metadata record for production grouping fields.

If trusted G1, dynamic-object, or target-policy digests cannot be recovered and
verified from current authoritative publications, sealing stops and the
missing anchor is reported instead of inventing a value.
