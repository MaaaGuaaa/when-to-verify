# Authenticated Training Store and Single-Node Risk DDP Design

## Status and research purpose

The user approved this design on 2026-07-20. It optimizes the production
SOP08/SOP09 training framework without changing the scientific dataset,
model-input boundary, labels, split policy, model architecture, or formal
training gates.

Two measured implementation problems motivate the change:

1. the strict production loaders repeatedly decompress authenticated shards
   and repeat checksums, semantic validation, and per-sample collation during
   subset construction, every epoch, validation, and occupancy joins;
2. the production risk trainer accepts one device only and cannot distribute
   one model run across multiple GPUs.

The result is an engineering framework and smoke-tested execution path. This
task does not run scale training or claim a new scientific result.

## Approved scope

The implementation will:

- add an explicit `authenticated_snapshot` training store for risk and
  occupancy data;
- keep the existing strict loader APIs and their mutation-detection behavior
  unchanged;
- add single-node PyTorch DDP to production risk training;
- preserve the existing single-device path when `WORLD_SIZE=1`;
- keep production occupancy training single-device because its current
  trainer implements only `one_shard_smoke`;
- accept sample counts that are not divisible by GPU count without dropping,
  duplicating, padding, or inventing samples;
- add no dependency and leave `src/contracts.py`, `configs/base.yaml`,
  `DECISIONS.md`, `STATUS.md`, raw data, and accepted upstream publications
  unchanged.

Multi-node DDP, asynchronous prefetch, formal occupancy training, and scale
training are out of scope. The existing typed-family gate for `formal_50k`
remains in force and must not be bypassed.

## Alternatives considered

### Selected: one strict authentication followed by a node-local snapshot

The training job strictly authenticates the source publication once. During
that same pass it atomically materializes fixed training tensors into a
digest-keyed, non-compressed, read-only node-local store. Later epochs read
the tensors through NumPy memory maps and do not reopen compressed source
shards or repeat semantic validation.

This removes the repeated decompression and validation while retaining an
explicit evidence boundary.

### Rejected: skip semantic validation but keep reading NPZ shards

This is a small change, but every epoch would still decompress the source
archives. It weakens the trust boundary for limited performance benefit.

### Rejected: retain the full decoded dataset in each process

An in-memory cache is simple but duplicates the decoded dataset in every DDP
rank. Its memory cost scales with GPU count and makes failures more likely.

## Authenticated training store

### Public boundary

A new context-managed API opens a production training store from an already
specified dataset seal, immutable collection, optional occupancy sidecar,
expected semantic digests, subset definition, and cache root. It separates
two layers: a decoded snapshot keyed only by the authenticated split
publication, and a lightweight training-view manifest keyed by subset,
ordering, and partition settings. Different seeds or subset sizes reuse the
same decoded tensors instead of copying or decompressing the split again.
The mode name is explicitly recorded as `authenticated_snapshot`; it is never
selected by silently changing the old strict iterator.

The old strict loader remains the audit/reference implementation. Entry-time
tampering continues to fail. Once a snapshot has been authenticated and
opened, later mutation of the source publication does not alter that running
job; the job is bound to the snapshot digest instead. This changed lifetime
semantics is recorded in training provenance.

### Snapshot contents and identity

Each cached risk shard contains contiguous, fixed-shape arrays for the
existing `RiskBatch` model inputs and targets, plus ordered sample IDs. While
building an occupancy snapshot, the consumer validates
`sample.metadata["provenance"]["trajectory_primitive"]` and the matching
`base_config_digest`, then calls the existing
`reconstruct_production_robot_endpoint_footprints()` helper to independently
reconstruct that sample's robot endpoint footprint masks. It compares those
masks bit-for-bit with the sidecar masks once and stores the reconstructed
masks under a separate `query_inputs/` namespace. It stores hidden future
occupancy only under a separate `targets/` namespace. The sidecar-provided
robot masks are never used directly as model query input, and oracle
occupancy remains outside `RiskSample.model_inputs` and every risk-model
batch.

The decoded snapshot identity binds at least:

- snapshot layout version;
- risk dataset manifest digest;
- ordered source shard semantic digests;
- occupancy sidecar collection digest when present;
- grid/channel specification and split;
- every cached file checksum and shape/dtype declaration.

The lightweight training-view identity separately binds the decoded snapshot
digest, exact subset membership digest, seed, split role, and distributed
partition specification. Train and validation always use independent decoded
snapshots and independent view digests; formal provenance records both.

Files are written to a unique staging directory. The complete marker is
written and fsynced last inside staging, then the entire directory is
atomically renamed without replacement. On `EEXIST`, the process strictly
authenticates and reuses the winning cache only if its complete identity is
the requested identity; otherwise it fails. Symlinks, partial snapshots,
unknown files, digest disagreement, wrong shapes/dtypes, NaN/Inf, and
overwrite attempts fail closed. Rebuilding uses a new staging directory; it
does not mutate an accepted cache in place.

The cache is runtime infrastructure, not a scientific upstream artifact and
is not committed. Production Slurm commands pass a cache root under shared
node-local scratch, normally `$SLURM_TMPDIR`. Resume may rebuild the cache from
the same authenticated source if the cache is absent.

### Decode and reuse rule

The strict seal pass exposes each already-authenticated decoded shard,
ordered sample IDs, and semantic digest to a snapshot consumer, so
authentication and materialization share one source decode. For occupancy,
the same coordinator passes that decoded risk shard to the sidecar consumer;
the sidecar validator must not call the risk loader again. For one node and
one job, each required source risk shard and sidecar shard is formally
decoded at most once while building the snapshot. All epochs, validation
passes, and ranks then reuse read-only memory maps.

The occupancy iterator joins a risk shard and sidecar shard once per snapshot
build. It does not repeat the current preflight, risk-stream reload, and
per-batch sidecar reload sequence.

## Single-node DDP runtime

### Launch and device selection

Production risk DDP uses `torchrun` environment variables. `RANK`,
`WORLD_SIZE`, and `LOCAL_RANK` must be either all present or all absent.
`WORLD_SIZE=1` follows the existing single-device behavior. A distributed
CUDA run uses NCCL and maps each process to `cuda:LOCAL_RANK`; a CPU test uses
Gloo. DDP rejects a hard-coded `cuda:N` and rejects multi-node topology in
this first version.

Distributed initialization happens before any CLI path opens a source seal or
compressed shard. Only local rank zero may call the source risk/sidecar
loaders and build or validate the node-local snapshot; other ranks must not
touch compressed source files. Rank zero broadcasts a bounded success/error
envelope containing the authenticated snapshot descriptor. On error, every
rank raises the same failure instead of waiting at a naked barrier. On
success, other ranks verify the small manifest/marker identity from that
descriptor and open the same read-only memory maps. Bulk cached-file hashes
are computed during the rank-zero build, or once by rank zero when reusing an
existing cache; they are not recomputed independently by every rank.

### Deterministic partition without divisibility

The existing stable seed/epoch ordering defines one global ordered sample
sequence. A deterministic partition plan divides this sequence into
contiguous rank ranges whose total sample counts differ by at most one.
No sample appears in two ranges, and the sorted union is exactly the frozen
subset.

The plan then divides every rank range into the same number of synchronous
microbatches. Local microbatch sizes may differ, especially at the tail, but
each rank performs the same number of forward/backward collectives. The plan
never pads, duplicates, drops, or creates samples. The selected sample count
must be at least `WORLD_SIZE`. Distributed mode also requires a per-rank batch
size of at least two so a non-divisible tail can be rebalanced without an
empty collective.

`batch_size` means the nominal maximum per-rank batch size. The partition
planner records the exact local microbatch sizes. A stable
`partition_spec_digest` binds the algorithm version, dataset/subset identity,
seed, world size, and batch/accumulation configuration without binding the
configured number of epochs. A separate `epoch_plan_digest` binds the
epoch-sensitive sample order and concrete microbatches. This permits a valid
resume that extends the epoch limit while still authenticating the current
epoch cursor. If an input cannot form non-empty synchronous microbatches under
the explicit gates, it fails with a specific configuration error rather than
changing data membership.

### Correct loss normalization

Unequal local batch sizes must not give each rank equal statistical weight.
For every gradient-accumulation window, ranks all-reduce the actual global
sample count before the window begins. The current risk loss is a local batch
mean, so each microbatch forms `local_loss_sum = losses["total"] *
local_real_sample_count` and backpropagates
`local_loss_sum * WORLD_SIZE / global_window_sample_count`. Every microbatch
in the window uses that same precomputed global denominator. Because DDP
averages gradients across ranks, the resulting gradient is the exact mean
over all real samples in that window.

`DDP.no_sync()` encloses both forward and backward for every non-terminal
accumulation microstep. Finite checks, quantile-crossing checks, sample counts,
loss sums, and validation metrics are reduced explicitly. The effective
global batch and every ragged tail count are recorded. A single-process
numeric reference test reconstructs the DDP synchronous schedule, rather
than comparing against the old contiguous `WORLD_SIZE=1` update order. No
parameter equivalence across different world sizes is claimed.

### Publication and resume

Rank zero alone creates staging directories, checkpoints, manifests, and
final publications. Other ranks perform collectives and return the broadcast
result without writing output files.

For `one_shard_smoke`, every rank consumes its portion of exactly the first
synchronous global microbatch and all ranks perform exactly one optimizer
step. Rank zero records the no-overlap union of actually consumed sample IDs,
its global count, and membership digest. The selected snapshot/view count and
the smaller smoke-consumed count remain distinct provenance fields.

Distributed resume uses a new training-state layout. It binds world size,
backend, partition-spec digest, current epoch-plan digest, train/validation
snapshot and view digests, and per-rank cursor and RNG state. Rank zero
authenticates the state and broadcasts each rank's portion. Resume with a
different world size, partition specification, or current epoch plan fails
explicitly. Increasing only the configured epoch limit remains legal.
Existing single-device artifacts remain loadable through their existing
layout; DDP state is not silently inserted into it.

Same seed, code, data, backend, hardware class, and world size must reproduce
the ordered membership and state lineage. Bitwise equivalence across a
different world size is not claimed.

## CLI and Slurm behavior

The risk training CLI gains explicit distributed and training-cache options;
it does not infer DDP merely from multiple visible GPUs. A typical production
smoke uses one Slurm node, two allocated GPUs, and `torchrun --nproc_per_node=2`.
Requesting multiple GPUs while launching one process remains a single-GPU run
and is reported as such.

For `formal_50k`, rank zero must authenticate the typed dataset family and
existing cross-split gate before decoded-cache materialization, model/optimizer
construction, or any training step. A raw JSON audit is not accepted as a
substitute. Rank zero broadcasts a gate failure to every rank.

The occupancy CLI may use the authenticated store in its existing smoke but
must reject `WORLD_SIZE>1` until a formal multi-epoch occupancy trainer and
resume contract exist.

No scale training is part of implementation verification.

## Error handling and provenance

The implementation fails before training on incomplete distributed
environment variables, unsupported multi-node topology, unavailable NCCL or
CUDA, rank/device mismatch, snapshot tampering, source/snapshot digest
disagreement, overlapping/missing partition membership, empty rank plans,
non-finite tensors, wrong labels, and resume incompatibility. Rank-zero setup
failures are broadcast before peers enter a wait. A training-time exception
causes the launcher to terminate the process group; every normal and error
path attempts `destroy_process_group()` in `finally`.

Training manifests record snapshot layout/digest, loader mode, source dataset
and sidecar digests, world size, backend, nominal per-rank batch size,
effective global counts, partition-spec and epoch-plan digests, rank
membership digests, code commit, and runtime environment. Absolute cache
paths do not contribute to scientific semantic identity.

## TDD and acceptance tests

Implementation proceeds test first. At minimum the test suite must prove:

1. entry-time source tampering is still rejected by the strict boundary;
2. two epochs formally decode each source shard at most once and never decode
   it again after the snapshot is complete;
3. strict and snapshot iterators produce identical ordered sample IDs,
   tensors, labels, cursor semantics, and deterministic replay;
4. occupancy snapshot construction loads each risk/sidecar shard once and
   passes the first decoded risk shard into the sidecar consumer without a
   hidden risk reload; independently reconstructed per-sample endpoint masks
   equal the sidecar evidence, query masks stay under `query_inputs`, and
   oracle targets remain absent from model inputs;
5. partial, corrupted, wrong-version, symlinked, or digest-mismatched caches
   fail or rebuild safely without overwriting an accepted cache;
6. CPU two-process Gloo partitions a non-divisible fixture with zero overlap,
   no missing or duplicate IDs, equal collective counts, and the expected
   ragged tail;
7. DDP gradient/loss updates match a single-process reconstruction of the same
   synchronous schedule within a stated numeric tolerance, including the
   batch-mean-to-local-sum conversion and a multi-microbatch accumulation
   window;
8. rank zero is the only writer and reduced metrics/counts are exact;
9. same-world-size interrupted/resumed DDP matches uninterrupted training,
   including a legal epoch-limit extension, while changed world size,
   partition specification, or current epoch plan is rejected;
10. the old single-device and strict-loader tests remain green;
11. occupancy DDP fails explicitly rather than pretending to run;
12. a minimal two-GPU Slurm smoke produces finite forward/loss/gradients and
    a reloadable rank-zero checkpoint;
13. `one_shard_smoke` records the exact no-overlap union consumed by all ranks
    and performs one global optimizer step;
14. `formal_50k` without an authenticated typed family still fails before
    cache materialization or training, and a raw JSON audit cannot satisfy the
    gate;
15. a rank-zero setup failure reaches every peer without a barrier hang, and
    process groups are destroyed on success and failure;
16. train and validation use independent snapshot/view digests and cannot be
    silently interchanged.

The verification sequence is focused unit tests, CPU/Gloo integration,
single-device regression tests, then one minimal two-GPU Slurm smoke. It does
not include formal 50k training.

## Expected implementation files

The implementation is expected to add or narrowly modify:

- `src/datasets/risk_training_store.py`;
- `src/datasets/risk_dataset_seal.py`;
- `src/datasets/risk_dataloader.py`;
- `src/training/distributed.py`;
- `src/training/risk_trainer.py`;
- `src/training/occupancy_trainer.py`;
- `scripts/05_train_occupancy_baseline.py`;
- `scripts/06_train_risk_model.py`;
- focused loader/store/DDP production tests;
- the parallel implementation plan only where commands and status need
  synchronization.

Any discovered need to modify a frozen contract, add a dependency, weaken a
formal gate, or change another agent's owned file is a blocker requiring user
approval.
