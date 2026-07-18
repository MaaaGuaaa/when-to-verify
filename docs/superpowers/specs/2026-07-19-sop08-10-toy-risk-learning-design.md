# SOP08--10 Toy-First Risk Learning Design

## Status

Approved by continuation of the user goal on 2026-07-19. The immediate
deliverable is a deterministic, contract-shaped toy pipeline for SOP08 through
SOP10. It is deliberately not a substitute for the missing G1 risk dataset.

## Problem

The repository has complete SOP03 and corrected SOP04 artifacts, but it does
not yet have a production SOP07 dataset publication. The committed SOP07
component branch can build and shard individual schema-3 `RiskSample` objects,
but it has no full-run entry point or dataset-level v2 manifest. Its shard
layout is named `risk_shard_npz_jsonl_v1`, while SOP09 requires old/v1 shards
to be rejected. The meaning of "v1" is therefore ambiguous and must not be
guessed downstream.

SOP08 additionally needs future dynamic occupancy labels and per-time robot
footprints. Neither is present in `RiskSample`. Putting oracle future arrays in
model inputs or metadata would violate the input/label boundary. SOP10 also
lacks a stable prediction-table contract and several real grouping fields.

At the same time, all model channels, shapes, labels, losses, calibration
algorithms, and most metrics can be implemented and verified with deterministic
toy data. The selected design makes that useful progress while failing closed
at every unavailable production boundary.

## Decision

Implement SOP08, SOP09, and SOP10 on a branch from clean `main` with two explicit
execution modes:

1. `toy`: supported now, deterministic, small-grid, and contract-shaped;
2. `production`: rejected until a dataset-level v2 manifest and required label
   sidecars are supplied.

The implementation may use PyTorch 2.0.1 already installed in the specified
environment. It does not modify `pyproject.toml`, install packages, or claim
that the undeclared runtime dependency is resolved. NumPy implementations are
used for calibration and metrics so SOP10 has no scikit-learn/SciPy dependency.

No code changes are made to `src/contracts.py`, `configs/base.yaml`, the three
authoritative SOP documents, `DECISIONS.md`, `STATUS.md`, SOP05--07 files, raw
data, legacy code, or published outputs.

## Frozen Toy Contract

Toy samples preserve the frozen schema semantics while using a smaller spatial
grid for fast CPU tests. A production module
`src/datasets/toy_risk_learning.py` constructs immutable `RiskSample` objects,
calls `validate_risk_sample`, then collates their real field names. Test
fixtures are thin wrappers over that module; CLIs never import from `tests/`.

The tensors are:

```text
schema_version: 3.0.0
bev_history:      float32 [B, 8, 2, H, W]
state_channels:   float32 [B, 9, H, W]
trajectory_channels: float32 [B, 4, H, W]
robot_state:      float32 [B, 2]
hidden_risk_occupancy: float32 [B, 15, H, W]  # labels only
robot_future_footprints: float32 [B, 15, H, W] # query geometry only
collision:        float32 [B]
severity:         float32 [B]
```

The two sidecars are joined to `RiskSample` only by ordered `sample_id` and a
toy dataset manifest digest. They are never inserted into a model-input mapping
or `RiskSample.metadata`. `hidden_risk_occupancy` contains only currently
hidden dynamic objects that participate in SOP07 hidden-risk GT; visible actors
are excluded. Whether production occupancy supervision should predict this
label or all future dynamic occupancy remains an explicit upstream contract
decision.

All 15 sidecar frames use the frozen endpoint time layout: frame `t` is at
`(t + 1) * 0.2 s`, from 0.2 through 3.0 s. Occupancy and robot footprint frame
`t` always aggregate at the same timestamp.

The channel names and ordering are imported from `src.contracts`; tests do not
invent a second order. Toy identities include split, recording, session,
source object, snippet, base state, pair group, and seed namespace. Train,
calibration, validation, and test have zero overlap for each identity set,
including pair groups. Session overlap is allowed only in a separately tested,
explicitly reported THÖR-policy example, never in the default toy publication.

Toy cases cover collision, near miss, temporal-safe, spatial/same-area-safe,
irrelevant-hidden, empty, and OOD. Labels are generated from simple moving
occupancy patterns rather than copied from the desired model output. Oracle
future occupancy is passed only to occupancy loss and evaluation code, never to
model `forward`.

## SOP08 Architecture

### Analytic baselines

`LastObservationHold` repeats the last observed dynamic occupancy for all 15
future steps. `AgeDecay` uses the current last-seen occupancy and normalized
age map:

```text
age_seconds = clipped_normalized_age * a_max_s
p_t = last_seen_occupancy * exp(-(age_seconds + (t + 1) * dt_s) / tau_s)
```

Both return finite float32 probabilities clipped to `[0, 1]`.

### Learned occupancy

A small ConvGRU consumes the eight two-channel history frames. It rolls forward
15 steps and returns logits/probabilities `[B,T,H,W]`. A separate learned risk
head implements B4 and consumes only predicted occupancy plus query geometry;
it never consumes occupancy labels.

### Aggregation

Both aggregation functions accept predicted occupancy and a per-time binary
robot-footprint mask with identical `[B,T,H,W]` shape. The mask is normalized to
boolean, so one `(batch,time,row,column)` element contributes at most once.
Time weights are fixed as `exp(-endpoint_time_s / sigma_time_s)`, with endpoint
times `0.2 ... 3.0 s`; all cells within a frame have equal weight.

- weighted sum: the normalized weighted sum
  `sum(p * mask * time_weight) / sum(mask * time_weight)`. An empty mask returns
  zero. This is the reported `[0,1]` score; there is no optional clipping mode;
- probabilistic union: `1 - product(1 - p)` over selected unique time/cells.

All-zero probabilities produce zero risk. Increasing one selected probability
cannot reduce either score. Hand-computed small-grid tests freeze semantics.
The selected hand score is sent unchanged into the common calibration
pipeline; both scores remain in metrics for audit.

### B1--B4 mapping

- B1: last-observation hold plus both hand aggregators;
- B2: age-decay plus both hand aggregators;
- B3: ConvGRU occupancy plus both hand aggregators;
- B4: the same ConvGRU plus a learned aggregator.

The learned aggregator receives per-time masked mean, maximum, and
probabilistic-union evidence, followed by a small MLP and sigmoid. It consumes
predicted occupancy and query geometry only. If B4 training is not verified,
the handoff must say that B1--B3 and the mandatory three baseline classes are
complete but B4 is incomplete; it may not claim B1--B4 closure.

## SOP09 Architecture

### Input validation

The dataloader validates schema version, exact frozen channel names/order,
shape, float32 dtype, finite values, split identity, and absence of forbidden
oracle keys before constructing a batch. A formal shard path is rejected with
a clear contract error until the v2 dataset manifest exists.

### Models

- R0 flattens the eight history frames into 16 spatial channels, concatenates
  the nine state and four trajectory channels, applies a small CNN and global
  pooling, then concatenates the two-element robot state before the MLP.
- R1 encodes each history frame with a shared spatial encoder and a ConvGRU,
  then fuses the temporal representation with current state, trajectory maps,
  and robot state.

Both return `quantiles [B,4]`, `collision_logits [B]`, and `p_collision [B]`.
Quantiles are non-crossing by construction without forcing Q95 to one:

```text
q0 = sigmoid(raw0)
qi = q(i-1) + (1 - q(i-1)) * sigmoid(raw_delta_i)
```

Tests require finite gradients, `Q50 <= Q80 <= Q90 <= Q95`, and examples where
Q95 remains strictly below one.

### Loss and checkpoint

The total loss is quantile pinball plus weighted collision BCE and optional
occupancy auxiliary loss. Pinball loss is independently hand-checked.

Checkpoint layout v2 has mode-specific provenance. A toy checkpoint binds
schema, ordered channel spec, model variant, config digest,
`toy_dataset_manifest_digest`, seed, model state, and inference parameters. It
must not populate a fake G1 digest. A future production checkpoint must instead
bind `g1_split_manifest_digest`, `risk_dataset_manifest_digest`,
dynamic-object config digest, and target-type policy digest. Toy and production
loaders mutually reject the other mode. Loading also rejects missing fields,
mismatches, legacy layout, or tampering. Reloading must reproduce the same
batch output exactly on CPU.

## SOP10 Architecture

### Prediction table

SOP10 consumes a validated prediction table rather than model internals. Each
row contains identities, labels, predictions, grouping fields, method ID, and
evidence digests. Label and grouping fields are never routed back to model
inputs. Toy writers produce this table deterministically; production readers
remain gated on the missing v2 manifest.

### Conformal calibration

For upper quantiles, residuals are `max(0, y - q)`. With `n` calibration rows
and target miscoverage `alpha`, use:

```text
k = min(n, ceil((n + 1) * (1 - alpha)))
q_cal = kth smallest residual, using one-based k
upper = clip(q + q_cal, 0, 1)
```

Only calibration rows fit residuals. Grouped calibration fits one dimension at
a time and falls back to the global correction when group size is below the
configured minimum, recording count and fallback reason. It does not combine
overlapping group corrections without a future contract decision.

### Metrics

Pure NumPy metrics include AUROC, trapezoidal AUPRC, separately named average
precision, Brier, clipped Bernoulli NLL, equal-width ECE, quantile coverage,
upper-bound tightness, false-safe count/rate, pairwise ordering accuracy, and
per-subset reports. Empty/single-class cases return `value: null` plus a reason,
never NaN or a fabricated zero.

Toy metric definitions are fixed and surfaced in every report:

- ECE uses 10 equal-width bins on `[0,1]`, with the final bin including 1;
- NLL clips probabilities to `[1e-7, 1 - 1e-7]`;
- false-safe uses `p_collision < 0.5`, with all true-collision rows as the
  denominator;
- upper-bound tightness reports mean nonnegative excess
  `max(0, calibrated_upper - severity)` and also the mean upper bound;
- pairwise ordering compares prediction in the direction of higher true
  severity, excludes true-severity ties, and reports eligible/missing counts;
- continuous toy groups use manifest-frozen bin edges: critical-area fraction
  `[0, .05, .20, 1]`, age seconds `[0, 1, 3, 5]`, and occupancy-density
  fraction `[0, .01, .05, 1]`.

No-object rows stay in collision and severity metrics; they are excluded only
from clearance aggregation. Pairwise metrics use only eligible pairs and report
missing/ineligible counts rather than assuming a complete six-pack.

## CLIs and Artifacts

The three CLIs support deterministic toy runs and write under a caller-provided
output directory through staging plus atomic rename:

1. occupancy training writes config snapshot, checkpoint, predictions, metrics,
   manifest, and checksums;
2. risk training writes the same plus R0/R1 comparison fields;
3. calibration/evaluation writes calibration artifact and structured metrics.

Every artifact records mode=`toy`, code commit when available, seed, channel
spec, schema, split identities/digests, configuration digest, counts, and file
checksums. A semantic digest covers arrays/predictions, configuration,
identities, and provenance while excluding timestamps, hostnames, job IDs, and
absolute output paths. Full-file checksums are stored separately and may differ
when runtime metadata differs. Existing outputs are never overwritten silently.

## Failure Handling

- Reject a production data root rather than interpreting layout v1.
- Reject missing/mismatched schema, channel order, digest, dtype, shape, finite,
  or split provenance before training.
- Reject oracle/future fields in model-input mappings recursively.
- Reject calibration/test strict identity overlap.
- Keep test labels out of calibration fitting and checkpoint selection.
- Emit structured reasons for undefined metrics and unavailable groups.
- Never lower scientific gates, filter the test distribution, or claim G2 from
  toy performance.

## Verification

1. Write failing unit tests before each implementation slice.
2. Run all Python tests through Slurm with the specified environment.
3. Hand-check aggregation, pinball loss, conformal rounding, AP/AUROC examples,
   no-object handling, and identity leakage.
4. Overfit 128 deterministic toy samples with R0 and the occupancy ConvGRU;
   require materially decreasing loss and zero quantile crossings.
5. Reload checkpoints and require identical CPU predictions.
6. Run an end-to-end toy train/calibrate/evaluate job twice and require stable
   semantic digests, while auditing full-file checksums separately.
7. Check shapes, float32 dtypes, finite values, probability bounds,
   determinism, oracle isolation, and split/source isolation.

Toy validation proves the software path, not the paper's G2 scientific gate.

## Contract Changes Requested for Production

1. Publish a dataset-level risk-shard v2 manifest and define whether legacy
   "v1" means schema 1 or the current schema-3 shard layout v1.
2. Publish label-only future occupancy and `[T,H,W]` robot-footprint sidecars
   bound to sample IDs and the dataset digest.
3. Freeze and propagate the SOP01/G1 split digest, channel spec, dynamic-object
   digest, target-policy digest, and ordered shard digests.
4. Freeze SOP10 false-safe, ECE, tightness, pairwise, group/bin, and OOD
   reporting definitions before formal evaluation.
5. Declare the PyTorch runtime dependency in project metadata after explicit
   approval.
6. Update package discovery: current `pyproject.toml` explicitly packages only
   `src` and `src.utils`, so new `src.models`, `src.evaluation`,
   `src.calibration`, and existing dataset modules are absent from built wheels.
   Passing repository tests does not prove installable-package completeness.

## Completion Boundary

This branch may truthfully report "SOP08--10 toy software closure" after all
tests and toy runs pass. It must not report SOP08, SOP09, SOP10, G1, or G2 as
production-complete until real v2 inputs, 1k/50k training, calibrated test
evaluation, and every authoritative gate are verified.
