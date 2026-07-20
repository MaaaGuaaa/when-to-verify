# SOP11–14 Verification-Value Toy-Exact / Real-Train-Smoke Design

> Status: approved design derived from the authoritative SOP documents. This
> document does not replace `src/contracts.py`, `configs/base.yaml`,
> `DECISIONS.md`, or the three authoritative SOP specifications.

## Goal

Implement the complete SOP11–14 verification-value path behind stable module
interfaces. Deterministic toy inputs remain the exact scientific oracle for
hand-checkable geometry, posterior, and value tests. The audited schema-3
SOP03/SOP05 train artifacts provide the 10–100-event real-input smoke path.
Both modes exercise the same geometry, posterior, value, dataset, model, and
metric code; neither toy results nor train-only smoke results are reported as
paper-scale evidence.

## Scope and delivery phases

The work is one data flow but three independently testable subprojects:

1. **SOP11–12 scientific core:** six verification actions, counterfactual
   observation, replanning, scenario-bank validation, exact/soft posterior,
   and net verification value `G*`.
2. **SOP13 dataset:** schema-3 `VerificationSample` construction, deterministic
   split-safe shards, manifests, checksums, leakage audits, and a toy CLI.
3. **SOP14 learning:** V0 concat CNN, dual heads, losses, baselines, metrics,
   deterministic training, checkpoint provenance, and toy overfit smoke.

The first milestone does not claim the SOP13 10,000-sample minimum, paper model
thresholds, validation/test performance, cross-split leakage proof, or G2/G3
completion. A real train smoke is now in scope, but the supplied SOP07 handoff
contains train only and explicitly marks global cross-split leakage
`NOT_PROVEN`.

## Frozen contracts

- `SCHEMA_VERSION == "3.0.0"`.
- Future pose index `k` represents `(k + 1) * future_dt`; current pose `q0` is
  never serialized in a future array.
- `VerificationSample` model inputs contain only deployment-available BEV,
  trajectory maps, expected-visible geometry, and the action vector.
- Oracle occupancy, post-verification occupancy, hidden object identity/type,
  world identity, `br_before`, `post_risk`, and `value_target` are never model
  input channels.
- `PostRisk` includes verification action cost exactly once and
  `G* = br_before - post_risk`.
- Every verification outcome replans from the post-action robot pose. It never
  appends the unused suffix of the nominal trajectory.
- Scenario-bank targets are simulator-defined empirical decision values, not
  strict Bayesian ground truth.

## Architecture and data flow

```text
BaseState + nominal LocalTrajectory + OracleWorld bank
    |
    +-- SOP11 action library --> feasible post-action pose/trace
    |                              |
    |                              +-- static-only ray cast --> FOV geometry input
    |                              +-- oracle label ray cast --> observation/signature
    |                              +-- anchored resampling --> replanned candidates
    |
    +-- SOP12 posterior + injected trajectory/world loss
             --> br_before, post_risk, G*, useful
                    |
                    +-- SOP13 VerificationSample shards + manifest
                              |
                              +-- SOP14 V0 CNN + baselines + metrics
```

SOP12 accepts a finite trajectory/world loss callable. The toy adapter uses
hand-enumerated losses. The real-train adapter uses the merged typed-footprint
risk implementation through that same callable boundary. This prevents a
second, divergent risk formula from being implemented inside the verification
module.

### Audited upstream input boundary

The dependency branch is integrated at
`263efb5ea6870a9c40a27abcfe93311efdc1f94e`; its prior audit is not repeated.
The verification adapter performs only downstream trust and join checks:

- SOP05 batch handoff:
  `outputs/sop05_schema3_405affe1_train_20k_v2/batch_complete_handoff.json`;
- SOP05 batch semantic digest:
  `b81cb428495b9275f218ddea9fd34f42675dfecbacccd0c2e771b957053f13e6`;
- SOP07 train collection handoff:
  `outputs/sop07_schema3_263efb5_train_from_sop05_20k_v1/collection_complete_handoff.json`;
- SOP07 collection semantic digest:
  `e5c7c02aeaba5e0889eff8714c3bdb12ae13b855869162161b66b35588926d51`.

For a deterministic event subset, the adapter reads the trusted per-shard
publication digest from the SOP05 batch handoff, calls
`load_complete_sop05_events`, joins each event's `base_state_id` through
`Sop03SplitInputs.load_pair`, and resolves its `trajectory_id` through the
strict SOP04 bank loader. Absolute paths are supplied explicitly at the CLI
boundary; paths embedded in producer runtime metadata are provenance only and
are never silently trusted. SOP07 risk shards remain an optional distribution
cross-check and provenance link; their flattened `RiskSample` payload is not
used to fabricate `OracleWorld`, counterfactual observations, or `G*`.

## SOP11 design

### Verification actions

`src/planning/verification_actions.py` defines an immutable
`VerificationAction` with `action_id`, `duration_s`, `delta_forward_m`, and
`delta_yaw_rad`. The canonical ordered library is:

| ID | duration | forward | yaw |
| --- | ---: | ---: | ---: |
| `yaw_left_10` | 0.4 s | 0.0 m | +10 deg |
| `yaw_right_10` | 0.4 s | 0.0 m | -10 deg |
| `yaw_left_20` | 0.7 s | 0.0 m | +20 deg |
| `yaw_right_20` | 0.7 s | 0.0 m | -20 deg |
| `forward_peek` | 0.8 s | 0.30 m | 0 deg |
| `stop_scan` | 0.6 s | 0.0 m | 0 deg |

The action vector is exactly
`float32[duration_s, delta_forward_m, delta_yaw_rad]`. Yaw actions rotate in
place and cannot create lateral translation. Forward motion follows the current
heading. Configuration validation rejects duplicate IDs, non-finite values,
non-positive durations, negative forward distance, unsupported simultaneous
forward/yaw motion, and values outside configured limits.

Action feasibility is checked along a trace sampled no coarser than
`future_dt`. The exact duration endpoint is always included. At each trace time,
the robot footprint is checked against static occupancy and every typed dynamic
object footprint. Dynamic trajectories are sampled at the same physical time;
the 0.7 s actions interpolate between the 0.6 s and 0.8 s future endpoints with
unwrapped yaw. Infeasible actions return stable rejection reasons and are not
silently saved.

### Counterfactual observation

`src/generation/counterfactual_verify.py` keeps two products separate:

1. `verification_fov_mask`: ray casting against static geometry only. This is
   expected-visible geometry and is safe as model input.
2. Label-side observation: ray casting against static geometry plus all typed
   dynamic footprints at the action completion time. It may be used only to
   calculate the posterior and `G*`.

The caller supplies the current visible mask and occlusion-age map because they
are rendered SOP06 state, not fields of `BaseState`. New visibility is
`post_visible & ~current_visible`. Rectangle occluders use an explicit
`kind/pose/length_m/width_m` representation; incomplete geometry is rejected.

The seven-dimensional observation signature is ordered as:

1. newly visible area;
2. newly visible area intersecting the nominal swept mask;
3. newly visible area intersecting the union of replanned swept masks;
4. number of newly visible occupied cells;
5. minimum newly visible occupied-cell distance to the local goal corridor;
6. whether newly visible dynamic occupancy is non-empty;
7. critical-region occlusion-age reduction.

It uses observed masks/cells only. It never contains object ID, object type,
footprint parameters, critical-object metadata, or world ID. Exact posterior
grouping uses a deterministic digest of observable visible-mask/occupancy
bytes; soft posterior uses the seven continuous features.

`SignatureNormalizer.fit` accepts train records only, records count/mean/std and
a stable digest, clamps zero standard deviation to one, and refuses fitting on
validation/test records. Validation and test only call `transform` with the
persisted train statistics.

### Replanning

`src/planning/replanning.py` stores the post-action pose explicitly and keeps
each sampled `LocalTrajectory` in the new robot-centric frame. The nominal
endpoint and direction are transformed into that frame as the task anchor.
Candidates are generated with the existing differential-drive sampler, scored
against the anchor, and filtered for static collision after transforming their
poses back to the pre-action frame. Stop/reject is retained.

The implicit candidate seed `q0` equals the post-action pose; serialized pose
index zero remains the endpoint at `future_dt`. Query maps are regenerated in
the post-action frame. No nominal pose suffix is copied into the new set.

## SOP12 design

### Scenario bank

`src/generation/scenario_bank.py` operates on `OracleWorld` and records an
explicit variant kind. Configuration provides composition presets for
`M=8/16/32`; the default `M=16` preset is exactly 1 current, 2 empty,
5 temporal, 4 spatial, 2 speed, and 2 irrelevant variants. A preset must sum to
`M` and use only the frozen variant kinds.

The validator proves:

- all worlds have identical current visible occupancy;
- differences occur only in unknown cells or future state;
- static occupancy is identical and finite;
- dynamic trajectory/spec keys align within each world;
- non-target objects are bit-identical across target variants;
- only declared empty/target-state variants may remove or change the target;
- split, source namespace, and deterministic seed namespace are consistent.

Toy banks include circle and rotated-rectangle objects. Stable hashes, not
Python `hash()`, derive all variant seeds.

### Posterior

`src/generation/observation_posterior.py` implements:

- exact discrete grouping: uniform probability over worlds with the same
  observable digest and zero elsewhere;
- soft posterior: row-wise softmax of negative squared distance between
  train-normalized signatures divided by `tau_o`.

Both return finite `float64[M, M]` matrices, reject invalid temperature and
non-finite signatures, and require every row to be non-negative and sum to one.

### Net verification value

`src/generation/verification_gt.py` is a pure orchestration layer:

1. Compute nominal execute loss in every world and
   `br_before = min(mean(execute_loss), reject_cost)`.
2. Simulate each world/action observation and regenerate candidates from the
   post-action pose.
3. For each observed world, apply its posterior row and select the lowest
   posterior expected candidate loss or reject.
4. Compute `post_risk = action_cost + mean(best_post_observation_risk)`.
5. Return `value = br_before - post_risk` and
   `useful = int(value > 0.0)`, matching the frozen `VerificationSample`
   validator. A decision margin may be applied later by the online policy, but
   it never changes the supervision label.

Action cost is computed from duration, distance, and yaw-in-degrees and added at
step 4 only. The result records posterior mode, bank size, temperature,
composition digest, and replanning evidence for audit, while those fields are
not exposed as model inputs.

## SOP13 design

`src/datasets/verification_dataset.py` converts each `(state, nominal)` group
into all six action samples. Input tensors are copied only from deployment-side
rendering, nominal query maps, static-only FOV geometry, and action vectors.
Targets and audit values are stored separately in the frozen
`VerificationSample` fields.

`src/datasets/verification_dataloader.py` provides deterministic index and batch
ordering without importing PyTorch. Shards are compressed NPZ files containing
numeric stacked arrays plus JSON metadata; object arrays and pickle are
forbidden. Writes use a staging directory and atomic rename, and existing
output directories are never overwritten.

The JSONL manifest includes schema 3.0.0, shard/index, split, group IDs, action
ID/vector, target object type, footprint kind, source object ID, source mode,
code/config/input digests, and seed namespace. A future publication mode
requires qualifying split/G2 evidence. Toy mode requires a deterministic
toy-fixture digest and stamps `scientific_status: toy_smoke_only`. The current
real mode binds both audited handoffs and stamps
`scientific_status: train_smoke_only`, so train-only evidence cannot masquerade
as a publication artifact.

The audit checks shape, float32, finite values, action ID/vector consistency,
group/split isolation, forbidden input tokens, shard checksums, sample counts,
action/blind/value distributions, and deterministic regeneration. A sampled
target is recomputed through SOP12 and compared within explicit floating-point
tolerance.

`scripts/08_generate_verification_dataset.py` requires explicit
`--input-mode toy` or `--input-mode sop05-train`; it never silently falls back
to toy. Real mode also requires explicit SOP03/SOP04 roots, the external SOP04
handoff digest, the SOP05 batch handoff, and the SOP07 collection handoff. The
first artifacts contain 10–100 schema-valid samples per smoke mode. Publication
generation remains disabled until validation/test inputs and cross-split proof
exist.

## SOP14 design

### Dependency and packaging

`pyproject.toml` gains an optional `verification` dependency group containing
`torch>=2.0,<3`. Core geometry/generation/dataset modules remain NumPy-only.
Setuptools package discovery is changed from the incomplete explicit two-package
list to include `src.*`, ensuring new generation/model/evaluation packages are
installed. No lockfile is introduced.

### Model

`src/models/verification_model.py` implements V0:

- flatten `K x history_channels` into spatial channels;
- concatenate history, state, trajectory, and one FOV channel;
- use a compact strided CNN and global average pooling;
- encode the three-element action vector with an MLP;
- fuse both embeddings into scalar `G_pred` and logit `useful_logit` heads.

The forward result has shape `[B]` for both outputs. No post-action observation
or target/audit field is accepted by the model API. Risk-encoder reuse is
outside the first V0 milestone.

`verification_loss` is Huber regression plus useful BCE and pairwise hinge
ranking. Ranking pairs are generated only between different actions sharing the
same `(base_state_id, nominal_trajectory_id)` group, with target ties excluded.

### Baselines, metrics, and training

`src/evaluation/verification_baselines.py` implements visible area, critical
swept coverage, and occupancy-entropy scores from legal inputs only.

`src/evaluation/verification_metrics.py` implements useful F1, MSE, Huber,
Spearman, Kendall, pairwise accuracy, top-1 regret, top-two selection rate, and
action/blind/object/footprint slices with documented empty/tie behavior. NumPy
implementations avoid undeclared SciPy/sklearn runtime dependencies.

`scripts/09_train_verification_model.py` controls seeds, train-only fitting,
optimizer, checkpointing, and structured metrics. Checkpoints bind schema 3.0.0,
model version, input manifest digest, split digests, configuration, code commit,
and seed. Loading rejects missing provenance, schema mismatch, or legacy model
versions.

Toy smoke must show finite forward/loss/gradients and deliberate overfitting of
the deterministic toy set. It is not evaluated against the paper F1/ranking
thresholds. Those thresholds are applied only after a qualifying production
dataset exists.

## Configuration

Three task-owned YAML files are added without modifying `configs/base.yaml`:

- `configs/verification_actions.yaml`: action library, sensor FOV/range, and
  feasibility sampling parameters;
- `configs/verification_gt.yaml`: M presets, posterior mode/temperature,
  signature settings, reject cost, and shard settings;
- `configs/verify_model.yaml`: V0 channels, loss weights, optimizer, epochs,
  batch size, and deterministic seed.

Each owning module has a strict loader that rejects unknown keys and validates
numeric ranges. The base configuration is loaded through the existing central
loader and remains frozen.

## Error handling and determinism

- Schema/version/digest mismatch is fatal; no compatibility shim or implicit
  upgrade is provided.
- Non-finite arrays, invalid shapes, inconsistent world keys, visible-world
  disagreement, split leakage, invalid posterior rows, empty candidate sets,
  and duplicate sample IDs produce explicit exceptions.
- Empty candidate sets still expose reject as the defined decision alternative;
  they are recorded rather than silently discarded.
- All randomness uses explicit NumPy/PyTorch generators derived from stable
  seed namespaces. Serial and parallel toy generation must be bit-identical.
- Importing a module performs no writes, training, external calls, or global
  random-state mutation.

## Test and audit strategy

Every production behavior is introduced through a failing test before code.
Required evidence includes:

### SOP11

- six exact action vectors and analytic endpoint poses;
- no yaw-action lateral motion;
- dynamic/static feasibility with circle and rotated rectangle objects;
- static-only FOV mask unchanged when hidden oracle occupancy changes;
- critical action reveals the toy conflict while an irrelevant action does not;
- seven-feature signature excludes oracle metadata;
- normalizer refuses non-train fitting;
- replanned implicit seed equals post-action pose and nominal suffix is absent.

### SOP12

- M presets and composition counts;
- visible occupancy consistency and target/non-target mutation guards;
- exact posterior hand grouping and soft row sums;
- hand-enumerated mixed circle/rectangle `G*`;
- action-cost monotonicity with exact decrement;
- critical observation beats irrelevant observation;
- empty blind-spot value is non-positive unless reject/task cost explains it;
- M/tau are configurable and deterministic.

### SOP13

- all six actions per toy group and no cross-split group;
- schema-valid sample round trip;
- forbidden-token/input isolation audit;
- action vector/ID, shape, dtype, finite, checksum, manifest, and distribution
  checks;
- deterministic 10–100 sample CLI smoke and sampled `G*` recomputation.
- real-train adapter rejects mismatched SOP03/SOP04/SOP05/SOP07 identities and
  deterministically selects 10–100 events without loading the 1.2 GB SOP07
  numeric collection;
- real-train smoke records both upstream semantic digests and remains marked
  `train_smoke_only`.

### SOP14

- model output shape and illegal-input API isolation;
- hand-calculated Huber/BCE/ranking directions;
- ranking pairs are group-local;
- hand-calculated F1, correlations, pairwise accuracy, and regret;
- legal-input baselines;
- finite gradients and deterministic toy overfit smoke;
- checkpoint manifest acceptance and legacy/mismatched rejection.

Final validation runs unit tests, contract/toy regression tests, the toy dataset
CLI, a 10–100-event real-train CLI smoke, a model training smoke, artifact
audits, shape/dtype/finite checks, input/oracle leakage scans, deterministic
reruns, and `git diff --check`. It does not rerun the upstream 256-shard audit
or the upstream 215-test suite.

## Ownership and integration

This branch owns only the SOP11–14 files named by the authoritative Agent SOP,
new package `__init__.py` files, the approved `pyproject.toml`
packaging/dependency change, this design, its implementation plan, and
corresponding tests. The audited SOP05–07 branch is merged as an immutable
dependency; this task does not edit its files. It also does not edit
`src/contracts.py`, `configs/base.yaml`, `DECISIONS.md`, `STATUS.md`, raw data,
legacy code, or other agents' worktrees.

Integration is performed through explicit adapters and trusted handoffs. Any
contract conflict stops the workflow and is reported instead of supporting two
interpretations. The known train-only and temporal-safe limitations are
propagated into downstream manifests rather than reinterpreted as ordinary safe
negatives or publication-ready split evidence.
