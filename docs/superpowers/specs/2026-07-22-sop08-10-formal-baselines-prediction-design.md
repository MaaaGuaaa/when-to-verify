# SOP08--10 Formal Baselines and Shared Prediction Tables

## Status

This design was approved by the user on 2026-07-22.

It extends the authenticated SOP08--10 training work already present on
`main`. The server-side runtime patch
`when-to-verify-training-fixes-20260722.patch` is treated as a prerequisite
runtime fix and is verified separately; it is not part of this design's
scientific contract.

## Goal

Unblock the formal baseline comparison by delivering three connected
capabilities:

1. an authenticated, independent evaluation-record collection;
2. formal 50k B3/B4 occupancy training with validation-only checkpoint
   selection;
3. one prediction-table producer that emits the main risk models and B1--B4
   on exactly the same calibration and test cohorts and protocol.

The implementation must not change the frozen RiskSample model-input boundary,
labels, split policy, risk model architecture, existing toy behavior, strict
dataset loader semantics, or accepted raw publications.

## Non-negotiable invariants

- `train`, `val`, `calibration`, and `test` remain authenticated members of one
  `risk_dataset_family_v1` publication.
- Formal training selects from `train` and `val` only. It never opens
  `calibration` or `test`.
- Calibration artifacts fit only the calibration prediction table. Test rows
  are not read until the selected checkpoint and calibration artifact are
  sealed.
- Every prediction method receives the same ordered sample-ID cohort for a
  split. Missing, duplicate, reordered, or extra IDs fail closed.
- Evaluation records remain outside `RiskSample.metadata`, every model-input
  mapping, occupancy query inputs, and occupancy targets.
- The risk family digest, occupancy-sidecar digest, and evaluation-record
  collection digest are independent identities. Matching one never
  authenticates another.
- B1/B2 are deterministic baselines and do not receive invented checkpoint
  state. Their table binds a canonical baseline-spec digest instead.
- B3/B4 use the selected formal occupancy checkpoint. B3 is frozen while B4
  is trained and selected.
- All writes use a new staging root, exact file sets, checksums, a completion
  marker written last, and atomic no-replace publication.

## Evaluation-record collection

### Source and replay

The existing oracle/renderer boundary already returns aligned
`(RiskSample, RiskLabelSidecar, evaluation_record)` values through
`build_risk_samples_sidecars_and_evaluation_records_from_sop06_group()`.
The replay path will use that triple without attaching the evaluation record to
the public sample.

For an already accepted risk collection, replay must be pinned to the same
SOP03/SOP04/SOP05 references, config, paired configuration, seed, shard index,
and deterministic ordering. Before publishing records, it reauthenticates the
reference risk seal and verifies every replayed sample ID, risk label, ordered
shard membership, and risk semantic digest. A mismatch leaves the existing
collection untouched and aborts.

### Storage and API

Add `src/datasets/risk_evaluation_store.py` with a sibling collection layout
`risk_evaluation_record_collection_v1`:

```text
evaluation_manifest.json
shard-00000/records.jsonl
shard-00000/summary.json
...
checksums.sha256
.producer-complete
```

The manifest binds the split, reference risk dataset manifest digest, grid and
base-config identity, ordered shard descriptors, ordered sample-ID digest,
record layout/rule versions, row count, and collection semantic digest. Each
record is canonical JSON, validated by
`validate_production_evaluation_record()`, and joined to exactly one accepted
risk sample with matching labels. Unknown files, symlinks, duplicate IDs,
wrong order, missing rows, label drift, and digest drift are rejected.

The public API is:

```python
publish_risk_evaluation_collection(
    output_dir: str | Path,
    *,
    dataset: LoadedRiskDataset,
    records_by_shard: Mapping[int, Sequence[Mapping[str, object]]],
) -> Path

load_risk_evaluation_collection(
    root: str | Path,
    *,
    dataset: LoadedRiskDataset,
    expected_manifest_digest: str | None = None,
) -> LoadedRiskEvaluationCollection
```

The replay/publish CLI accepts explicit source roots and a reference seal;
there is no discovery of raw data or silent fallback to sample metadata.

## Formal B3/B4 training

### Inputs and gates

Extend the production occupancy trainer while preserving the current
`one_shard_smoke` and strict single-device path. `formal_50k` requires:

- an authenticated train and val dataset from the same typed family;
- `global_cross_split_leakage=PROVEN`;
- a train subset of exactly 50,000 samples selected by the existing stable
  subset rule;
- authenticated occupancy sidecars for both train and val;
- no calibration or test roots, cursors, or sample IDs in the training path.

Validation uses the complete authenticated `val` member in its published order.
Train-only occupancy and collision class weights are fitted once from the
selected train subset and reused for validation; validation statistics never
change training weights.

### Two-phase optimization and selection

1. **B3 phase:** train `ConvGRUOccupancyPredictor` over the train subset. At
   each configured epoch boundary, evaluate weighted occupancy BCE on the full
   val member. Keep the lowest finite validation loss; ties resolve to the
   earliest epoch and optimizer step.
2. **B4 phase:** restore the selected B3 state, freeze all B3 parameters, and
   train `LearnedOccupancyRiskAggregator` on B3 predictions and robot query
   footprints. At each epoch boundary, evaluate weighted collision BCE on val
   with B3 still frozen. Keep the lowest finite validation loss using the same
   tie rule.

The selected checkpoint contains the selected B3 and B4 states plus both
selection records. The final checkpoint contains the terminal states for
engineering diagnosis. B3 state digests before and after B4 must match.
Quantile/table prediction code loads only the selected checkpoint.

### Formal publication

Formal occupancy artifacts use an explicit formal layout version and include:

- `best_checkpoint.pt` and `final_checkpoint.pt`;
- training state/checkpoint interval artifacts sufficient for restart at an
  optimizer boundary;
- config snapshot and metrics with per-phase train/val loss histories;
- manifest, exact checksums, completion marker, and publication instance
  digest;
- provenance for family, train/val member, sidecar, subset, code, config,
  selected epochs/steps, and zero test usage.

`scientific_claim_eligible` is true only for an exact 50k train subset,
validated family isolation, finite metrics, and a valid selected checkpoint.
Smoke/overfit publications retain their existing engineering-only meaning.

## Shared prediction-table producer

### Public module and CLI

Add `src/evaluation/prediction_tables.py` and `scripts/09_predict_risk.py`.
The CLI loads the authenticated family members, calibration/test occupancy
sidecars, evaluation-record collections, formal R0/R1 checkpoints, and the
selected B3/B4 checkpoint. It does not train, fit calibration, or load test
before checkpoint/protocol validation succeeds.

The producer creates one immutable prediction publication containing:

```text
prediction_manifest.json
prediction_protocol.json
calibration/
  risk-r0.json  risk-r1.json  B1.json  B2.json  B3.json  B4.json
test/
  risk-r0.json  risk-r1.json  B1.json  B2.json  B3.json  B4.json
checksums.sha256
.producer-complete
```

The manifest records every table semantic digest, the common calibration/test
cohort digest, family/member digests, evaluation-record digests, sidecar
digests, checkpoint or baseline-spec digests, code/config digests, row counts,
and the protocol digest.

### One cohort, six methods

For each split, the producer constructs one deterministic stream from the
authenticated evaluation-record order and joins risk/sidecar tensors by
sample ID. In each batch it computes:

- `risk-r0` and `risk-r1`: model `p_collision` and frozen quantiles;
- `B1`: last-observation occupancy with the frozen hand aggregator;
- `B2`: age-decay occupancy with the frozen hand aggregator;
- `B3`: selected ConvGRU occupancy with the frozen hand aggregator;
- `B4`: selected ConvGRU occupancy with the selected learned aggregator.

All six tables receive the same method-independent evaluation fields and
ordered IDs. B1--B3 retain the existing explicit scalar-score-to-quantile
proxy policy; B4 records its learned aggregator semantics. Production rows
use the evaluation-record schema rather than filling toy-only fields such as
`background_id` or `occluder_id` with constants.

### Shared calibration protocol

`prediction_protocol.json` freezes the alpha, prediction key, global
split-conformal rule, grouped-calibration dimensions/rule versions, score
semantics, and protocol layout version. `scripts/07_calibrate_risk.py` gains
an explicit production protocol input and rejects any table whose family,
cohort, method semantics, or protocol digest differs. One calibration artifact
is fit independently per method from its calibration table, but all artifacts
must bind the same protocol and calibration cohort digest.

The existing offline evaluator remains responsible for test metrics and can
compare any selected baseline table against a risk table only after validating
that both artifacts bind the same family, test cohort, calibration cohort, and
protocol.

## Failure handling

The implementation fails before training or prediction on absent evaluation
records, family/member mismatch, sidecar mismatch, unsupported checkpoint
layout, incomplete formal selection, non-finite output, cohort mismatch,
calibration/test overlap, or test access before calibration sealing. It never
filters failed samples to make a metric or table complete.

## Verification plan

Tests are added before each implementation slice:

1. evaluation collection round trip, replay-to-seal label binding, atomic
   publication, unknown-entry rejection, and tamper detection;
2. formal B3/B4 gate, validation selection, B3 freeze, best-state reload, and
   no-calibration/test access;
3. producer six-method cohort equality, row/provenance validation, baseline
   spec binding, and deterministic rerun digest;
4. calibration protocol mismatch, family mismatch, and test isolation;
5. existing toy, strict loader, sidecar, risk trainer, and runtime-patch
   regressions.

The final ladder is CPU fixture unit tests, authenticated integration tests,
one-GPU formal smoke/selection tests, and Slurm target-scale smoke. No formal
scientific result is claimed until the 50k run and all six prediction tables
are regenerated from the sealed inputs.

## Out of scope

Multi-node occupancy DDP, changing the risk model architecture or loss,
changing split definitions, modifying raw data or accepted upstream
publications, and adding dependencies remain out of scope.
