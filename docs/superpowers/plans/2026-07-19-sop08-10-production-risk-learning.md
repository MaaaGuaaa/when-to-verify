# SOP08--10 Production Risk Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the accepted schema-3 SOP07 collection to deterministic streaming SOP09 training, publish SOP08 oracle-isolated companion labels, and enable split-safe SOP10 production calibration and evaluation.

**Architecture:** Keep every accepted SOP07 shard immutable. Add an authenticated dataset/family seal above the shard layout, stream one formally loaded shard at a time, and publish occupancy/evaluation information in separate sample-ID-bound sidecars. Production CLIs use distinct control paths and remain fail-closed whenever validation, calibration, test, or provenance evidence is absent.

**Tech Stack:** Python 3.10, NumPy 1.24, PyTorch 2.0, pytest, existing schema-3 contracts and geometry, Slurm; no new dependencies.

### Approved Task 6 amendment (2026-07-20)

- The accepted occupancy sidecar layout remains
  `risk_label_sidecar_v1`; evaluation metadata does not add fields or files to
  that layout.
- Production evaluation records form an independent, sample-ID-bound sibling
  collection beside the risk dataset and occupancy-sidecar collections.
- Task 6 owns the required oracle/renderer-boundary integration in
  `src/datasets/risk_dataset.py`, including the formal triple-return API.  The
  existing risk-only and risk-plus-occupancy-sidecar APIs retain their exact
  signatures and results.
- Task 7 binds and validates three distinct identities: the four-split risk
  dataset family digest, the occupancy-sidecar collection digest, and the
  evaluation-record collection digest.  None may substitute for another.
- Records returned directly by the oracle/renderer boundary are in-memory
  `unpublished` values.  Formal authentication belongs to the sibling
  evaluation-collection publisher, which must join each record to an
  authenticated risk shard and verify its sample ID and labels.  This status
  is not added to `RiskSample` inputs or metadata.

---

## Global execution rules

Use these fixed executables and roots in every command:

```bash
GIT=/home/home/ccnt_zq/zq_zhouyiqun/.local/git/bin/git
PY=/home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI/.conda-envs/sop4-risk/bin/python
ROOT=/home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI
WT=/home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI-worktrees/sop-08-10-production
```

All Python tests, fixture generation, smoke runs, and training run through
Slurm. Red tests use `sbatch --wait` and must fail for the named missing
behavior. Training jobs request a real GPU resource in addition to the `gpu`
partition. Do not modify `src/contracts.py`, `configs/base.yaml`,
`DECISIONS.md`, `STATUS.md`, `pyproject.toml`, `data/`, `legacy/`, or accepted
outputs. Do not use `git add .`.

## File map

Create these focused modules:

```text
src/datasets/risk_dataset_seal.py
  dataset/family publication, digest validation, trusted provenance
src/datasets/risk_sidecars.py
  immutable sidecar data structures and shard I/O
src/datasets/risk_evaluation_metadata.py
  production-only grouping/evaluation records
src/training/risk_trainer.py
  streaming SOP09 optimizer/checkpoint loop
src/training/occupancy_trainer.py
  streaming SOP08 ConvGRU/B4 optimizer loop
src/evaluation/prediction_tables.py
  mode-specific production prediction table validation and construction
scripts/04_seal_risk_dataset.py
  dataset/family seal CLI
scripts/09_predict_risk.py
  common risk/baseline prediction-table producer
configs/risk_model_production.yaml
configs/occupancy_baseline_production.yaml
tests/fixtures/formal_risk_publication.py
  real small schema-3 shard/collection/seal fixtures
```

Modify only the direct integration files listed in each task.

### Task 0: Create the integration worktree

**Files:**
- No source files

- [ ] **Step 1: Confirm all three source worktrees are clean**

```bash
$G -C "$ROOT" status --short --branch
$G -C "$ROOT/../AAAI-worktrees/sop-05-06-joint-pairs" status --short --branch
$G -C "$ROOT/../AAAI-worktrees/sop-08-10-risk-learning" status --short --branch
```

Expected: only branch headers, with no short-status entries.

- [ ] **Step 2: Create a branch from current main and integrate both dependencies**

```bash
$G worktree add "$WT" -b feat/sop-08-10-production main
$G -C "$WT" merge --no-ff feat/sop-05-06-07-v5-integration
$G -C "$WT" merge --no-ff feat/sop-08-10-risk-learning
```

Expected: code merges cleanly. If the SOP document conflicts, retain schema-3
future endpoints `0.2 ... 3.0 s` and reject any current-time future frame.

- [ ] **Step 3: Run the existing merged baseline through Slurm**

```bash
sbatch --wait -p gpu --cpus-per-task=8 --mem=32G --time=00:20:00 \
  --output="$ROOT/.tmp/agent/logs/%x-%j.log" \
  --wrap="cd $WT && $PY -m pytest -q \
    tests/test_risk_dataset.py tests/test_04_generate_risk_dataset_cli.py \
    tests/test_risk_model.py tests/test_risk_training_smoke.py \
    tests/test_occupancy_baseline.py tests/test_occupancy_aggregation.py \
    tests/test_split_conformal.py tests/test_risk_metrics.py \
    tests/test_calibration_isolation.py"
```

Expected: all selected pre-change tests pass. Stop and report if the merged
baseline fails.

### Task 1: Publish and validate `risk_dataset_v2`

**Files:**
- Create: `src/datasets/risk_dataset_seal.py`
- Create: `scripts/04_seal_risk_dataset.py`
- Create: `tests/fixtures/formal_risk_publication.py`
- Create: `tests/test_risk_dataset_seal.py`
- Modify: `src/datasets/risk_dataloader.py`

- [ ] **Step 1: Write failing digest and round-trip tests**

The fixture must call the real `write_risk_shard()` API and build a compact
collection handoff with two contiguous shards. Add tests with these public
calls:

```python
seal = publish_risk_dataset_seal(
    output_dir=tmp_path / "seal",
    collection_root=collection.root,
    base_config_path=collection.base_config,
    split_provenance_path=collection.sop03_run_manifest,
    expected_split="train",
    expected_collection_handoff_sha256=collection.handoff_sha256,
)
loaded = load_risk_dataset_seal(
    seal,
    collection_root=collection.root,
    expected_split="train",
)
assert loaded.sample_count == 12
assert [item.shard_index for item in loaded.shards] == [0, 1]
assert len(loaded.provenance["g1_split_manifest_digest"]) == 32
assert len(loaded.provenance["target_type_policy_digest"]) == 32
assert len(loaded.provenance["dynamic_objects_config_digest"]) == 64
assert len(loaded.risk_dataset_manifest_digest) == 64
```

Add separate tests that mutate handoff SHA, shard semantic digest, ordered
index, split, grid, channel order, G1 digest, target policy, and dynamic config.
Each mutation must raise `RiskDataContractError` before returning a dataset.

- [ ] **Step 2: Verify RED through Slurm**

```bash
sbatch --wait -p gpu --cpus-per-task=4 --mem=16G --time=00:10:00 \
  --output="$ROOT/.tmp/agent/logs/%x-%j.log" \
  --wrap="cd $WT && $PY -m pytest tests/test_risk_dataset_seal.py -q"
```

Expected: failure because `src.datasets.risk_dataset_seal` does not exist.

- [ ] **Step 3: Implement the seal data types and field-specific digests**

Expose exactly these public types and functions:

```python
RISK_DATASET_LAYOUT_VERSION = "risk_dataset_v2"
RISK_DATASET_FAMILY_LAYOUT_VERSION = "risk_dataset_family_v1"

@dataclass(frozen=True)
class RiskShardDescriptor:
    shard_index: int
    relative_root: str
    sample_count: int
    manifest_digest: str
    semantic_digest: str
    payload_sha256: str
    metadata_sha256: str
    summary_sha256: str

@dataclass(frozen=True)
class LoadedRiskDataset:
    seal_root: Path
    collection_root: Path
    manifest: dict[str, object]
    grid: GridSpec
    shards: tuple[RiskShardDescriptor, ...]
    split: str
    sample_count: int
    risk_dataset_manifest_digest: str
    provenance: dict[str, str]

def canonical_dynamic_objects_digest(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def publish_risk_dataset_seal(
    output_dir: str | Path,
    *,
    collection_root: str | Path,
    base_config_path: str | Path,
    split_provenance_path: str | Path,
    expected_split: str,
    expected_collection_handoff_sha256: str,
) -> Path

def load_risk_dataset_seal(
    seal_root: str | Path,
    *,
    collection_root: str | Path,
    expected_split: str,
    expected_manifest_digest: str | None = None,
) -> LoadedRiskDataset
```

`publish_risk_dataset_seal()` must obtain the G1 digest from the authenticated
SOP03 `run_manifest.json`, derive dynamic config from its frozen base-config
snapshot, obtain one consistent target-policy digest from formally loaded
shards, call `load_risk_shard()` for every shard, and atomically publish only
`dataset_manifest.json`, `checksums.sha256`, and `.producer-complete`.

Validate digest algorithms per field: G1 and target policy are lowercase
BLAKE2b-128 with length 32; dataset and dynamic config are lowercase SHA-256
with length 64. Never pad or re-hash upstream values.

- [ ] **Step 4: Replace the production fail-closed placeholder with the strict seal loader**

Change the public dataloader entry point to:

```python
def load_production_risk_dataset(
    seal_root: str | Path,
    *,
    collection_root: str | Path,
    expected_split: str,
    expected_manifest_digest: str | None = None,
) -> LoadedRiskDataset:
    return load_risk_dataset_seal(
        seal_root,
        collection_root=collection_root,
        expected_split=expected_split,
        expected_manifest_digest=expected_manifest_digest,
    )
```

Do not accept `risk_shard_npz_jsonl_v1`, a shard root in place of a seal, or a
manifest without `.producer-complete`.

- [ ] **Step 5: Verify GREEN and CLI seal behavior through Slurm**

```bash
sbatch --wait -p gpu --cpus-per-task=8 --mem=32G --time=00:15:00 \
  --output="$ROOT/.tmp/agent/logs/%x-%j.log" \
  --wrap="cd $WT && $PY -m pytest -q \
    tests/test_risk_dataset_seal.py tests/test_risk_model.py"
```

Expected: all selected tests pass, including tamper and overwrite rejection.

- [ ] **Step 6: Commit exact Task 1 files**

```bash
$G add src/datasets/risk_dataset_seal.py src/datasets/risk_dataloader.py \
  scripts/04_seal_risk_dataset.py tests/fixtures/formal_risk_publication.py \
  tests/test_risk_dataset_seal.py
$G commit -m "feat(risk-data): seal schema3 production collections"
```

### Task 2: Stream deterministic production batches

**Files:**
- Modify: `src/datasets/risk_dataloader.py`
- Modify: `tests/fixtures/formal_risk_publication.py`
- Create: `tests/test_risk_production_dataloader.py`

- [ ] **Step 1: Write failing selection, order, resume, dtype, and label tests**

Use these public contracts:

```python
subset = select_production_risk_subset(dataset, max_samples=9, seed=42)
first = list(iter_production_risk_batches(
    dataset, subset=subset, batch_size=4, seed=42, epoch=0
))
second = list(iter_production_risk_batches(
    dataset, subset=subset, batch_size=4, seed=42, epoch=0
))
assert flatten_ids(first) == flatten_ids(second)
assert all(t.dtype == torch.float32 for batch, _ in first
           for t in (*batch.model_inputs.values(), *batch.targets.values()))
```

Resume from the cursor returned after batch one and require the concatenated
IDs to equal uninterrupted execution with no duplicates or omissions. Include
a `temporal_safe` fixture whose labels are `collision=0, near_miss=1` and prove
collation preserves both values.

- [ ] **Step 2: Verify RED through Slurm**

```bash
sbatch --wait -p gpu --cpus-per-task=4 --mem=20G --time=00:10:00 \
  --output="$ROOT/.tmp/agent/logs/%x-%j.log" \
  --wrap="cd $WT && $PY -m pytest tests/test_risk_production_dataloader.py -q"
```

Expected: import failure for the new subset/iterator APIs.

- [ ] **Step 3: Implement immutable subset and cursor contracts**

```python
@dataclass(frozen=True)
class ProductionRiskSubset:
    sample_ids: tuple[str, ...]
    sample_ids_digest_sha256: str
    dataset_manifest_digest: str
    seed: int
    max_samples: int

@dataclass(frozen=True)
class RiskStreamCursor:
    epoch: int
    shard_order_position: int
    shard_index: int
    row_order_position: int
    samples_yielded: int
    dataset_manifest_digest: str
    subset_digest_sha256: str
```

Subset membership uses SHA-256 of dataset digest, selection seed, and sample
ID without epoch. Shard and row ordering additionally include epoch. Sort by
the digest bytes and use sample ID/index as a deterministic tie breaker.

- [ ] **Step 4: Implement one-shard-at-a-time collation and resume**

```python
def collate_production_risk_samples(
    samples: Sequence[RiskSample],
    *,
    grid: GridSpec,
    expected_split: str,
    dataset_provenance: Mapping[str, object],
) -> RiskBatch

def iter_production_risk_batches(
    dataset: LoadedRiskDataset,
    *,
    subset: ProductionRiskSubset,
    batch_size: int,
    seed: int,
    epoch: int,
    start_cursor: RiskStreamCursor | None = None,
) -> Iterator[tuple[RiskBatch, RiskStreamCursor]]
```

Load each shard exclusively through `load_risk_shard()`. Explicitly convert
all scalar targets to float32. Release the loaded shard before opening the next
one. First implementation is single-process and performs no multi-worker
interleaving.

- [ ] **Step 5: Verify GREEN through Slurm**

```bash
sbatch --wait -p gpu --cpus-per-task=4 --mem=24G --time=00:12:00 \
  --output="$ROOT/.tmp/agent/logs/%x-%j.log" \
  --wrap="cd $WT && $PY -m pytest -q \
    tests/test_risk_production_dataloader.py tests/test_risk_dataset_seal.py \
    tests/test_risk_model.py"
```

Expected: all tests pass and the stream-retention test observes at most one
loaded shard plus one batch.

- [ ] **Step 6: Commit exact Task 2 files**

```bash
$G add src/datasets/risk_dataloader.py tests/fixtures/formal_risk_publication.py \
  tests/test_risk_production_dataloader.py
$G commit -m "feat(risk-data): stream deterministic production batches"
```

### Task 3: Run SOP09 production smoke and 1k overfit

**Files:**
- Create: `src/training/__init__.py`
- Create: `src/training/risk_trainer.py`
- Create: `configs/risk_model_production.yaml`
- Create: `tests/test_risk_production_training.py`
- Modify: `src/models/risk_model.py`
- Modify: `scripts/06_train_risk_model.py`

- [ ] **Step 1: Write failing trainer and CLI tests**

Cover one-shard finite forward/backward, deterministic 1k subset history,
resume equivalence, checkpoint reload, trajectory-query sensitivity, atomic
overwrite rejection, and `formal_50k` rejection without validation evidence.
The CLI test must patch the toy constructor to raise if production mode calls
it; a successful fixture run proves the production branch never falls through.

- [ ] **Step 2: Verify RED through Slurm**

```bash
sbatch --wait -p gpu --cpus-per-task=4 --mem=24G --time=00:12:00 \
  --output="$ROOT/.tmp/agent/logs/%x-%j.log" \
  --wrap="cd $WT && $PY -m pytest tests/test_risk_production_training.py -q"
```

Expected: import failure for `src.training.risk_trainer`.

- [ ] **Step 3: Expose the shared batch loss without changing its math**

Rename `_batch_loss()` to:

```python
def compute_risk_batch_loss(
    model: RiskModel,
    batch: RiskBatch,
    *,
    lambda_collision: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    return output, risk_loss(
        output["quantiles"],
        output["collision_logits"],
        batch.targets["risk_severity"],
        batch.targets["collision_label"],
        lambda_collision=lambda_collision,
    )
```

Update the toy trainer to call the public function and prove its existing
metrics remain unchanged.

- [ ] **Step 4: Implement the streaming production trainer**

```python
@dataclass(frozen=True)
class ProductionRiskTrainingConfig:
    stage: Literal["one_shard_smoke", "real_1k_overfit", "formal_50k"]
    variant: Literal["r0", "r1"]
    seed: int
    device: str
    hidden_channels: int
    batch_size: int
    epochs: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    lambda_collision: float
    checkpoint_interval_steps: int

def train_production_risk_model(
    *,
    train_dataset: LoadedRiskDataset,
    train_subset: ProductionRiskSubset,
    config: ProductionRiskTrainingConfig,
    output_dir: str | Path,
    validation_dataset: LoadedRiskDataset | None = None,
    resume_from: str | Path | None = None,
    cross_split_audit: Mapping[str, object] | None = None,
) -> ProductionRiskTrainingResult
```

Save optimizer state and `RiskStreamCursor` only at optimizer-step boundaries.
Smoke/1k stages publish no best-validation checkpoint. `formal_50k` requires a
validation dataset and `global_cross_split_leakage=PROVEN`; it never opens test.

Production checkpoint provenance uses the approved field-specific digest
rules and additionally binds training stage, subset digest, configuration,
validation digest when present, and family digest when present.

- [ ] **Step 5: Wire a distinct production CLI control path**

Add production arguments for train/validation seal and collection roots,
variant, output root, resume checkpoint, and stage. After the production
trainer returns, exit without constructing toy datasets. Keep toy behavior and
artifact validation unchanged.

- [ ] **Step 6: Verify GREEN through Slurm**

```bash
sbatch --wait -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=48G \
  --time=00:20:00 --output="$ROOT/.tmp/agent/logs/%x-%j.log" \
  --wrap="cd $WT && $PY -m pytest -q \
    tests/test_risk_production_training.py tests/test_risk_training_smoke.py \
    tests/test_risk_model.py tests/test_risk_losses.py"
```

Expected: all tests pass on an allocated GPU, with finite gradients and zero
quantile crossings.

- [ ] **Step 7: Commit exact Task 3 files**

```bash
$G add src/training/__init__.py src/training/risk_trainer.py \
  src/models/risk_model.py scripts/06_train_risk_model.py \
  configs/risk_model_production.yaml tests/test_risk_production_training.py
$G commit -m "feat(risk): train schema3 production batches"
```

### Task 4: Publish SOP08 oracle-isolated sidecars

**Files:**
- Create: `src/generation/risk_sidecars.py`
- Create: `src/datasets/sidecar_writer.py`
- Create: `tests/test_risk_sidecars.py`
- Create: `tests/test_sidecar_writer.py`
- Modify: `src/datasets/risk_dataset.py`
- Modify: `scripts/04_generate_risk_dataset.py`
- Modify: `tests/test_risk_dataset.py`
- Modify: `tests/test_04_generate_risk_dataset_cli.py`

- [ ] **Step 1: Write failing geometric sidecar tests**

Hand-build a small grid with one declared hidden circle and one undeclared
context actor. Require exact per-frame footprint rasterization, exclusion of
the context actor, all-zero hidden occupancy for `empty_blind_spot`, inflated
robot footprint masks, uint8 binary storage, and float32 endpoints
`0.2 ... 3.0 s`.

- [ ] **Step 2: Verify RED through Slurm**

```bash
sbatch --wait -p gpu --cpus-per-task=4 --mem=20G --time=00:10:00 \
  --output="$ROOT/.tmp/agent/logs/%x-%j.log" \
  --wrap="cd $WT && $PY -m pytest -q \
    tests/test_risk_sidecars.py tests/test_sidecar_writer.py"
```

Expected: import failures for both new modules.

- [ ] **Step 3: Implement sidecar construction at the oracle boundary**

```python
@dataclass(frozen=True)
class RiskLabelSidecar:
    sample_id: str
    hidden_risk_occupancy: np.ndarray
    robot_future_footprints: np.ndarray
    future_endpoint_times_s: np.ndarray

def build_risk_label_sidecar(
    *,
    sample_id: str,
    trajectory: LocalTrajectory,
    world: OracleWorld,
    hidden_object_ids: tuple[str, ...],
    robot_footprint: Footprint,
    grid: GridSpec,
    future_dt_s: float,
) -> RiskLabelSidecar
```

For each hidden object and endpoint, OR the exact
`rasterize_footprint(footprint_from_spec(...), pose, grid)` mask. Rasterize the
inflated `_robot_footprint(base_config)` at each robot trajectory pose. Return
owned immutable arrays.

Add `build_risk_samples_and_sidecars_from_sop06_group()` at the existing
`compute_hidden_risk_gt()` boundary. Keep
`build_risk_samples_from_sop06_group()` as a wrapper returning only samples so
existing callers and RiskSample digests do not change.

- [ ] **Step 4: Implement immutable sidecar shard I/O**

```python
def write_risk_sidecar_shard(
    sidecars: Sequence[RiskLabelSidecar],
    output_dir: str | Path,
    *,
    grid: GridSpec,
    split: str,
    shard_index: int,
    source_risk_shard_semantic_digest: str,
) -> dict[str, Path]

def load_risk_sidecar_shard(
    output_dir: str | Path,
    *,
    grid: GridSpec,
    expected_sample_ids: Sequence[str],
    expected_source_risk_shard_semantic_digest: str,
) -> LoadedRiskSidecarShard
```

Store binary masks as uint8 in `sidecars.npz`; loader converts to float32 only
for SOP08 batches. Bind ordered IDs, source risk-shard digest, split/index,
grid, endpoints, shapes, dtypes, and bytes in `summary.json`. Reload fully
before atomic rename and reject extra files, overwrite, reorder, and tampering.

- [ ] **Step 5: Add optional sidecar publication to the SOP07 CLI**

The formal CLI writes risk and sidecar shards to separate caller-specified
roots. Either both reload successfully or the report is not complete. Increment
the producer version and report both semantic digests. Existing accepted risk
shard layout stays unchanged.

- [ ] **Step 6: Verify GREEN through Slurm**

```bash
sbatch --wait -p gpu --cpus-per-task=8 --mem=48G --time=00:20:00 \
  --output="$ROOT/.tmp/agent/logs/%x-%j.log" \
  --wrap="cd $WT && $PY -m pytest -q \
    tests/test_risk_sidecars.py tests/test_sidecar_writer.py \
    tests/test_risk_dataset.py tests/test_04_generate_risk_dataset_cli.py"
```

Expected: exact geometry, input/oracle isolation, deterministic digest, and
atomic failure tests pass.

- [ ] **Step 7: Commit exact Task 4 files**

```bash
$G add src/generation/risk_sidecars.py src/datasets/sidecar_writer.py \
  src/datasets/risk_dataset.py scripts/04_generate_risk_dataset.py \
  tests/test_risk_sidecars.py tests/test_sidecar_writer.py \
  tests/test_risk_dataset.py tests/test_04_generate_risk_dataset_cli.py
$G commit -m "feat(occupancy): publish oracle-isolated risk sidecars"
```

### Task 5: Run SOP08 production baselines

**Files:**
- Create: `src/training/occupancy_trainer.py`
- Create: `configs/occupancy_baseline_production.yaml`
- Create: `tests/test_occupancy_production_training.py`
- Modify: `src/datasets/risk_dataset_seal.py`
- Modify: `src/datasets/risk_dataloader.py`
- Modify: `src/evaluation/risk_baselines.py`
- Modify: `scripts/05_train_occupancy_baseline.py`
- Modify: `tests/test_risk_baselines.py`
- Modify: `tests/test_occupancy_training_smoke.py`

- [ ] **Step 1: Write failing production join and training tests**

Require risk/sidecar ordered sample-ID equality, sidecar digest binding,
float32 `[B,15,H,W]` labels/query masks, absence from `model_inputs`, B1--B4
scores in `[0,1]`, one-shard finite gradients, fixed 1k loss reduction, and
production checkpoint provenance. Tampering or reordering must fail before a
batch is returned.

- [ ] **Step 2: Verify RED through Slurm**

```bash
sbatch --wait -p gpu --cpus-per-task=4 --mem=24G --time=00:12:00 \
  --output="$ROOT/.tmp/agent/logs/%x-%j.log" \
  --wrap="cd $WT && $PY -m pytest tests/test_occupancy_production_training.py -q"
```

Expected: production occupancy contract remains unavailable.

- [ ] **Step 3: Extend the dataset seal with an optional authenticated sidecar collection**

Add a sidecar section containing sidecar layout, collection digest, ordered
shard descriptors, and base risk-dataset digest. `load_production_risk_dataset`
continues to support SOP09 without sidecars; a distinct occupancy join function
requires and verifies them:

```python
def iter_production_occupancy_batches(
    dataset: LoadedRiskDataset,
    *,
    sidecar_root: str | Path,
    subset: ProductionRiskSubset,
    batch_size: int,
    seed: int,
    epoch: int,
) -> Iterator[ProductionOccupancyBatch]
```

`ProductionOccupancyBatch.model_inputs` contains only RiskSample inputs;
`label_sidecars` contains occupancy and robot masks.

- [ ] **Step 4: Implement production occupancy training and B1--B4 streaming scores**

Reuse `LastObservationHold`, `AgeDecay`, `ConvGRUOccupancyPredictor`, hand
aggregators, and `LearnedOccupancyRiskAggregator`. Add a mini-batch trainer for
B3/B4; never call `fit_toy_occupancy_model()` in production. Store best
validation checkpoint only when validated split evidence exists.

- [ ] **Step 5: Wire the production occupancy CLI**

Accept dataset seal, risk collection, sidecar collection, stage, subset size,
batch size, device, and output root. Production exits through its own artifact
publisher. Keep toy CLI output byte-compatible with existing tests.

- [ ] **Step 6: Verify GREEN through Slurm**

```bash
sbatch --wait -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G \
  --time=00:25:00 --output="$ROOT/.tmp/agent/logs/%x-%j.log" \
  --wrap="cd $WT && $PY -m pytest -q \
    tests/test_occupancy_production_training.py \
    tests/test_occupancy_training_smoke.py tests/test_risk_baselines.py \
    tests/test_occupancy_aggregation.py tests/test_occupancy_baseline.py"
```

Expected: all selected tests pass; oracle sidecars never appear in model input
keys.

- [ ] **Step 7: Commit exact Task 5 files**

```bash
$G add src/training/occupancy_trainer.py \
  configs/occupancy_baseline_production.yaml \
  src/datasets/risk_dataset_seal.py src/datasets/risk_dataloader.py \
  src/evaluation/risk_baselines.py scripts/05_train_occupancy_baseline.py \
  tests/test_occupancy_production_training.py tests/test_risk_baselines.py \
  tests/test_occupancy_training_smoke.py
$G commit -m "feat(occupancy): train production risk baselines"
```

### Task 6: Publish dataset family and production evaluation metadata

**Files:**
- Create: `src/datasets/risk_evaluation_metadata.py`
- Create: `tests/test_risk_dataset_family.py`
- Create: `tests/test_production_evaluation_metadata.py`
- Modify: `src/datasets/risk_dataset_seal.py`
- Modify: `src/datasets/risk_dataset.py`
- Modify: `src/generation/risk_sidecars.py`
- Modify: `src/datasets/sidecar_writer.py`
- Modify: `scripts/04_seal_risk_dataset.py`

- [ ] **Step 1: Write failing family isolation and evaluation-record tests**

Use four small split seals. Session overlap must be allowed and reported;
recording, cross-role recording, source snippet, pair group, and seed namespace
overlap must fail. Hand-check critical-area fraction, mean age over the critical
region, visible-cell dynamic density, inflated robot footprint spec, target
footprint spec, OOD rule, and complete-six-pack pair eligibility.

- [ ] **Step 2: Verify RED through Slurm**

```bash
sbatch --wait -p gpu --cpus-per-task=4 --mem=24G --time=00:12:00 \
  --output="$ROOT/.tmp/agent/logs/%x-%j.log" \
  --wrap="cd $WT && $PY -m pytest -q \
    tests/test_risk_dataset_family.py \
    tests/test_production_evaluation_metadata.py"
```

Expected: family/evaluation APIs do not exist.

- [ ] **Step 3: Implement and atomically publish the four-split family seal**

```python
def publish_risk_dataset_family(
    output_dir: str | Path,
    *,
    members: Mapping[str, LoadedRiskDataset],
) -> Path

def load_risk_dataset_family(
    root: str | Path,
    *,
    expected_member_digests: Mapping[str, str] | None = None,
) -> LoadedRiskDatasetFamily
```

Require exactly `train`, `val`, `calibration`, and `test`, canonical THOR
recording-generalization policy, and `global_cross_split_leakage=PROVEN`.

- [ ] **Step 4: Implement immutable evaluation-only records**

```python
def derive_production_evaluation_record(
    *,
    sample: RiskSample,
    source: RiskBuildInput,
    rendered: RenderedObservation,
    ground_truth: RiskGroundTruth,
    robot_footprint: Footprint,
    age_max_s: float,
    pair_eligible: bool,
    ood_tag: str,
    robot_footprint_provenance: Mapping[str, object],
    ood_evidence: Mapping[str, object],
) -> dict[str, object]

def validate_production_evaluation_record(
    record: Mapping[str, object],
    *,
    expected_sample_id: str | None = None,
) -> dict[str, object]
```

Freeze rule versions and calculate fields once at the oracle/renderer boundary.
Do not put them in `RiskSample` inputs or metadata.  Add a formal API returning
aligned `(samples, occupancy_sidecars, evaluation_records)` while leaving the
two existing wrappers unchanged.  Publish evaluation records to their own
sibling collection, without changing `risk_label_sidecar_v1`, and bind their
ordered sample-ID digest independently.

- [ ] **Step 5: Verify GREEN through Slurm and commit**

```bash
sbatch --wait -p gpu --cpus-per-task=6 --mem=32G --time=00:15:00 \
  --output="$ROOT/.tmp/agent/logs/%x-%j.log" \
  --wrap="cd $WT && $PY -m pytest -q \
    tests/test_risk_dataset_family.py \
    tests/test_production_evaluation_metadata.py \
    tests/test_risk_sidecars.py tests/test_sidecar_writer.py"
$G add src/datasets/risk_dataset_seal.py \
  src/datasets/risk_evaluation_metadata.py src/generation/risk_sidecars.py \
  src/datasets/sidecar_writer.py scripts/04_seal_risk_dataset.py \
  tests/test_risk_dataset_family.py tests/test_production_evaluation_metadata.py
$G commit -m "feat(risk-data): seal split family and evaluation metadata"
```

### Task 7: Enable SOP10 production prediction, calibration, and evaluation

**Files:**
- Create: `src/evaluation/prediction_tables.py`
- Create: `scripts/09_predict_risk.py`
- Create: `tests/test_production_prediction_table.py`
- Create: `tests/test_production_calibration_isolation.py`
- Create: `tests/test_production_eval_cli.py`
- Modify: `src/calibration/split_conformal.py`
- Modify: `src/evaluation/risk_metrics.py`
- Modify: `scripts/07_calibrate_risk.py`
- Modify: `scripts/10_eval_offline.py`
- Modify: `tests/test_split_conformal.py`
- Modify: `tests/test_risk_metrics.py`
- Modify: `tests/test_calibration_isolation.py`

- [ ] **Step 1: Write failing production table and calibration tests**

Require mode-specific production rows without toy-only `background_id`, strict
base/source role identities, structured target/robot footprint specs, family
and evaluation-metadata digests, field-specific provenance digests, canonical
`empty_blind_spot`, and cohort digest changes after any label/group tamper.
Prediction tables and downstream artifacts must separately bind the risk
dataset family digest, occupancy-sidecar collection digest, and evaluation-
record collection digest; equality or presence of one never authenticates the
other two.

Calibration tests must fit calibration only, allow/report session overlap, and
reject recording/source/snippet/pair/seed overlap or family-member mismatch.
CLI tests cover G2 `pass`, `fail`, and `unavailable` without filtering rows.

- [ ] **Step 2: Verify RED through Slurm**

```bash
sbatch --wait -p gpu --cpus-per-task=6 --mem=32G --time=00:15:00 \
  --output="$ROOT/.tmp/agent/logs/%x-%j.log" \
  --wrap="cd $WT && $PY -m pytest -q \
    tests/test_production_prediction_table.py \
    tests/test_production_calibration_isolation.py \
    tests/test_production_eval_cli.py"
```

Expected: production remains fail-closed.

- [ ] **Step 3: Separate prediction-table validation from conformal math**

Implement mode-specific row/table builders and validators in
`src/evaluation/prediction_tables.py`. Re-export the existing public names from
`split_conformal.py` so toy callers remain compatible. Validate production
provenance lengths per approved algorithm rather than the old all-64 loop.

- [ ] **Step 4: Enable production calibration and THOR isolation**

Reuse one-sided residuals, finite-sample quantile, and grouped calibration
unchanged. Production artifacts bind checkpoint, dataset/family, evaluation
metadata, G1, dynamic config, target policy, cohort, and rule versions.
Isolation uses the family seal: session overlap is allowed/reported; forbidden
identity overlap raises before applying calibration to test.

- [ ] **Step 5: Enable production metrics and structured G2 status**

```python
def evaluate_risk_g2(
    metrics: Mapping[str, object],
    *,
    auroc_min: float = 0.80,
    coverage_min: float = 0.85,
    coverage_max: float = 0.95,
    false_safe_reduction_min: float = 0.10,
    minimum_improved_hard_negative_subsets: int = 1,
) -> dict[str, object]
```

Return `status` as exactly `pass`, `fail`, or `unavailable` with structured
reasons. Recognize `empty_blind_spot`, retain no-object collision/severity rows,
and exclude only null critical-object rows from clearance aggregation.

- [ ] **Step 6: Wire production prediction/calibration/evaluation CLIs**

`scripts/09_predict_risk.py` is the sole production table writer for both main
and occupancy methods. Calibration requires the calibration family member;
offline evaluation requires test and a matching calibration artifact. Main and
baseline must share family, test cohort, and calibration protocol.

- [ ] **Step 7: Verify GREEN through Slurm**

```bash
sbatch --wait -p gpu --cpus-per-task=8 --mem=40G --time=00:20:00 \
  --output="$ROOT/.tmp/agent/logs/%x-%j.log" \
  --wrap="cd $WT && $PY -m pytest -q \
    tests/test_production_prediction_table.py \
    tests/test_production_calibration_isolation.py \
    tests/test_production_eval_cli.py tests/test_split_conformal.py \
    tests/test_risk_metrics.py tests/test_calibration_isolation.py"
```

Expected: production and toy tests pass; hand-calculated conformal and metric
values remain unchanged.

- [ ] **Step 8: Commit exact Task 7 files**

```bash
$G add src/evaluation/prediction_tables.py scripts/09_predict_risk.py \
  src/calibration/split_conformal.py src/evaluation/risk_metrics.py \
  scripts/07_calibrate_risk.py scripts/10_eval_offline.py \
  tests/test_production_prediction_table.py \
  tests/test_production_calibration_isolation.py \
  tests/test_production_eval_cli.py tests/test_split_conformal.py \
  tests/test_risk_metrics.py tests/test_calibration_isolation.py
$G commit -m "feat(risk-eval): calibrate and evaluate production cohorts"
```

### Task 8: Real artifact ladder and final verification

**Files:**
- Modify: `docs/event_centered_blind_spot_agent_sops.md` only if the integrated
  branch already owns the corresponding production handoff text
- Generated outputs: versioned `outputs/` roots, never committed

- [ ] **Step 1: Publish the train dataset seal on Slurm**

Use the accepted collection root and handoff SHA from the design. Publish to a
new root; do not add files to the accepted collection.

- [ ] **Step 2: Run one real shard SOP09 GPU smoke**

Require finite arrays/loss/gradients, exact shape, no oracle keys, deterministic
IDs, zero quantile crossing, and successful checkpoint reload.

- [ ] **Step 3: Run the fixed real 1k SOP09 overfit**

Run R0 then R1 as separate Slurm jobs with the same subset digest. Record job
IDs, GPU, CPU, memory, wall time, throughput, initial/final loss, trajectory
sensitivity, checkpoint digests, and limitations. Do not call either result a
formal best checkpoint.

- [ ] **Step 4: Replay and publish a one-shard SOP08 sidecar smoke**

Require regenerated RiskSample IDs and risk-shard semantic digest to match the
accepted shard before publishing the sidecar. Then run B1--B4 one-shard smoke.

- [ ] **Step 5: Run the complete relevant test suite through Slurm**

```bash
sbatch --wait -p gpu --cpus-per-task=8 --mem=64G --time=00:45:00 \
  --output="$ROOT/.tmp/agent/logs/%x-%j.log" \
  --wrap="cd $WT && $PY -m pytest -q \
    tests/test_contracts.py tests/test_toy_fixture.py \
    tests/test_risk_dataset.py tests/test_04_generate_risk_dataset_cli.py \
    tests/test_risk_dataset_seal.py tests/test_risk_production_dataloader.py \
    tests/test_risk_production_training.py tests/test_risk_sidecars.py \
    tests/test_sidecar_writer.py tests/test_occupancy_production_training.py \
    tests/test_risk_dataset_family.py \
    tests/test_production_evaluation_metadata.py \
    tests/test_production_prediction_table.py \
    tests/test_production_calibration_isolation.py \
    tests/test_production_eval_cli.py tests/test_risk_model.py \
    tests/test_risk_losses.py tests/test_risk_training_smoke.py \
    tests/test_occupancy_baseline.py tests/test_occupancy_aggregation.py \
    tests/test_risk_baselines.py tests/test_occupancy_training_smoke.py \
    tests/test_split_conformal.py tests/test_risk_metrics.py \
    tests/test_calibration_isolation.py"
```

Expected: zero failures. Report unavailable formal gates rather than weakening
tests if validation/calibration/test publications do not yet exist.

- [ ] **Step 6: Review exact diff and commit the final handoff**

```bash
$G status --short
$G diff --check
$G diff --stat main...HEAD
$G log --oneline --decorate main..HEAD
```

Stage only any remaining owned documentation. Commit it separately with a
message describing production bridge evidence. Do not merge main, push, or
commit outputs/logs.
