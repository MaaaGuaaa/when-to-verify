# SOP08 Sidecar Backfill And Complete Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish sample-ID-aligned future hidden occupancy labels for all 127,787 accepted SOP07 samples and rebuild a portable four-split bundle that can train SOP08 baselines and a later risk-model occupancy auxiliary head without upstream data generation.

**Architecture:** Keep every accepted SOP07 shard immutable. Replay each shard from its accepted SOP03/SOP04/SOP05 inputs into a temporary risk shard plus an immutable occupancy sidecar, and accept the sidecar only when the replayed ordered sample IDs, manifest digest, and semantic digest match the accepted risk shard. Seal each split against its sidecars, audit all four splits, then add the sidecars and seals to a new archive rather than modifying the existing v1 archive.

**Tech Stack:** Python 3.10, NumPy 1.24, pytest, existing schema-3 loaders/writers, Slurm arrays, tar/zstd; no new dependencies.

---

## Frozen supervision contract

- Label name: `hidden_risk_occupancy`.
- Logical/load dtype and shape: `float32 [N,15,160,160]`; immutable disk storage is binary `uint8` in compressed NPZ.
- Time indices: endpoint `k` is `(k+1)*0.2 s`, exactly `0.2 ... 3.0 s`; no `t=0` future frame.
- Spatial convention: the existing schema-3 robot-centric 160x160 grid at 0.1 m resolution, using the existing footprint rasterizer.
- Contents: binary union of only caller-declared hidden dynamic objects that participate in the SOP07 hidden-risk target. Static obstacles and undeclared/context actors are excluded, and objects are not cropped when they become visible in the future. `empty_blind_spot` or no hidden target means an all-zero label.
- Query companion: loader `float32 robot_future_footprints [N,15,160,160]` on the same time/grid layout and `float32 future_endpoint_times_s [15]`.
- Join: require exact ordered `sample_id`, source risk-shard semantic digest, risk/sidecar pair completion marker, and split collection seal. Missing, extra, duplicate, reordered, or digest-mismatched evidence fails closed; no loose positional join is allowed.
- Isolation: sidecars are supervision/offline-analysis only and never appear in model inputs.
- Consumers: SOP08 B3 ConvGRU/B4 aggregation training and the optional SOP09/R3 occupancy auxiliary head use the same `hidden_risk_occupancy` target. Four-split sidecar preparation is mandatory for a complete training bundle; enabling the auxiliary head remains an experimental option.
- Versioning: this is an external additive schema-3 label publication. It does not alter `RiskSample`, its model-input schema, or the core schema/version.

### Task 1: Add accepted-shard replay verification

**Files:**
- Create: `scripts/04_backfill_risk_sidecars.py`
- Create: `tests/test_04_backfill_risk_sidecars.py`

- [ ] **Step 1: Write failing real-I/O tests**

Create accepted and replay risk shards with the real `write_risk_shard()` API and a sidecar with the real `write_risk_sidecar_shard()` API. Require exact ordered sample IDs, split/index, manifest digest, semantic digest, sidecar source digest, and pair marker. Add separate failures for changed risk semantics, reordered IDs, wrong basename/index, missing marker, and tampered sidecar.

- [ ] **Step 2: Run RED through Slurm**

```bash
sbatch --wait -p gpu --cpus-per-task=4 --mem=20G --time=00:10:00 \
  --output="$ROOT/.tmp/agent/logs/%x-%j.log" \
  --wrap="cd $WT && $PY -m pytest -q tests/test_04_backfill_risk_sidecars.py"
```

Expected: import failure because `scripts/04_backfill_risk_sidecars.py` does not exist.

- [ ] **Step 3: Implement one Slurm-array-task runner**

The CLI must resolve one task from the accepted batch report and SOP05 batch handoff, invoke the existing formal SOP07 CLI with both temporary replay-risk and final sidecar roots, formally reload accepted risk/replay risk/sidecar/marker, and return one canonical JSON report. It must fail closed before returning success unless all identities and digests match. Existing complete outputs may be resumed only after the same full reload.

- [ ] **Step 4: Run GREEN and regressions through Slurm**

```bash
sbatch --wait -p gpu --cpus-per-task=8 --mem=48G --time=00:20:00 \
  --output="$ROOT/.tmp/agent/logs/%x-%j.log" \
  --wrap="cd $WT && $PY -m pytest -q \
    tests/test_04_backfill_risk_sidecars.py tests/test_risk_sidecars.py \
    tests/test_sidecar_writer.py tests/test_04_generate_risk_dataset_cli.py \
    tests/test_risk_dataset_seal.py tests/test_occupancy_production_training.py"
```

Expected: zero failures.

- [ ] **Step 5: Commit exact Task 1 files**

```bash
$GIT add scripts/04_backfill_risk_sidecars.py \
  tests/test_04_backfill_risk_sidecars.py
$GIT commit -m "feat(occupancy): backfill accepted risk sidecars"
```

### Task 2: Synchronize the authoritative SOP text

**Files:**
- Modify: `docs/event_centered_blind_spot_implementation_spec.md`
- Modify: `docs/parallel_acceleration_implementation_plan.md`
- Modify: `docs/event_centered_blind_spot_agent_sops.md`

- [ ] **Step 1: Add the frozen supervision contract**

Document the exact arrays, object inclusion policy, endpoint timing, rasterization reuse, input/label isolation, sample-ID join, baseline/aux-head consumers, and four-split packaging requirement. State that this is an additive label publication and does not alter `RiskSample` or schema 3 model inputs.

- [ ] **Step 2: Run contract and sidecar tests through Slurm**

```bash
sbatch --wait -p gpu --cpus-per-task=4 --mem=20G --time=00:12:00 \
  --output="$ROOT/.tmp/agent/logs/%x-%j.log" \
  --wrap="cd $WT && $PY -m pytest -q tests/test_contracts.py \
    tests/test_input_oracle_isolation.py tests/test_risk_sidecars.py"
```

Expected: zero failures and no oracle/future key in model inputs.

- [ ] **Step 3: Commit exact Task 2 files**

```bash
$GIT add docs/event_centered_blind_spot_implementation_spec.md \
  docs/parallel_acceleration_implementation_plan.md \
  docs/event_centered_blind_spot_agent_sops.md \
  docs/superpowers/plans/2026-07-20-sop08-sidecar-backfill-and-bundle.md
$GIT commit -m "docs(occupancy): freeze sidecar supervision contract"
```

### Task 3: Real smoke and four-split Slurm generation

**Files:**
- Generated only: versioned roots under `outputs/`
- Temporary only: `.tmp/agent/outputs/` replay roots and `.tmp/agent/logs/`

- [ ] **Step 1: Run one 10--100 event real shard smoke**

Replay one accepted shard, then check exact risk identity, sidecar shape/dtype/binary/finite bounds, endpoint times, nonzero occupancy where applicable, all-zero empty-blind-spot targets, collision/occupancy temporal-spatial agreement, and deterministic sidecar digest on a second independent replay.

- [ ] **Step 2: Submit four split arrays**

Use one array task per accepted shard, at most 192 concurrent tasks, one CPU and 4--6 GiB per task. Train uses 256 tasks; calibration/val/test use 36 tasks each. All tasks read the existing formal SOP03/SOP04/SOP05 publications and write only new sidecar/replay roots.

- [ ] **Step 3: Audit every shard**

Require exactly 364 sidecar shards and 127,787 ordered sample IDs, zero missing/extra/duplicate IDs, exact source risk semantic digests, exact split/index/counts, binary finite masks, and endpoint times `0.2 ... 3.0 s`.

- [ ] **Step 4: Publish four authenticated dataset seals**

Run `scripts/04_seal_risk_dataset.py` once per split with its accepted risk root and complete sidecar root. Formally reload each seal and sidecar collection.

Four-split release gate:

- [ ] `train`: 256 accepted risk shards have one sidecar and pair marker each; seal reload passes.
- [ ] `calibration`: 36 accepted risk shards have one sidecar and pair marker each; seal reload passes.
- [ ] `val`: 36 accepted risk shards have one sidecar and pair marker each; seal reload passes.
- [ ] `test`: 36 accepted risk shards have one sidecar and pair marker each; seal reload passes.
- [ ] Across all splits, exactly 364 shards and 127,787 ordered IDs have zero missing/extra/duplicate IDs; source risk semantic digests, binary/finite masks, all-zero empty targets, and endpoint times all pass formal reload.

### Task 4: Build and verify the complete v2 transfer bundle

**Files:**
- Generated only: `outputs/sop08_schema3_complete_four_split_training_bundle_v2.tar.zst`

- [ ] **Step 1: Stage from the verified v1 bundle**

Extract the immutable v1 bundle through Slurm into `.tmp/agent/outputs/`, retaining its exact risk data and split authority. Add four sidecar roots, four authenticated dataset seals, the committed replay tool, updated README, and a new top-level manifest.

- [ ] **Step 2: Generate file checksums and compress through Slurm**

Generate `SHA256SUMS` over every regular file, reject symlinks and unsafe paths, and use multi-CPU zstd compression. Never overwrite the v1 archive.

- [ ] **Step 3: Verify the archive after extraction**

Check archive SHA-256, all member SHA-256 values, exact file count, no symlinks or unsafe paths, and no mutation of the v1 archive. Require all four immutable risk roots, all sidecar shards and adjacent pair markers, four authenticated seals, a top-level manifest, and `SHA256SUMS`. After extraction, formally reload risk/sidecar/seal data for every split and verify exact split/shard/sample counts, full sample-ID plus source-risk-digest alignment, shape/dtype/finite/binary/time/all-zero invariants, and deterministic collection digests.

Complete-bundle acceptance:

- [ ] `train`, `calibration`, `val`, and `test` each pass risk + sidecar + marker + seal reload from the extracted archive.
- [ ] B3/B4 can train from the extracted package without upstream generation; R3 can optionally enable the same target.
- [ ] Sidecar arrays remain absent from every `model_inputs` payload and checkpoint channel specification.
- [ ] Archive/member checksums, exact counts, strict joins, and collection digests are recorded in the bundle manifest and verification report.

- [ ] **Step 4: Final Git verification and commit**

```bash
$GIT status --short
$GIT diff --check
$PY -m pytest -q <relevant test list>
$GIT add <exact owned files only>
$GIT commit -m "feat(occupancy): publish complete sidecar data workflow"
```

Do not commit generated outputs or logs. Push only after the local commit and fresh verification succeed.
