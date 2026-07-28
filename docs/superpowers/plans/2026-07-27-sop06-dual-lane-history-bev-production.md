# SOP06 Dual-Lane History BEV Production Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one strict SOP06 production path that renders finalized SOP05 natural and A-supplement scenarios into separate immutable, resumable history-BEV releases and publishes catalogs that reference both lanes without mutating either lane.

**Architecture:** The production CLI directly accepts the source mode, mother/partial root, finalized-scenario root, split, source family, and output root frozen by `docs/sop05_to_sop06_dual_lane_handoff.md`. A source resolver validates both roots and converts persisted accepted records into the existing `Sop06SinglePublication` boundary without rerunning SOP05 sampling. Each entry is one immutable release containing bounded physical subshards, because the largest natural entry is about 181 GiB before compression; a separate catalog binds completed entry releases across source families.

**Tech Stack:** Python 3.11, NumPy, existing SOP05/SOP06 dataclasses and strict loaders, deterministic NPZ + canonical JSON/JSONL, `atomic_rename_noreplace`, pytest through Slurm.

---

## Frozen Decisions

- `docs/sop05_to_sop06_dual_lane_handoff.md` at commit `561334c` is authoritative where its direct CLI differs from the earlier catalog-only CLI sketch.
- The CLI requires `--source-family natural|a_supplement`; the family cannot be inferred from a directory name.
- One handoff `entry` is one immutable release. A release contains fixed subshards of 128 observations by default; it is not one giant NPZ.
- A completed release is never appended to. An interrupted run resumes only from its deterministic hidden in-progress directory after validating the exact request identity and every completed subshard.
- SOP06 writes only `bev_history`, `state_channels`, model-safe renderer metadata, and join identities. It writes no trajectory channels and no SOP07 labels.
- Existing natural train entries use partial-M6 reconstruction; held-out and future supplement entries use complete mothers.

### Task 1: Expose Validated Finalized Scenario Arrays

**Files:**
- Modify: `src/generation/sop05_final_scenarios.py`
- Modify: `tests/test_sop05_final_scenarios.py`

- [ ] **Step 1: Write the failing loader test**

Add a focused test that publishes one accepted scenario, reloads it, and asserts the loader owns the already-validated arrays instead of forcing downstream code to reopen `oracle_targets.npz`:

```python
def test_final_scenario_loader_exposes_validated_accepted_payload(tmp_path: Path) -> None:
    output, source = _publish_one_accepted_final_scenario(tmp_path)

    loaded = load_sop05_final_scenarios(
        output,
        expected_source_publication_semantic_digest=source.publication_semantic_digest,
    )

    assert loaded.history_poses.shape == (1, 8, 3)
    assert loaded.future_poses.shape == (1, 32, 3)
    assert loaded.target_present.tolist() == [True]
    assert loaded.source_record_indices.tolist() == [0]
    assert loaded.history_poses.dtype == np.float32
    assert loaded.future_poses.dtype == np.float32
    assert loaded.target_present.dtype == np.bool_
    assert loaded.source_record_indices.dtype == np.int64
```

- [ ] **Step 2: Run RED through Slurm**

Run:

```bash
srun --partition=gpu --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:05:00 \
  --job-name=sop05-final-loader env PYTHONPATH=. \
  pytest -q tests/test_sop05_final_scenarios.py \
  -k final_scenario_loader_exposes_validated_accepted_payload
```

Expected: fail because `LoadedSop05FinalScenarios` has no payload-array fields.

- [ ] **Step 3: Extend the loaded contract without weakening validation**

Add these fields to `LoadedSop05FinalScenarios`:

```python
@dataclass(frozen=True)
class LoadedSop05FinalScenarios:
    root: Path
    manifest: Mapping[str, object]
    records: tuple[dict[str, object], ...]
    history_poses: np.ndarray
    future_poses: np.ndarray
    target_present: np.ndarray
    source_record_indices: np.ndarray
    accepted_count: int
    deficit_count: int
    full_source_coverage: bool
```

Return C-contiguous copies only after the existing checksum, dtype, shape, finite-value, source-index, and row-alignment checks pass:

```python
return LoadedSop05FinalScenarios(
    root=root,
    manifest=dict(manifest),
    records=tuple(dict(record) for record in records),
    history_poses=np.array(history, dtype=np.float32, order="C", copy=True),
    future_poses=np.array(future, dtype=np.float32, order="C", copy=True),
    target_present=np.array(present, dtype=np.bool_, order="C", copy=True),
    source_record_indices=np.array(
        source_indices, dtype=np.int64, order="C", copy=True
    ),
    accepted_count=len(accepted_rows),
    deficit_count=len(records) - len(accepted_rows),
    full_source_coverage=bool(manifest["full_source_coverage"]),
)
```

- [ ] **Step 4: Run GREEN through Slurm**

Run the command from Step 2. Expected: `1 passed`.

- [ ] **Step 5: Record the focused change**

Before staging, run `git diff -- src/generation/sop05_final_scenarios.py tests/test_sop05_final_scenarios.py` and confirm no unrelated user hunks are included. Commit only when that check is clean:

```bash
git add src/generation/sop05_final_scenarios.py tests/test_sop05_final_scenarios.py
git commit -m "feat(sop05): expose validated final scenario payload"
```

### Task 2: Add a Persisted Final-Record SOP06 Adapter

**Files:**
- Modify: `src/generation/sop06_pipeline.py`
- Modify: `tests/test_sop06_pipeline.py`

- [ ] **Step 1: Write failing A/B boundary tests**

Add tests for one public adapter whose input is persisted final-record data, not a live SOP05 sampler result:

```python
def test_persisted_final_adapter_keeps_a_target_out_of_renderer_history(
    single_context: Sop06SinglePublicationContext,
) -> None:
    publication = adapt_finalized_sop05_scenario(
        context=single_context,
        regime="unseen_in_history_window",
        target_present=True,
        history_poses=np.ones((8, 3), dtype=np.float32),
        future_poses=np.ones((32, 3), dtype=np.float32),
    )

    target_id = single_context.target_dynamic_object_id
    assert target_id not in publication.renderer_input.scene_dynamic_history
    assert target_id in publication.oracle_world.dynamic_object_trajectories


def test_persisted_final_adapter_uses_only_b_observed_history(
    single_context: Sop06SinglePublicationContext,
) -> None:
    observed = np.array([True, True, False, False, False, False, False, False])
    context = replace(single_context, target_history_observed=observed)
    history = np.arange(24, dtype=np.float32).reshape(8, 3)

    publication = adapt_finalized_sop05_scenario(
        context=context,
        regime="seen_then_occluded",
        target_present=True,
        history_poses=history,
        future_poses=np.ones((32, 3), dtype=np.float32),
    )

    target_id = context.target_dynamic_object_id
    stored = publication.renderer_input.scene_dynamic_history[target_id]
    np.testing.assert_array_equal(stored[2:], np.repeat(stored[1:2], 6, axis=0))
    np.testing.assert_array_equal(
        publication.renderer_input.scene_dynamic_history_observed[target_id],
        observed,
    )
```

Also test that B-empty is rejected and A-empty removes the target from both renderer and oracle world.

- [ ] **Step 2: Run RED through Slurm**

```bash
srun --partition=gpu --ntasks=1 --cpus-per-task=1 --mem=3G --time=00:05:00 \
  --job-name=sop06-final-adapter env PYTHONPATH=. \
  pytest -q tests/test_sop06_pipeline.py \
  -k 'persisted_final_adapter'
```

Expected: fail because `adapt_finalized_sop05_scenario` does not exist.

- [ ] **Step 3: Implement the minimal public adapter**

Add this public function next to the existing single-result adapters and reuse `_single_publication` so the model-safe/oracle split stays in one place:

```python
def adapt_finalized_sop05_scenario(
    *,
    context: Sop06SinglePublicationContext,
    regime: str,
    target_present: bool,
    history_poses: np.ndarray,
    future_poses: np.ndarray,
) -> Sop06SinglePublication:
    history, future = _single_target_arrays(history_poses, future_poses)
    if regime == "seen_then_occluded":
        if not target_present:
            raise ValueError("seen-then-occluded final scenario must contain a target")
        return _single_publication(
            context=context,
            regime=regime,
            target_history=history,
            target_history_observed=context.target_history_observed,
            target_future=future,
            target_footprint_spec=context.target_footprint_spec,
        )
    if regime != "unseen_in_history_window":
        raise ValueError("final scenario regime is invalid")
    return _single_publication(
        context=context,
        regime=regime,
        target_history=None,
        target_history_observed=None,
        target_future=future if target_present else None,
        target_footprint_spec=context.target_footprint_spec,
    )
```

Export the function through the module's existing public surface.

- [ ] **Step 4: Run GREEN through Slurm**

Run Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Commit only the adapter hunks**

`src/generation/sop06_pipeline.py` and `tests/test_sop06_pipeline.py` already contain user changes. Inspect the complete diff before staging. If unrelated hunks cannot be separated safely, leave this task uncommitted and report that fact rather than committing user work.

### Task 3: Resolve Complete and Partial Mother Sources

**Files:**
- Create: `src/generation/sop06_finalized_source.py`
- Modify: `src/generation/sop05_partial_m6_final.py`
- Create: `tests/test_sop06_finalized_source.py`

- [ ] **Step 1: Write failing complete-source tests**

Build a one-mother complete fixture plus a finalized release and assert:

```python
def test_complete_source_joins_final_record_by_mother_and_scenario_id(
    tmp_path: Path,
) -> None:
    mother_root, final_root = _complete_source_fixture(tmp_path)

    source = load_sop06_finalized_source(
        source_mode="complete_mother",
        source_root=mother_root,
        final_scenario_root=final_root,
        split="train",
    )
    resolved = source.resolve(source.accepted[0])

    assert resolved.publication.sample_id == source.accepted[0].scenario_id
    assert resolved.publication.mother_id == source.accepted[0].mother_id
    assert resolved.publication.split == "train"
```

Add negative tests for digest mismatch, wrong split, missing mother, duplicate scenario ID, and a final record whose source index/mother ID no longer align.

- [ ] **Step 2: Write failing partial-M6 source tests**

Reuse the existing partial-M6 fixture and assert the source identity is recomputed from the partial roots and reconstruction inputs:

```python
def test_partial_source_recomputes_final_release_identity(tmp_path: Path) -> None:
    fixture = _partial_source_fixture(tmp_path)

    source = load_sop06_finalized_source(
        source_mode="partial_m6_reconstruction",
        source_root=fixture.partial_root,
        final_scenario_root=fixture.final_root,
        split="train",
        sop03_root=fixture.sop03_root,
        long40_human_artifact=fixture.long40_human_artifact,
        base_state_start=fixture.base_state_start,
        max_base_states=fixture.max_base_states,
        base_config=fixture.base_config,
        source_config_digest=fixture.source_config_digest,
        centerline_epsilon_m=fixture.centerline_epsilon_m,
    )

    assert source.source_publication_semantic_digest == (
        source.finalized.manifest["source_publication_semantic_digest"]
    )
```

Add a negative test changing `base_state_start` and requiring a fail-closed identity or missing-state error.

- [ ] **Step 3: Run RED through Slurm**

```bash
srun --partition=gpu --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:08:00 \
  --job-name=sop06-source env PYTHONPATH=. \
  pytest -q tests/test_sop06_finalized_source.py
```

Expected: fail because the source resolver is absent.

- [ ] **Step 4: Expose only the partial-M6 primitives needed by SOP06**

In `sop05_partial_m6_final.py`, rename or wrap the existing private source loader with a public immutable contract. Do not duplicate its identity calculation:

```python
def load_partial_m6_source(
    input_root: str | Path,
    *,
    source_config_digest: str,
    max_mothers: int | None = None,
) -> PartialM6Source:
    return _load_partial_source(
        Path(input_root),
        source_config_digest=source_config_digest,
        max_mothers=max_mothers,
    )
```

Expose a resolver that calls the existing `build_partial_mother_view` for a selected source index. Keep finalization workers and sampling out of this API.

- [ ] **Step 5: Implement the source resolver**

Define these model-facing types in `sop06_finalized_source.py`:

```python
@dataclass(frozen=True)
class Sop06AcceptedFinalRecord:
    source_index: int
    mother_id: str
    scenario_id: str
    split: str
    regime: str
    target_present: bool
    target_row: int


@dataclass(frozen=True)
class ResolvedSop06Scenario:
    accepted: Sop06AcceptedFinalRecord
    publication: Sop06SinglePublication


class Sop06FinalizedSource(Protocol):
    source_publication_semantic_digest: str
    base_config: Mapping[str, object]
    accepted: tuple[Sop06AcceptedFinalRecord, ...]

    def resolve(self, accepted: Sop06AcceptedFinalRecord) -> ResolvedSop06Scenario:
        raise NotImplementedError
```

The concrete complete and partial resolvers must:

1. strictly load the finalized six-file root;
2. validate or recompute its source digest;
3. require every record split to equal the CLI split;
4. join by `mother_id` and verify `source_index` against the source order;
5. create `Sop06SinglePublicationContext` using the finalized `scenario_id` as `sample_id`;
6. call `adapt_finalized_sop05_scenario` with the validated target arrays;
7. never call either SOP05 generator.

- [ ] **Step 6: Run GREEN through Slurm**

Run Step 3. Expected: all tests pass.

- [ ] **Step 7: Commit focused source-boundary files**

```bash
git add src/generation/sop05_partial_m6_final.py \
  src/generation/sop06_finalized_source.py \
  tests/test_sop06_finalized_source.py
git commit -m "feat(sop06): resolve persisted finalized sources"
```

### Task 4: Add History-Only Physical Shards

**Files:**
- Create: `src/datasets/sop06_history_bev.py`
- Create: `tests/test_sop06_history_bev.py`

- [ ] **Step 1: Write failing shard round-trip tests**

```python
def test_history_bev_shard_round_trips_without_oracle_fields(tmp_path: Path) -> None:
    samples = (_history_sample("scenario-a"), _history_sample("scenario-b"))
    provenance = Sop06HistoryShardProvenance(
        source_family="natural",
        source_mode="complete_mother",
        split="val",
        source_publication_semantic_digest="a" * 64,
        final_release_identity="b" * 64,
        final_scenario_root="outputs/final-val",
    )

    write_sop06_history_shard(
        samples,
        tmp_path / "shard-00000",
        shard_index=0,
        expected_sample_count=2,
        provenance=provenance,
    )
    loaded = load_sop06_history_shard(tmp_path / "shard-00000")

    assert loaded.provenance == provenance
    assert tuple(item.sample_id for item in loaded.samples) == (
        "scenario-a",
        "scenario-b",
    )
    assert all(
        not any(
            token in key.lower()
            for token in ("future", "oracle", "angle", "attempt", "collision", "clearance", "risk")
        )
        for item in loaded.samples
        for key in item.renderer_metadata
    )
```

Add tests for checksum tampering, duplicate sample/mother IDs, mixed split, invalid array shape/dtype/range, wrong expected count, overwrite refusal, and deterministic bytes.

- [ ] **Step 2: Run RED through Slurm**

```bash
srun --partition=gpu --ntasks=1 --cpus-per-task=1 --mem=3G --time=00:06:00 \
  --job-name=sop06-shard env PYTHONPATH=. \
  pytest -q tests/test_sop06_history_bev.py
```

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement the immutable schema**

Use these public dataclasses:

```python
SOP06_HISTORY_SHARD_VERSION = "sop06_history_bev_shard_v1"


@dataclass(frozen=True)
class Sop06HistoryBevSample:
    sample_id: str
    mother_id: str
    split: str
    regime: str
    bev_history: np.ndarray
    state_channels: np.ndarray
    renderer_metadata: Mapping[str, str]


@dataclass(frozen=True)
class Sop06HistoryShardProvenance:
    source_family: str
    source_mode: str
    split: str
    source_publication_semantic_digest: str
    final_release_identity: str
    final_scenario_root: str
```

Write exactly:

```text
observations.npz
metadata.jsonl
summary.json
checksums.json
COMPLETE.json
```

`observations.npz` contains only stacked `bev_history`, `state_channels`, and canonical `meta_json`. Every JSON file is ASCII canonical JSON with a trailing newline. The loader validates the exact file set, checksums, schema version, ordered IDs, shapes `[N,8,2,H,W]` and `[N,9,H,W]`, float32 dtype, finite values, split/family/mode enums, and the forbidden metadata tokens.

- [ ] **Step 4: Run GREEN through Slurm**

Run Step 2. Expected: all tests pass.

- [ ] **Step 5: Commit the storage contract**

```bash
git add src/datasets/sop06_history_bev.py tests/test_sop06_history_bev.py
git commit -m "feat(sop06): add immutable history BEV shards"
```

### Task 5: Add Resumable Entry Releases

**Files:**
- Create: `src/generation/sop06_history_release.py`
- Create: `tests/test_sop06_history_release.py`

- [ ] **Step 1: Write failing fixed-boundary and resume tests**

Use a fake finalized source with five accepted scenarios and `samples_per_shard=2`. Assert three deterministic subshards, exact replay reuse, and resumption after the first subshard:

```python
def test_entry_release_resumes_only_matching_completed_subshards(tmp_path: Path) -> None:
    request = _release_request(tmp_path, accepted_count=5, samples_per_shard=2)
    renderer = _interrupt_after_first_shard_renderer()

    with pytest.raises(RuntimeError, match="intentional interruption"):
        publish_sop06_history_release(request, render_one=renderer)

    in_progress = request.output_dir.parent / f".{request.output_dir.name}.inprogress"
    assert (in_progress / "shards" / "shard-00000" / "COMPLETE.json").is_file()

    result = publish_sop06_history_release(
        request,
        render_one=_deterministic_renderer,
    )

    assert result.sample_count == 5
    assert result.shard_count == 3
    assert result.reused_shard_count == 1
    assert request.output_dir.is_dir()
    assert not in_progress.exists()
```

Add tests that a changed source family, root, digest, split, shard size, or accepted scenario sequence rejects the in-progress directory; a completed output only reuses after full reload; and no new shard may be added after `COMPLETE.json` exists.

- [ ] **Step 2: Run RED through Slurm**

```bash
srun --partition=gpu --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:08:00 \
  --job-name=sop06-release env PYTHONPATH=. \
  pytest -q tests/test_sop06_history_release.py
```

Expected: import failure because the release coordinator does not exist.

- [ ] **Step 3: Implement deterministic release requests**

```python
@dataclass(frozen=True)
class Sop06HistoryReleaseRequest:
    source_family: str
    source_mode: str
    source_root: Path
    final_scenario_root: Path
    split: str
    output_dir: Path
    workers: int = 1
    samples_per_shard: int = 128
    sop03_root: Path | None = None
    long40_human_artifact: Path | None = None
    base_state_start: int | None = None
    max_base_states: int | None = None
    base_config_path: Path | None = None
    generator_config_path: Path | None = None
```

Normalize repository-relative path strings into a canonical request manifest. Require all partial arguments only for `partial_m6_reconstruction`, forbid them for `complete_mother`, and compute a SHA-256 request identity.

- [ ] **Step 4: Implement bounded rendering and recovery**

The coordinator must:

1. load the source resolver and sort accepted scenarios by `scenario_id`;
2. split them into stable boundaries of `samples_per_shard`;
3. render one boundary at a time and release observation references before the next boundary;
4. use `ProcessPoolExecutor` only with the `fork` context when `workers > 1`;
5. write each child through `write_sop06_history_shard`;
6. strictly reload matching existing child shards in the in-progress directory;
7. write entry `manifest.json`, `checksums.json`, and `COMPLETE.json` only after all children reload;
8. atomically rename the in-progress directory to `output_dir` without replacement;
9. strictly reload a completed release before reporting reuse.

The entry manifest records the ordered shard descriptors, `source_family`, `source_mode`, `split`, source digest, final-release identity, accepted count, scenario-ID digest, and renderer layout version.

- [ ] **Step 5: Run GREEN through Slurm**

Run Step 2. Expected: all tests pass.

- [ ] **Step 6: Commit the release coordinator**

```bash
git add src/generation/sop06_history_release.py tests/test_sop06_history_release.py
git commit -m "feat(sop06): publish resumable history BEV releases"
```

### Task 6: Add the Direct Production CLI

**Files:**
- Create: `scripts/06_generate_single_scene_bev.py`
- Create: `tests/test_06_generate_single_scene_bev_cli.py`

- [ ] **Step 1: Write failing parser and dispatch tests**

Assert the direct handoff interface and family field:

```python
def test_cli_dispatches_complete_mother_request(tmp_path: Path, monkeypatch) -> None:
    cli = _load_cli()
    captured: dict[str, object] = {}

    def publish(request: Sop06HistoryReleaseRequest) -> SimpleNamespace:
        captured["request"] = request
        return SimpleNamespace(
            output_dir=request.output_dir,
            source_family=request.source_family,
            source_mode=request.source_mode,
            source_publication_semantic_digest="a" * 64,
            split=request.split,
            sample_count=1,
            shard_count=1,
            reused_shard_count=0,
            manifest_digest="b" * 64,
        )

    monkeypatch.setattr(
        cli,
        "publish_sop06_history_release",
        publish,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--source-family", "natural",
            "--source-mode", "complete_mother",
            "--source-root", str(tmp_path / "mother"),
            "--final-scenario-root", str(tmp_path / "final"),
            "--split", "val",
            "--output-dir", str(tmp_path / "output"),
            "--workers", "2",
        ],
    )

    assert cli.main() == 0
    request = captured["request"]
    assert request.source_family == "natural"
    assert request.source_mode == "complete_mother"
    assert request.split == "val"
```

Add parser-error tests for missing partial arguments, partial arguments supplied to complete mode, output equal to either input root, and unsupported family/split.

- [ ] **Step 2: Run RED through Slurm**

```bash
srun --partition=gpu --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:05:00 \
  --job-name=sop06-cli env PYTHONPATH=. \
  pytest -q tests/test_06_generate_single_scene_bev_cli.py
```

Expected: fail because the CLI does not exist.

- [ ] **Step 3: Implement the CLI**

Expose exactly these arguments:

```text
--source-family natural|a_supplement
--source-mode complete_mother|partial_m6_reconstruction
--source-root PATH
--final-scenario-root PATH
--split train|calibration|val|test
--output-dir PATH
--workers N
--samples-per-shard N
--sop03-root PATH
--long40-human-artifact PATH
--base-state-start N
--max-base-states N
--base-config PATH
--generator-config PATH
```

On success, print one canonical JSON object containing status, output directory, source family/mode/digest, split, sample count, shard count, reused shard count, and release manifest digest. Catch only expected validation/publication exceptions and return exit code `2`; never fall back to local rendering or silently skip a record.

- [ ] **Step 4: Run GREEN through Slurm**

Run Step 2. Expected: all tests pass.

- [ ] **Step 5: Commit the CLI**

```bash
git add scripts/06_generate_single_scene_bev.py \
  tests/test_06_generate_single_scene_bev_cli.py
git commit -m "feat(sop06): add finalized-scene production CLI"
```

### Task 7: Add Cross-Lane Output Catalogs

**Files:**
- Create: `src/datasets/sop06_history_catalog.py`
- Create: `scripts/06_publish_history_bev_catalog.py`
- Create: `tests/test_sop06_history_catalog.py`
- Create: `tests/test_06_publish_history_bev_catalog_cli.py`

- [ ] **Step 1: Write failing catalog tests**

```python
def test_catalog_references_natural_and_supplement_without_copying(tmp_path: Path) -> None:
    natural = _completed_release(tmp_path / "natural", family="natural", split="train")
    supplement = _completed_release(
        tmp_path / "supplement", family="a_supplement", split="train"
    )

    publish_sop06_history_catalog(
        (natural, supplement),
        tmp_path / "catalog",
    )
    loaded = load_sop06_history_catalog(tmp_path / "catalog")

    assert [entry.source_family for entry in loaded.entries] == [
        "a_supplement",
        "natural",
    ]
    assert loaded.sample_count == sum(entry.sample_count for entry in loaded.entries)
```

Add failures for duplicate entry roots, duplicate sample IDs, duplicate `(split, mother_id)`, missing completion marker, path escape, mixed/tampered release descriptors, output overwrite, and catalog self-reference.

- [ ] **Step 2: Run RED through Slurm**

```bash
srun --partition=gpu --ntasks=1 --cpus-per-task=1 --mem=3G --time=00:06:00 \
  --job-name=sop06-catalog env PYTHONPATH=. \
  pytest -q tests/test_sop06_history_catalog.py \
    tests/test_06_publish_history_bev_catalog_cli.py
```

Expected: import failure because the catalog module and CLI do not exist.

- [ ] **Step 3: Implement immutable catalog publication**

The catalog contains:

```text
manifest.json
checksums.json
COMPLETE.json
```

Each manifest entry stores repository-relative release root, source family/mode, split, source digest, final-release identity, sample count, shard count, scenario-ID digest, release manifest digest, and ordered child-shard semantic digests. Publication strictly loads every release, streams metadata rows to prove global sample/mother uniqueness, stages all three files, reloads, and atomically renames without replacement.

The CLI accepts repeated `--entry-root PATH` plus `--output-dir PATH`. It does not move, copy, or append to either source-family release.

- [ ] **Step 4: Run GREEN through Slurm**

Run Step 2. Expected: all tests pass.

- [ ] **Step 5: Commit the catalog**

```bash
git add src/datasets/sop06_history_catalog.py \
  scripts/06_publish_history_bev_catalog.py \
  tests/test_sop06_history_catalog.py \
  tests/test_06_publish_history_bev_catalog_cli.py
git commit -m "feat(sop06): catalog immutable history BEV lanes"
```

### Task 8: Synchronize the Handoff Documentation

**Files:**
- Modify: `docs/sop05_to_sop06_dual_lane_handoff.md`
- Modify: `docs/superpowers/specs/2026-07-27-sop05-a-supplement-sop06-dual-lane-design.md`

- [ ] **Step 1: Update only implementation-resolved details**

Document that:

- direct CLI is authoritative;
- `--source-family` is required;
- each entry is an immutable release containing bounded subshards;
- default `--samples-per-shard` is 128;
- the catalog is an independent immutable artifact and never mutates natural or supplement roots;
- partial-M6 arguments remain exactly those in commit `561334c`.

Do not change quotas, source paths, digests, source modes, or lane ownership.

- [ ] **Step 2: Review content and diff**

Run:

```bash
rg -n "source-family|samples-per-shard|partial_m6_reconstruction|561334c|catalog" \
  docs/sop05_to_sop06_dual_lane_handoff.md \
  docs/superpowers/specs/2026-07-27-sop05-a-supplement-sop06-dual-lane-design.md
git diff --check
```

Expected: all required terms are present and `git diff --check` exits `0`.

- [ ] **Step 3: Commit documentation separately**

```bash
git add docs/sop05_to_sop06_dual_lane_handoff.md \
  docs/superpowers/specs/2026-07-27-sop05-a-supplement-sop06-dual-lane-design.md
git commit -m "docs(sop06): finalize dual-lane production contract"
```

### Task 9: Run Focused Integration Acceptance

**Files:**
- Modify only if a real defect is found: files from Tasks 1-8

- [ ] **Step 1: Run the focused Slurm suite**

```bash
srun --partition=gpu --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:15:00 \
  --job-name=sop06-dual-lane-tests env PYTHONPATH=. \
  pytest -q \
    tests/test_sop05_final_scenarios.py \
    tests/test_sop06_pipeline.py \
    tests/test_sop06_finalized_source.py \
    tests/test_sop06_history_bev.py \
    tests/test_sop06_history_release.py \
    tests/test_06_generate_single_scene_bev_cli.py \
    tests/test_sop06_history_catalog.py \
    tests/test_06_publish_history_bev_catalog_cli.py \
  -k 'final_scenario_loader or persisted_final_adapter or finalized_source or history_bev or history_release or single_scene_bev or history_catalog'
```

Expected: all selected tests pass with no warnings from the implementation.

- [ ] **Step 2: Run a two-scenario end-to-end fixture**

Use the production CLI against a Slurm-created fixture containing one A and one B accepted final record, with `--samples-per-shard 1`. Strictly reload the release and assert:

```text
sample_count=2
shard_count=2
source_family=natural
no SOP07 label arrays
one A renderer history without target
one B renderer history using its observed mask
```

- [ ] **Step 3: Check repository state**

```bash
git status --short
git diff --check
```

Move any test-only outputs or logs into `.trash/`; do not delete or revert unrelated user changes.

### Task 10: Preflight and Launch the Natural Releases

**Files:**
- No tracked source changes expected
- Temporary Slurm logs: `.tmp/agent/logs/`

- [ ] **Step 1: Check compute and storage before generation**

Use the repository's resource-discovery workflow, then verify Slurm availability and free space. The natural release has 122,967 accepted scenarios and roughly 300 GiB of uncompressed history/state arrays; do not submit production jobs if the requested output filesystem cannot safely hold the compressed artifacts plus staging headroom.

- [ ] **Step 2: Validate all five source joins without rendering**

Run a Slurm preflight that strictly loads each source/final pair and reports accepted count and source digest. Expected values:

```text
train-first10k  6790   98ae57564d18cefe958f8780f21e5732263a355553353c1f1a6bf6d9f5e221b6
train-after10k  75889  23c1c53db976eb7464c5f1dadfe1e1c724347cdbbb6b359d29109eb42aca0a99
calibration     11622  9bab0d997912a14fc108e3e20fa73d42be247ac8bd8ed6acdc7bf9d0d0e7b8ed
val             14402  7c490904d5b4a61c088ddd1e6418ac6da45c180a6e7932a05b493e4c011803bb
test            14264  a97f39830dbebd66a41f1075fdbb6f2d2367df1a1284ea8e5fa88ec4d98b142c
```

- [ ] **Step 3: Submit five independent Slurm jobs**

Submit one job per handoff entry to:

```text
outputs/sop06_history_bev_natural_v1/train-first10k
outputs/sop06_history_bev_natural_v1/train-after10k
outputs/sop06_history_bev_natural_v1/calibration
outputs/sop06_history_bev_natural_v1/val
outputs/sop06_history_bev_natural_v1/test
```

Use the exact roots, BaseState ranges, and digests from `docs/sop05_to_sop06_dual_lane_handoff.md`. Do not submit A-supplement jobs until their complete mother/final pairs exist.

- [ ] **Step 4: Publish the natural catalog only after all five reload**

Run `scripts/06_publish_history_bev_catalog.py` with the five completed entry roots and output to `outputs/sop06_history_bev_catalog_natural_v1/`. Do not add supplement entries to this immutable catalog; publish `outputs/sop06_history_bev_catalog_combined_v1/` as a new downstream catalog after supplement completion.

## Plan Self-Review

- Every handoff requirement maps to a task: dual roots and digest validation (Tasks 1-3), source-family/split/scenario provenance (Tasks 4-6), immutable separate lanes and catalog references (Tasks 5 and 7), partial-M6 reconstruction (Task 3), resume/replay (Tasks 4-5), and natural launch (Task 10).
- No SOP05 generator is called by the SOP06 path.
- No task writes SOP07 labels or oracle fields into the history shard.
- Physical subsharding resolves the largest-entry memory problem without changing the immutable entry boundary.
- All compute-heavy tests and generation run through Slurm; local actions are limited to inspection, editing, Git, and command composition.
