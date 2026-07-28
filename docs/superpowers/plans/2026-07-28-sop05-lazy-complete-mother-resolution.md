# SOP05 Lazy Complete-Mother Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let SOP6 and SOP7 resolve finalized complete-mother SOP5 scenarios by shard without eagerly loading every mother state, world, and trajectory.

**Architecture:** Preserve the existing two-root contract and publication digest. Add a read-only, authenticated selection boundary that loads source metadata once, verifies every selected record and payload on demand, and exposes the same resolved SOP6 publication. The partial-M6 path remains unchanged.

**Tech Stack:** Python 3.11, NumPy, existing SOP05R/SOP06 dataclasses, deterministic JSON/NPZ loaders, pytest through Slurm.

---

### Task 1: Specify Lazy Complete-Mother Behavior

**Files:**
- Modify: `tests/test_sop06_finalized_source.py`

- [x] **Step 1: Write a failing complete-mother regression test**

```python
def test_complete_source_uses_selected_loader_not_full_collection(tmp_path, monkeypatch):
    collection = _strict_collection(tmp_path)
    final_root = tmp_path / "final"
    _publish_complete_final(collection, final_root, monkeypatch)
    monkeypatch.setattr(
        finalized_source_module,
        "load_sop05r_teb_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("complete source eagerly loaded the full SOP5 collection")
        ),
    )
    source = load_sop06_finalized_source(
        source_mode="complete_mother",
        source_root=tmp_path / "m7",
        final_scenario_root=final_root,
        split="train",
    )
    resolved = source.resolve(source.accepted[0])
    assert resolved.publication.sample_id == source.accepted[0].scenario_id
```

- [x] **Step 2: Run the single test through Slurm**

Run: `srun --cpus-per-task=1 --mem=4G --time=00:10:00 .conda-envs/sop4-risk/bin/python -m pytest -q tests/test_sop06_finalized_source.py::test_complete_source_uses_selected_loader_not_full_collection`

Expected: fail because the complete source currently calls `load_sop05r_teb_output`.

### Task 2: Add Authenticated Selected-Mother Loading

**Files:**
- Modify: `src/generation/sop05r_teb_output_loader.py`
- Modify: `src/generation/sop06_finalized_source.py`
- Test: `tests/test_sop06_finalized_source.py`

- [x] **Step 1: Add a source-selection type**

Add an immutable complete-mother selection state that retains source metadata, accepted source rows, the final release arrays, and an on-demand trajectory reader. It must recompute the source publication semantic digest from `events.json`, the trajectory collection digest, target-motion digest, and decision-state digests before accepting the source.

- [x] **Step 2: Load only one boundary's payloads**

Implement `prepare_boundary(boundary)` so it opens only the selected trajectory entries, decision-state NPZ files, and oracle-world NPZ files. Validate each selected `BaseState` digest, trajectory record identity, world identity/digest, split, source index, and mother ID before constructing the same `Sop06SinglePublicationContext` as today.

- [x] **Step 3: Keep direct resolution safe**

`resolve(accepted)` must lazily load its one record when no prepared boundary is present, and reject records not owned by the finalized accepted set. No selected payload may be reused for a different mother ID.

- [x] **Step 4: Run the focused tests through Slurm**

Run: `srun --cpus-per-task=1 --mem=4G --time=00:10:00 .conda-envs/sop4-risk/bin/python -m pytest -q tests/test_sop06_finalized_source.py`

Expected: all complete and partial source tests pass.

### Task 3: Preserve SOP6 and SOP7 Boundaries

**Files:**
- Modify: `tests/test_sop06_history_release.py`
- Modify: `tests/test_sop07_risk_release.py`
- Modify: `src/generation/sop06_history_release.py` only if required for lifecycle cleanup
- Modify: `src/generation/sop07_risk_release.py` only if required for lifecycle cleanup

- [x] **Step 1: Add a boundary-lifecycle test**

Assert that a prepared complete-mother boundary resolves its records and that the next boundary cannot reuse the previous trajectory/state/world mapping. Preserve the existing `sample_id`, `mother_id`, `split`, and `regime` checks.

- [x] **Step 2: Add a risk-release integration test**

Use the existing SOP6 fixture and assert SOP7 still rejects a final-release identity mismatch while successfully consuming a lazy complete-mother source.

- [x] **Step 3: Run focused release tests through Slurm**

Run: `srun --cpus-per-task=1 --mem=4G --time=00:10:00 .conda-envs/sop4-risk/bin/python -m pytest -q tests/test_sop06_history_release.py tests/test_sop07_risk_release.py`

Expected: all focused release tests pass.

### Task 4: Document the Operational Contract

**Files:**
- Modify: `docs/sop05_to_sop06_dual_lane_handoff.md`

- [x] **Step 1: Document the bounded-read behavior**

State that complete-mother mode authenticates source metadata and resolves state/world/trajectory payloads by output shard; it does not permit bypassing source/final digest checks or mixing lanes. Preserve existing source roots, split ownership, hard caps, and output paths.

- [x] **Step 2: Run final focused verification through Slurm**

Run: `srun --cpus-per-task=1 --mem=4G --time=00:10:00 .conda-envs/sop4-risk/bin/python -m pytest -q tests/test_sop06_finalized_source.py tests/test_sop06_history_release.py tests/test_sop07_risk_release.py`

Expected: all tests pass; no published SOP5/SOP6/SOP7 artifact is modified.
