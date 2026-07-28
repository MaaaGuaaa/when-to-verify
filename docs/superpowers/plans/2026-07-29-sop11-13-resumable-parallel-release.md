# SOP11–13 Resumable Parallel Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish SOP11–13 verification data with one SOP5 sampled child per task, bounded cross-mother parallelism, immutable resumable shards, and bounded payload I/O.

**Architecture:** Add a production release layer over `Sop06FinalizedSource`: each accepted SOP5 mother is evaluated once and produces either one complete six-action group or one typed rejection. A fork pool shares each prepared source boundary read-only; the parent publishes fixed task-boundary shards atomically and reuses matching completed shards.

**Tech Stack:** Python 3.11, NumPy, `concurrent.futures`, existing schema-4 verification pipeline, immutable NPZ/JSONL shards, pytest under Slurm.

---

### Task 1: Bound verification payload hashing

**Files:**
- Modify: `src/datasets/verification_dataloader.py`
- Test: `tests/test_verification_dataset.py`

- [ ] Add a regression test that monkeypatches `Path.read_bytes` to fail only for
  `samples.npz`, then writes and loads a verification shard successfully.
- [ ] Run with Slurm:
  `srun -p gpu -c 1 --mem=4G pytest -q tests/test_verification_dataset.py -k stream`
  and confirm the test fails because the payload is read wholesale.
- [ ] Add `_sha256_file(path, block_size=1 << 20)` and use it in writer and loader;
  keep JSON metadata reads unchanged.
- [ ] Re-run the focused test and expect `1 passed`.

### Task 2: Adapt finalized SOP5 tasks to the production pipeline

**Files:**
- Modify: `src/generation/verification_pipeline.py`
- Test: `tests/test_verification_pipeline.py`

- [ ] Add a test that supplies a finalized publication and asserts the resulting
  `VerificationPipelineInput` binds the publication world, nominal trajectory,
  hidden target, deployment tensors, and non-input provenance.
- [ ] Add `build_finalized_verification_input(publication, *, source_identity,
  action_library)` by moving the already audited rendering/visibility adapter
  logic into the production module.
- [ ] Run:
  `srun -p gpu -c 1 --mem=6G pytest -q tests/test_verification_pipeline.py`
  and expect all tests to pass.

### Task 3: Implement resumable task-boundary releases

**Files:**
- Create: `src/generation/verification_release.py`
- Create: `tests/test_verification_release.py`

- [ ] Test interruption after one shard, exact resume without rebuilding the
  completed shard, and idempotent replay of a completed release.
- [ ] Test that one task returns exactly six ordered actions, a typed ineligible
  source produces one rejection row, and unexpected exceptions fail immediately.
- [ ] Test workers 1 and 2 produce identical task/sample identities and shard
  semantic digests.
- [ ] Implement immutable `VerificationReleaseRequest/Result`, request identity,
  `.inprogress` root, fixed boundaries of `groups_per_shard`, per-boundary
  `task_summary.json` plus optional nested verification data shard, and atomic
  completion markers.
- [ ] Implement a fork-only bounded evaluator: prepare one boundary before fork,
  pass integer indices, keep at most `workers` futures in flight, preserve source
  order, and set worker thread environment to one in the launcher.
- [ ] Finalize a manifest from shard summaries only, including processed,
  accepted, rejected, sample/action sign counts, value quantiles, digests, and
  `sampled_child_world_id` audit counts.
- [ ] Run:
  `srun -p gpu -c 2 --mem=8G pytest -q tests/test_verification_release.py`
  and expect all tests to pass.

### Task 4: Expose the production release CLI

**Files:**
- Modify: `scripts/08_generate_verification_dataset.py`
- Modify: `tests/test_08_generate_verification_dataset_cli.py`

- [ ] Add parser tests for `--mode sop05-final`, `--workers`,
  `--groups-per-shard`, source/final roots, split, and reconstruction-only
  arguments; assert bank/posterior/M controls remain absent.
- [ ] Route `sop05-final` directly to `publish_verification_release`, retain the
  existing toy and handoff-bound smoke paths unchanged, and print sparse shard
  progress.
- [ ] Validate `workers <= allocated CPUs` when `SLURM_CPUS_PER_TASK` is present;
  reject nonpositive worker/shard values.
- [ ] Run:
  `srun -p gpu -c 2 --mem=8G pytest -q tests/test_08_generate_verification_dataset_cli.py`
  and expect all tests to pass.

### Task 5: Focused verification and resource smoke

**Files:**
- Modify only files above if a focused failure identifies a direct defect.

- [ ] Run the focused suite through Slurm:
  `srun -p gpu -c 4 --mem=12G pytest -q tests/test_verification_dataset.py tests/test_verification_pipeline.py tests/test_verification_release.py tests/test_08_generate_verification_dataset_cli.py`.
- [ ] Run a bounded real smoke twice, workers 1 and 4, with one 16-mother shard;
  compare manifest/sample digests and record `/usr/bin/time -v` maximum RSS and
  elapsed time. Do not proceed to full data if either run exceeds 70% of its
  Slurm memory allocation.
- [ ] Run `git diff --check` and `git status --short`; report exact changed files,
  Slurm commands, results, and any real smoke blocked by unavailable inputs.
