# SOP05 Throughput Optimization Implementation Plan

> Execute with the repository TDD and Slurm-only compute rules. Do not modify
> `src/contracts.py`, `configs/base.yaml`, `DECISIONS.md`, or upstream data.

**Goal:** Remove the dominant SOP05 CPU and memory waste while preserving the
current scientific result and publication identity.

**Primary files:** `src/generation/event_sampler.py`,
`src/generation/sop05_run.py`, `tests/test_dynamic_object_transplant.py`, and
`tests/test_sop05_run.py`.

## Task 1: Freeze sampler equivalence with failing tests

1. Add a toy test proving circle footprints with different raw yaw reuse one
   centre mask inside one blind region.
2. Add a call-count test proving conflict geometry is computed once per eligible
   index, not per snippet/proposal.
3. Add a helper-level test proving bounded accepted retention equals
   `sorted(all_candidates, key=existing_key)[:event_count]`.
4. Run the focused tests with the project Python through `srun -p gpu`; record
   the expected RED failures.

## Task 2: Implement sampler memory/CPU bounds

1. Introduce small internal immutable descriptors for conflict geometry and
   prevalidated snippet invariants.
2. Hoist conflict direction and snippet footprint/source calculations out of
   the proposal Cartesian loop.
3. Canonicalize circle cache yaw to zero, scope centre masks per proposal, and
   apply the centre-cell necessary-condition check first.
4. Replace the unbounded accepted list with a bounded top-K helper and defer
   final world occupancy construction.
5. Keep proposal/candidate/exact counters and accepted evidence membership
   conserved.
6. Run focused sampler tests, then the complete direct test module through
   Slurm.

## Task 3: Freeze bounded scheduling with failing tests

1. Add a controlled fake executor/future test that tracks outstanding work and
   completes ranks out of order.
2. Assert outstanding futures never exceed `2 × workers`, every scheduled rank
   is processed exactly once, and canonical output matches the old full-schedule
   behavior.
3. Assert failures cancel outstanding work and propagate.
4. Run the focused tests through Slurm and record RED.

## Task 4: Implement bounded in-flight pair execution

1. Add a private scheduler using explicit `submit`, `wait(FIRST_COMPLETED)`, and
   rank buffering.
2. Retain the single-worker path and the complete schedule contract.
3. Validate restored reports before committing them in rank order.
4. Run scheduling, worker-count determinism, transport, publication, and loader
   tests through Slurm.

## Task 5: Real-data performance and scientific verification

1. Run one real pair under `/usr/bin/time -v` through Slurm and capture wall
   time and maximum RSS in `.tmp/agent/logs/` only while debugging.
2. Run deterministic 5-pair and 20-pair smoke jobs; compare repeated seeds and
   worker counts.
3. Run a 10–100 accepted-sample smoke target if preceding gates fit the resource
   envelope.
4. Validate shape, dtype, finite values, current hidden state, later emergence,
   occluder/robot/target nonpenetration, same-index collision, source provenance,
   stage conservation, and publication loading.
5. Remove temporary logs/scripts after extracting command/result evidence.

## Task 6: Commit and handoff

1. Run `git status --short` and inspect the exact diff.
2. Stage only the owned files named by this plan; never use `git add .`.
3. Create one local implementation commit after all applicable tests pass.
4. Report commit hash, changed files, exact Slurm commands/results, measured
   limits, remaining risks, and the next safe task.
