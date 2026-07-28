# SOP05 A Supplement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate immutable split-local regime-A supplements that raise the final A counts to the approved quotas without changing the existing natural releases.

**Architecture:** Reuse the existing M4-M6 mother generator, but add an explicit `h0_hidden` M5 selection mode so accepted mothers start with the target occluded. A separate deterministic finalizer scans those mothers, accepts target-present realizations after uniform `[-pi, pi)` retries, fills the remaining quota with target-empty realizations, and publishes the existing six-file final-scenario contract.

**Tech Stack:** Python 3.10, NumPy, YAML, existing SOP05R loaders/publishers, pytest, Slurm.

---

### Task 1: Add H0-Hidden Mother Selection

**Files:**
- Modify: `src/generation/anchored_human_placement.py`
- Modify: `src/generation/sop05r_teb_run.py`
- Modify: `scripts/05_generate_events.py`
- Test: `tests/test_anchored_human_placement.py`
- Test: `tests/test_sop05r_teb_run.py`
- Test: `tests/test_05_generate_events_cli.py`

- [ ] **Step 1: Write failing placement tests**

Add a test whose mocked visibility batch contains both a normal seen-first candidate and an eligible candidate with `blocked[0] == True`. Assert that the default mode keeps the current candidate, while `selection_mode="h0_hidden"` selects the latter and records `candidate_search="synchronized_half_plane_step_h0_hidden_v1"`.

- [ ] **Step 2: Run the placement test through Slurm and verify RED**

Run:

```bash
srun --partition=gpu --cpus-per-task=2 --mem=8G --time=00:10:00 \
  bash -lc 'source ~/.bashrc && conda activate hyz && pytest -q tests/test_anchored_human_placement.py -k h0_hidden'
```

Expected: failure because `solve_anchored_human_placement` does not accept the new mode.

- [ ] **Step 3: Implement the minimal selector**

Add validated modes `seen_first` and `h0_hidden`. Keep the current preferred/fallback order for `seen_first`; for `h0_hidden`, retain only already-eligible indices whose synchronized blockage has `blocked[0] == True`. Do not weaken physics, M6 collision, or visibility evidence checks.

- [ ] **Step 4: Write failing propagation tests**

Assert that `Sop05rTebRunRequest(placement_selection_mode="h0_hidden")` reaches the M5 solver, appears in preflight/source evidence, and is accepted by `scripts/05_generate_events.py --placement-selection-mode h0_hidden`.

- [ ] **Step 5: Run propagation tests through Slurm and verify RED**

```bash
srun --partition=gpu --cpus-per-task=2 --mem=8G --time=00:10:00 \
  bash -lc 'source ~/.bashrc && conda activate hyz && pytest -q tests/test_sop05r_teb_run.py tests/test_05_generate_events_cli.py -k placement_selection'
```

- [ ] **Step 6: Implement request and CLI propagation**

Add a defaulted request field so existing callers remain unchanged. Include the mode in the worker context, M5 call, preflight payload, and published `source_evidence`.

- [ ] **Step 7: Run focused GREEN tests**

Run the three focused test files with `-q` through Slurm and require all selected tests to pass.

### Task 2: Add a Reusable Selected-Scenario Publisher

**Files:**
- Modify: `src/generation/sop05_final_scenarios.py`
- Test: `tests/test_sop05_final_scenarios.py`

- [ ] **Step 1: Write a failing publisher test**

Create two selections from a complete mother fixture, one present and one empty. Assert that the new publisher creates a strict six-file release, uses one record per selected mother, stores zero target arrays for empty, preserves the source digest, and reloads with `accepted_count == 2` and `deficit_count == 0`.

- [ ] **Step 2: Run the test through Slurm and verify RED**

```bash
srun --partition=gpu --cpus-per-task=2 --mem=8G --time=00:10:00 \
  bash -lc 'source ~/.bashrc && conda activate hyz && pytest -q tests/test_sop05_final_scenarios.py -k selected_scenarios'
```

- [ ] **Step 3: Implement the selected publisher**

Add an immutable public selection value carrying `mother_id`, `split`, `target_present`, `history_poses`, `future_poses`, and provenance. Validate source membership, unique mothers, array shape/dtype/finite values, and split consistency. Publish the unchanged `manifest.json`, `records.jsonl`, `oracle_targets.npz`, `provenance.jsonl`, `checksums.json`, and `COMPLETE.json` contract via staging, strict reload, and atomic rename.

- [ ] **Step 4: Run focused GREEN tests**

Run the selected-publisher test and the existing final-scenario tests through Slurm.

### Task 3: Implement Deterministic A-Supplement Selection

**Files:**
- Create: `configs/sop05_a_supplement.yaml`
- Create: `src/generation/sop05_a_supplement.py`
- Test: `tests/test_sop05_a_supplement.py`

- [ ] **Step 1: Write failing config and sampler tests**

Assert the exact additional quotas:

```text
train       total=16531 present=2859
calibration total=2221  present=383
val         total=2049  present=382
test        total=2073  present=384
```

Assert `present_max_attempts == 256`, deterministic per-mother uniform angles in `[-pi, pi)`, rejection counting, exact present/empty output counts, no repeated mother, and refusal of a source whose `source_evidence.placement_selection_mode` is not `h0_hidden`.

- [ ] **Step 2: Run tests through Slurm and verify RED**

```bash
srun --partition=gpu --cpus-per-task=2 --mem=8G --time=00:10:00 \
  bash -lc 'source ~/.bashrc && conda activate hyz && pytest -q tests/test_sop05_a_supplement.py'
```

- [ ] **Step 3: Implement the sampler**

Use a stable SHA-256 seed namespace containing the configured seed, split, source digest, and mother ID. Scan mothers in a stable seeded order. Attempt target-present mothers until the present quota is accepted; select target-empty mothers from unused mothers until the total quota is reached. If either quota cannot be met, raise a structured deficit before creating a release.

- [ ] **Step 4: Publish through the selected-scenario API**

Store `sampling_origin=targeted_a_supplement`, `stratum=a_present|a_empty`, attempt count, selected angle, rejection counts, candidate rank, and quota metadata in per-record provenance. Keep the natural `p_hidden_human=0.30` config and releases untouched.

- [ ] **Step 5: Run focused GREEN tests**

Run `tests/test_sop05_a_supplement.py` and the selected publisher tests through Slurm.

### Task 4: Add the Production CLI

**Files:**
- Create: `scripts/05_generate_a_supplement.py`
- Create: `tests/test_05_generate_a_supplement_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Assert required `--source-root`, `--output-dir`, `--split`, and optional `--config`; assert successful JSON summary and exit code `2` for an invalid or insufficient source.

- [ ] **Step 2: Run the CLI tests through Slurm and verify RED**

```bash
srun --partition=gpu --cpus-per-task=2 --mem=8G --time=00:10:00 \
  bash -lc 'source ~/.bashrc && conda activate hyz && pytest -q tests/test_05_generate_a_supplement_cli.py'
```

- [ ] **Step 3: Implement the thin CLI**

The script loads the complete mother root, strict supplement config, runs selection/publication, and prints counts plus source/final digests. It performs no mother generation and never mutates an existing output.

- [ ] **Step 4: Run focused GREEN tests**

Run all four directly affected test groups through Slurm with quiet output.

### Task 5: Smoke, Size, Generate, and Audit

**Files:**
- Create outputs only under `.tmp/agent/outputs/` for smoke work
- Create final mothers under `outputs/sop05r_teb_a_supplement_<split>_v1/`
- Create final releases under `outputs/sop05_final_a_supplement_<split>_v1/`

- [ ] **Step 1: Run a small H0-hidden mother smoke per split**

Use the frozen split-local SOP03 and Long40 inputs, independent source seeds, `--placement-selection-mode h0_hidden`, and 16 CPU workers. Measure accepted mothers per BaseState and confirm every accepted event has `target_visibility_history[0] == false`.

- [ ] **Step 2: Run a present-acceptance smoke**

Apply 256-angle conditional present sampling to the smoke mothers. Use the observed acceptance rate to choose the smallest mother quota that safely covers present failures plus the empty quota.

- [ ] **Step 3: Generate complete split-local mother roots**

Run four independent Slurm jobs. Stop at the sized accepted quota when possible; use `--all-accepted` only if the split cannot meet the sized quota before exhausting its BaseStates. Require `COMPLETE.json` and strict loader reload.

- [ ] **Step 4: Publish four final supplement releases**

Run the new CLI once per split. Require exact additional total/present counts and no overwrite of natural outputs.

- [ ] **Step 5: Run the release gate**

Verify all final records are regime A by H0 visibility, exact split identity, unique mother/scenario IDs, zero forbidden cross-split overlap, source/final digest binding, six-file checksums, and natural release checksums unchanged.

- [ ] **Step 6: Update the SOP05-to-SOP06 handoff with actual roots and digests**

Append the four completed supplement source/final pairs to `docs/sop05_to_sop06_dual_lane_handoff.md`; do not alter the natural lane entries.

