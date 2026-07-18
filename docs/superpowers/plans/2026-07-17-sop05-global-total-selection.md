# SOP05 Global Total Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a deterministic total quota of ten physically valid SOP05 collision-and-occlusion events without any hard event-type ratio.

**Architecture:** `sop05_selection.py` remains the single producer/consumer source of truth, but advances from per-kind pools to one global stable ranking. The producer, manifest, summary, and loader advance together to one new contract and reject the old hard-quota version. Final review also freezes SOP05 generator v4, which certifies continuous full-motion occluder clearance for robot, context, and target sweeps and requires a formal-loader staging round-trip before atomic publication. Proposal weights remain unchanged.

**Tech Stack:** Python 3.11, NumPy, pytest, YAML, Slurm, existing atomic/checksum publication utilities.

---

### Task 1: Freeze the global selector

**Files:**
- Create: `tests/test_sop05_selection.py`
- Modify: `src/generation/sop05_selection.py`

- [x] **Step 1: Write failing global-selection tests**

Add tests that call `select_sop05_event_ids` with `accepted_quota=10` and assert that input order and event kind do not affect the result:

```python
entries = tuple(
    (f"event-{index:02d}", "structural") for index in range(12)
)
expected = tuple(
    event_id
    for event_id, _ in sorted(
        entries,
        key=lambda item: sop05_selection_key(60505, item[0]),
    )[:10]
)
assert select_sop05_event_ids(
    entries, seed=60505, accepted_quota=10
) == expected
assert select_sop05_event_ids(
    reversed(entries), seed=60505, accepted_quota=10
) == expected
```

Also assert duplicate IDs, unknown kinds, non-positive quotas, and malformed IDs are rejected.

- [x] **Step 2: Run the selector tests and confirm RED**

Run:

```bash
srun --partition=gpu --cpus-per-task=8 --mem=8G --time=00:10:00 \
  /home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI/.conda-envs/sop4-risk/bin/python \
  -m pytest -q tests/test_sop05_selection.py
```

Expected: failure because the old function requires `required_event_kind_counts`.

- [x] **Step 3: Implement the single global ordering**

Advance the constants and replace the per-kind selector with:

```python
SOP05_TOTAL_QUOTA_SELECTION_VERSION = "sop05_total_quota_selection_v1"
SOP05_RUN_PRODUCER_VERSION = "sop05_generation_run_v4"

def select_sop05_event_ids(
    accepted_events: Iterable[tuple[str, str]],
    *,
    seed: int,
    accepted_quota: int,
) -> tuple[str, ...]:
    if isinstance(accepted_quota, bool) or not isinstance(accepted_quota, int):
        raise ValueError("SOP05 accepted quota must be a positive integer")
    if accepted_quota <= 0:
        raise ValueError("SOP05 accepted quota must be a positive integer")
    seen: set[str] = set()
    event_ids: list[str] = []
    for generated_event_id, event_kind in accepted_events:
        if not isinstance(generated_event_id, str) or not generated_event_id:
            raise ValueError("SOP05 selection event ID must be a nonempty string")
        if event_kind not in SOP05_EVENT_KIND_ORDER:
            raise ValueError("SOP05 selection event kind is unsupported")
        if generated_event_id in seen:
            raise ValueError("SOP05 selection event IDs must be unique")
        seen.add(generated_event_id)
        event_ids.append(generated_event_id)
    return tuple(
        sorted(event_ids, key=lambda value: sop05_selection_key(seed, value))[
            :accepted_quota
        ]
    )
```

Update `sop05_selection_key` to bind the new selection-version token.

- [x] **Step 4: Run selector GREEN**

Run the Step 2 command. Expected: all selector tests pass.

### Task 2: Make the producer total-quota only

**Files:**
- Modify: `src/generation/sop05_run.py`
- Modify: `tests/test_sop05_run.py`
- Modify: `tests/test_05_generate_events_cli.py`

- [x] **Step 1: Write producer RED tests**

Change the fixture so ten structural events are sufficient for `complete`, and
assert the manifest and summary do not contain `required_event_kind_counts` or
`quota_deficits`. Add a request test proving `accepted_quota=7` is valid while
`events_per_pair` must still be a multiple of ten.

- [x] **Step 2: Run producer/CLI tests and confirm RED**

```bash
srun --partition=gpu --cpus-per-task=8 --mem=12G --time=00:15:00 \
  /home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI/.conda-envs/sop4-risk/bin/python \
  -m pytest -q tests/test_sop05_run.py tests/test_05_generate_events_cli.py
```

Expected: old kind quotas reject the new fixture or retain forbidden fields.

- [x] **Step 3: Advance and simplify the producer contract**

Remove `PreparedSop05Run.required_event_kind_counts` and the derived 60/30/10
quota helper. Call the selector with `accepted_quota`. Define:

```python
SOP05_RUN_MANIFEST_VERSION = "sop05_run_manifest_v2"
SOP05_GENERATION_SUMMARY_VERSION = "sop05_generation_summary_v2"
```

Set `quota_met = len(selected_events) == accepted_quota`. Preserve
`generated_event_kind_counts` and `selected_event_kind_counts` as diagnostics,
but remove required counts and deficits from the summary, scientific request,
run-identity payload, and final validation. Keep pair-report v2 entries and add
the new selection-version token to every pair row.

- [x] **Step 4: Run producer/CLI GREEN**

Run the Step 2 command. Expected: all producer and CLI tests pass.

### Task 3: Make the loader recompute the same top-N set

**Files:**
- Modify: `src/generation/sop05_output_loader.py`
- Modify: `tests/test_sop05_output_loader.py`

- [x] **Step 1: Write loader RED and tamper tests**

Update the complete fixture to contain ten selected structural events plus
additional accepted events of arbitrary kinds. Assert that the loader accepts
the exact global top ten. Add fully resealed attacks that replace a selected ID
with a worse-ranked valid ID, inject a better-ranked fake accepted ID, restore
old hard-quota fields, or change the selection version.

- [x] **Step 2: Run loader tests and confirm RED**

```bash
srun --partition=gpu --cpus-per-task=8 --mem=12G --time=00:15:00 \
  /home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI/.conda-envs/sop4-risk/bin/python \
  -m pytest -q tests/test_sop05_output_loader.py
```

Expected: failures at the old exact manifest/summary schema and kind-quota
recomputation.

- [x] **Step 3: Implement the exact new consumer**

Advance the accepted manifest and summary versions, remove required-count and
deficit parsing from `_ScientificContract`, and call:

```python
selected_event_ids = select_sop05_event_ids(
    accepted_entries,
    seed=contract.seed,
    accepted_quota=contract.accepted_quota,
)
```

Require exact ID set, canonical shard order, event kind, pair identity, total
count, and diagnostic kind counts. Reject all old producer/selection/schema
versions rather than adding compatibility branches.

- [x] **Step 4: Run loader GREEN and adjacent regressions**

```bash
srun --partition=gpu --cpus-per-task=8 --mem=16G --time=00:20:00 \
  /home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI/.conda-envs/sop4-risk/bin/python \
  -m pytest -q tests/test_sop05_selection.py tests/test_sop05_run.py \
  tests/test_05_generate_events_cli.py tests/test_sop05_output_loader.py \
  tests/test_event_target_motion_shard.py tests/test_sop06_pipeline.py
```

Expected: all selected tests pass.

### Task 3b: Bind a complete publication to an external handoff digest

**Files:**
- Create: `src/generation/sop05_publication_identity.py`
- Modify: `src/generation/event_sampler.py`
- Modify: `src/generation/sop05_run.py`
- Modify: `src/generation/sop05_output_loader.py`
- Modify: `scripts/05_generate_events.py`
- Modify: associated SOP05 tests

- [x] **Step 1: Add RED tests for coherent resealing**

Cover target-motion replacement, static-world replacement, counterfeit
accepted IDs, and coherent aggregate-summary changes after every local hash,
checksum, shard digest, and marker has been recomputed. Retain the producer's
original external digest and require all attacks to fail.

- [x] **Step 2: Persist and recompute event/world identities**

Persist `event_slot_index`, expose the existing deterministic event/world ID
builders, and have the loader recompute both identities from frozen lineage and
exact target motion. Full `OracleWorld` content remains covered by the shard
semantic digest.

- [x] **Step 3: Produce and require the external semantic digest**

Domain-separate a canonical digest over the complete run-manifest SHA-256,
checksum-manifest SHA-256, target-motion manifest digest, and target-motion
payload semantic digest. Store it in a v2 completion marker, return it in
`Sop05RunResult`/CLI JSON, and make it a required loader argument. The expected
value must come from the producer handoff, never from the directory being
loaded.

- [x] **Step 4: Run producer/loader/CLI GREEN and independent review**

Run the scoped suites on Slurm 8 CPU and obtain an independent P0/P1 review of
the external trust boundary and fully resealed attack regressions.

### Task 4: Align the authoritative SOP wording

**Files:**
- Modify: `docs/event_centered_blind_spot_implementation_spec.md`
- Modify: `docs/event_centered_blind_spot_agent_sops.md`

- [x] **Step 1: Change proposal-versus-publication semantics**

At the existing `60/30/10` passages, state that these values are default
proposal weights and audit targets only. Add the exact publication rule:

```text
SOP05 publication ranks all physically valid accepted events by the frozen
seeded global selection key and publishes the requested total quota. Event
kind never determines completion; type counts are diagnostics only.
```

Do not change `configs/generator_train.yaml`; its weights continue to define
proposal generation.

- [x] **Step 2: Check documentation consistency**

```bash
rg -n "60/30/10|60%|required_event_kind_counts|quota_deficits" \
  docs/event_centered_blind_spot_implementation_spec.md \
  docs/event_centered_blind_spot_agent_sops.md
```

Expected: remaining percentage text explicitly says proposal distribution and
no authoritative text requires a hard publication ratio.

### Task 5: Verify and publish the real ten-event smoke

**Files:**
- No committed output files; write only under the ignored main-repository `outputs/` directory.

- [x] **Step 1: Run the complete scoped test suite on eight CPUs**

Run the SOP05/SOP06 10-file pytest command used in the design validation.
Expected: all tests pass with no NaN/Inf, shape, dtype, determinism, lineage, or
future-information failures.

Evidence: after the final v4 sweep, producer round-trip, SOP06 version-gate,
and temporal-rejection regressions, the exact ten directly related test files
completed with `463 passed`
on Slurm using eight CPUs. There is no `tests/test_event_sampler.py`; event
sampler behavior is covered through the producer, CLI, paired-variant, and
loader suites.

- [x] **Step 2: Generate a fresh real publication**

Use a new output path and the already sufficient eight-pair schedule:

```bash
srun --partition=gpu --cpus-per-task=32 --mem=64G --time=01:00:00 \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI/.conda-envs/sop4-risk/bin/python \
  scripts/05_generate_events.py \
  --sop03-root /home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI/outputs/sop03_thor_motion_snippet_v2_4d164c3_v1 \
  --sop04-root /home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI/outputs/sop04_canonical_trajectory_bank_2009547_v1 \
  --split train --base-config configs/base.yaml \
  --generator-config configs/generator_train.yaml \
  --output-dir /home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI/outputs/sop05_joint_events_23point_20260718_v4 \
  --seed 60505 --accepted-quota 10 --events-per-pair 10 \
  --max-base-states 8 --trajectory-count 1 --max-pairs 8 \
  --checksum-workers 8 --workers 32 \
  --git-executable /home/home/ccnt_zq/zq_zhouyiqun/.local/git/bin/git
```

Expected: exit zero, `run_state=complete`, `selected_count=10`, with no hard
per-kind quota fields.

Evidence: run `sop05-run-b469ab6824d7996e95371c2a8c2df18e` completed with ten
selected events and publication semantic digest
`1090d1bd99c708303e2454028e72af06bc4b6ea6579c990cf32b2abf35b7df05`.
The selected set naturally contained one environment and nine structural
events from six base states and ten distinct real source snippets; these
diagnostic counts did not affect completion.

- [x] **Step 3: Load twice and validate scientific invariants**

Run the existing ignored fixture-check helper through Slurm 8 CPU. Require ten
events, exact repeated-load equality, float32 8/15/23-point target arrays,
finite values, current-hidden/future-emergent visibility, record/world joins,
collision geometry, and stable source lineage.

Evidence: the real loader smoke passed all listed invariants, including exact
repeat-load identities and payload semantic digest
`eff34a49db3e5cf272ad15484669c1d2`. The only selected programmatic
occluder also passed an independent continuous robot/context/target 23-frame
sweep check.

- [x] **Step 4: Regenerate and inspect five real SOP06 visual audits**

Reuse only the five prior stable base/trajectory/seed identities as a sampling
plan, regenerate every mother and six-pack with generator v4, pair v2, and
multi-LOS placement v2, then render one `event_replay.gif` and one
`paired_events.png` per sample. Require complete six-way coverage, finite
float32 renderer arrays, current-hidden/future-emergent targets, and the frozen
clearance ranges before atomic publication.

Evidence: five complete visual groups were written under ignored output
`outputs/sop06_real_sixpack_visual_audit_20260718_v2`; manifest SHA-256 is
`e03974f357e901afe8cf71ea3e9234b52b48e8827ea6fff2543c4741e704da6f`.
All five GIFs contain 16 frames at `935x880`. Manual inspection confirmed that
the programmatic obstacle lies on the current target LOS, the red current
unobservable wedge begins behind it, source trajectories differ across real
snippets, and the six paired clearance regimes are visually distinct.

- [x] **Step 5: Final Git evidence and implementation commit**

Run `git diff --check` and `git status --short`, stage only the exact owned
source, tests, configs, authoritative docs, and plan file, then create the
single implementation commit required by the SOP. Do not stage `outputs/` or
`.tmp/agent/`.

Evidence: both staged and unstaged `git diff --check` passed; the exact 28
owned source/test/config/document files were staged explicitly, with no
`outputs/` or `.tmp/agent/` paths and no forbidden contract/config/status
files.
