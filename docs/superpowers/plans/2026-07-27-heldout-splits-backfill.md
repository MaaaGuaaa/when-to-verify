# Held-Out Splits Backfill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Materialize independent calibration, val, and test SOP03 inputs, SOP05 mothers, and final H0-classified A/B scenarios without modifying the existing train release.

**Architecture:** Reuse the frozen 52-recording authority and raw THOR CSVs. Build each held-out split independently through recording indexes, BaseState indexes, Long40 human snippets, complete lightweight-TEB mothers, and final single-scenario synthesis. Validate identities and forbidden overlap across all four splits before release.

**Tech Stack:** Python 3.10 conda environment, NumPy/JSONL/NPZ artifacts, existing SOP03/SOP05 CLIs, Slurm.

---

### Task 1: Preflight Frozen Inputs

**Files:**
- Read: `outputs/sop03_split_authority_schema4_v1/split_manifest.jsonl`
- Read: `data/raw/thor_magni/THOR_MAGNI/CSVs_Scenarios/`
- Read: `configs/generator_obstacle_first_teb_test.yaml`

- [ ] Verify the authority contains exactly 5 unique recording IDs for each of calibration, val, and test, with no train overlap.
- [ ] Verify all 15 declared raw CSVs exist and their filenames match recording IDs.
- [ ] Run focused split/input tests through Slurm.

### Task 2: Materialize Held-Out SOP03 Inputs

**Outputs:**
- Create: `outputs/sop03_thor_heldout_schema4_v1/recording_indexes/{calibration,val,test}/`
- Create: `outputs/sop03_thor_heldout_schema4_v1/{calibration,val,test}/`

- [ ] Run `scripts/01_index_recordings.py` independently for each split with the frozen manifest.
- [ ] Run `scripts/03_extract_base_states.py` independently for each split.
- [ ] Strictly reload each recording and BaseState index; record exact counts and split provenance digest.

### Task 3: Build Split-Local Long40 Human Libraries

**Outputs:**
- Create: `outputs/sop03_thor_motion_snippet_long40_human_heldout_schema4_v1/{calibration,val,test}/human/`

- [ ] Run `scripts/02_build_long_snippet_library.py` for each split using only its held-out recording indexes.
- [ ] Reload every library and verify split, shape, finite values, source recording IDs, and zero overlap with train and the other held-out libraries.

### Task 4: Smoke Held-Out Mother Generation

**Inputs:**
- `configs/generator_obstacle_first_teb_test.yaml`
- Frozen generator seeds: calibration `2188111920`, val `1899409623`, test `97780527`.

- [ ] Run one small Slurm smoke per split with `obstacle_first_teb` and the held-out generator config.
- [ ] Require complete publication, direct loader reload, correct split, and no source identity overlap.
- [ ] Use smoke throughput and BaseState counts to size the full Slurm jobs.

### Task 5: Generate Complete Held-Out Mothers

**Outputs:**
- Create: `outputs/sop05r_teb_long40_calibration_m6_v1/`
- Create: `outputs/sop05r_teb_long40_val_m6_v1/`
- Create: `outputs/sop05r_teb_long40_test_m6_v1/`

- [ ] Run all accepted mothers for every available BaseState in each split.
- [ ] Require `COMPLETE.json`, loader checksum validation, unique mother IDs, and exact split identity.
- [ ] Record accepted mother counts and generation deficits without changing scientific thresholds.

### Task 6: Publish Final H0-Only A/B Scenarios

**Outputs:**
- Create: `outputs/sop05_final_blindspot_calibration_p30_h0_only_v1/`
- Create: `outputs/sop05_final_blindspot_val_p30_h0_only_v1/`
- Create: `outputs/sop05_final_blindspot_test_p30_h0_only_v1/`

- [ ] Run `scripts/05_generate_final_blindspot_scenarios.py` on each complete mother release.
- [ ] Keep `p_hidden_human=0.30`, one scene per mother, H0-only A/B classification, and the existing A/B angle priors.
- [ ] Strictly reload all six-file releases and report accepted/deficit, A/B, A-present, and deficit-reason counts.

### Task 7: Cross-Split Release Gate

- [ ] Verify recording ID, source snippet ID, mother ID, scenario ID, and generator seed namespace have zero forbidden overlap across train/calibration/val/test.
- [ ] Verify session overlap is reported but not rejected under `unseen_recording_within_known_sessions`.
- [ ] Verify calibration, val, and test contain only their declared split and that train v3 files and checksums are unchanged.
- [ ] Retain commands and logs under `.tmp/agent/` during execution, then archive temporary smoke artifacts under `.trash/`.
