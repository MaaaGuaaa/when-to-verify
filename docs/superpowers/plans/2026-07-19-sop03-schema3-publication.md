# SOP03 Schema-3 Publication Plan

> Execute with the repository Python environment through Slurm.  The source
> THÖR recordings remain read-only and every publication uses a new output root.

**Goal:** Regenerate the already approved 23-point/4.4-second SOP03 corpus with
the frozen schema-3 contract, publish a fail-closed checksum envelope, and
immediately unblock a 10--100 example SOP05 smoke run.

**Non-goals:** Re-auditing SOP01--04 scientific choices, changing the frozen
split, migrating schema-2 payloads in place, or changing contracts/configuration.

## Task 1: Add a tested formal publication finalizer

**Files:**

- Create `src/datasets/sop03_publication.py`
- Create `scripts/03_finalize_sop03_artifact.py`
- Create `tests/test_sop03_publication.py`

1. Write failing tests for refusal to overwrite, malformed schema/arrays,
   base-oracle misalignment, checksum completeness, and marker-last behavior.
2. Implement the minimum reusable finalizer.  It must reopen all recording,
   snippet, BaseState, and OracleContext payloads with current validators;
   reconcile split/type counts; verify shape, dtype, finite values, layout,
   source IDs, and split overlap; write truthful run/audit/rejection manifests;
   cover every payload with SHA-256; and write an empty `.producer-complete`
   only after all checks pass.
3. Keep the producer commit and finalizer commit as separate provenance fields;
   retain `repository.code_commit` as the exact producer source identity used by
   the downstream frozen loader.

## Task 2: Regenerate SOP00--03 through Slurm

Run the existing split freezer, four recording-index jobs, four typed-snippet
jobs, and all-split base-state extraction in one 8-CPU allocation.  Use current
`configs/base.yaml`, `configs/data_thor.yaml`, and
`configs/data_thor_recording_generalization.yaml`.  Refuse overwrite and write
only to `outputs/sop03_thor_motion_snippet_v2_schema3_47b3acd_v1/`.

## Task 3: Independently audit and publish

Run the new finalizer in a separate 8-CPU Slurm allocation.  Then load all four
splits with `load_sop03_split_inputs`, sample deterministic state pairs, and
confirm schema, shape, dtype, finite values, time layout, split digest, source
session lineage, checksum coverage, and deterministic identities.

## Task 4: Unblock SOP05

Use the new SOP03 root and the accepted schema-3 SOP04 bank for a 10--100 pair
SOP05 smoke run.  Validate collision/occlusion scientific gates and publication
loading before proceeding to SOP06/07.
