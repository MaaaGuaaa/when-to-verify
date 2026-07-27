# SOP05 A Supplement And SOP06 Dual-Lane Design

## Decision

Run SOP06 in two immutable lanes:

1. render the existing natural-distribution SOP05 releases immediately;
2. generate targeted regime-A supplements in parallel and render them later through
   the same SOP06 production interface.

Never append supplement samples into an existing SOP06 directory. Downstream code
combines immutable shards through a catalog.

## Goals

- Preserve the existing natural releases and let SOP06 consume them now.
- Reach the accepted regime-A quotas below without crossing frozen split boundaries.
- Keep one mother to at most one finalized scenario and one SOP06 observation.
- Bind every SOP06 shard to the exact SOP05 source and final-scenario release.
- Avoid the SOP06 combined-release `125000` hard cap by publishing per-source shards.

| split | final A accepted | final A-present | additional A | additional A-present |
|---|---:|---:|---:|---:|
| train | 20000 | 3000 | 16531 | 2859 |
| calibration | 2703 | 406 | 2221 | 383 |
| val | 2703 | 406 | 2049 | 382 |
| test | 2703 | 406 | 2073 | 384 |

## Existing Source Modes

SOP06 needs more than the finalized six-file scenario release. It also needs the
authenticated mother environment, BaseState, robot trajectory, obstacles, and dynamic
history used to construct the history-only renderer input.

Two source modes are required.

### Complete Mother

Calibration, val, test, and future A-supplement releases use complete SOP05R mother
collections. SOP06 must strictly load the mother root and finalized scenario root and
require:

```text
final.source_publication_semantic_digest == mother.publication_semantic_digest
```

### Partial M6 Reconstruction

The two existing train final releases were produced through
`sop05_partial_m6_direct_source_v1`. Their source digests intentionally differ from the
ordinary complete-mother publication digest. SOP06 must reconstruct the same partial
mother view with the same inputs used by SOP05 finalization, recompute the partial source
identity, and require it to equal the finalized release source digest.

The natural catalog therefore records the partial M6 root, SOP03 root, Long40 artifact,
split, BaseState start/count, base config, generator config, and final scenario root.
The interrupted after-10k partial root receives a stable repository-relative alias; no
large data copy is made.

## Source Catalogs

Publish two separate immutable catalogs:

```text
outputs/sop05_sop06_source_catalog_natural_v1/
outputs/sop05_sop06_source_catalog_a_supplement_v1/
```

Each catalog contains `manifest.json`, `checksums.json`, and `COMPLETE.json`. Each entry
contains:

```text
entry_id
source_family                 natural | a_supplement
sampling_origin               natural_prior | targeted_a_supplement
split
source_mode                   complete_mother | partial_m6_reconstruction
mother_or_partial_root
final_scenario_root
source_publication_semantic_digest
accepted_scenario_count
sop06_output_relpath
partial reconstruction fields, when applicable
```

All paths are repository-relative. Catalog publication rejects missing completion
markers, digest mismatches, duplicate mother/scenario IDs, wrong splits, or overlaps
between entries.

## Natural Catalog Entries

The natural catalog contains five independently renderable entries:

```text
train-first10k       partial_m6_reconstruction
train-after10k       partial_m6_reconstruction
calibration          complete_mother
val                  complete_mother
test                 complete_mother
```

SOP06 outputs one immutable directory per entry under:

```text
outputs/sop06_history_bev_natural_v1/<entry_id>/
```

The two train entries remain separate. They must not be coordinated into one in-memory
release before writing.

## A-Supplement Generation

Generate new split-local mothers with independent seed namespaces and retain only
events whose target is hidden at H0. Do not clone an existing mother or publish multiple
scenarios for one mother.

The supplement finalizer uses two explicit conditional strata:

- `a_present`: rotate uniformly in `[-pi, pi)`, retain the existing legality filters,
  and continue across new mothers until the accepted quota is filled;
- `a_empty`: publish the target-absent realization directly until the remaining A quota
  is filled.

This is a targeted conditional supplement, not a replacement for the frozen natural
`p_hidden_human=0.30` release. The supplement manifest records its stratum and selection
probability metadata. Natural releases remain unchanged.

SOP06 outputs supplement entries under:

```text
outputs/sop06_history_bev_a_supplement_v1/<split>/
```

## SOP06 Production Interface

Add one production CLI that consumes one catalog entry at a time:

```text
scripts/06_generate_single_scene_bev.py
  --source-catalog <catalog-root>
  --entry-id <entry-id>
  --output-dir <new-output-root>
  --workers <n>
```

The CLI dispatches on `source_mode`, strictly reconstructs or loads the source, joins
accepted final records by mother ID, and passes only `Sop06SingleRendererInput` into the
renderer. `scenario_id` becomes the SOP06 `sample_id`; `mother_id` and `split` are
preserved. Target future, angle, attempts, rejection information, collision, clearance,
and risk remain outside the renderer input.

Every SOP06 shard records the catalog digest, source-family label, source publication
digest, final-release checksum identity, split, and accepted count. Existing output is
reused only after deterministic replay matches exactly.

## Downstream Combination

- Start natural SOP06 jobs as soon as the natural catalog is complete.
- Start supplement SOP06 jobs only after each supplement entry is complete.
- Training may concatenate natural and supplement catalogs and apply the declared
  sampling weights.
- Primary calibration/val/test prevalence metrics use the natural lane only.
- Supplement lanes provide A-stratified evaluation and challenge metrics.
- No physical concatenation or in-place mutation is required.

## Acceptance

- All natural catalog entries strictly reload and reproduce their finalized source
  digest.
- Natural SOP06 can run before any supplement exists.
- Supplement generation reaches every table quota or reports an explicit resumable
  deficit without publishing `COMPLETE.json`.
- Recording, snippet, mother, scenario, sample, and seed namespaces have zero forbidden
  overlap across splits and source families.
- Each accepted scenario produces exactly one history-only SOP06 observation.
- Each SOP06 entry remains below the existing `125000` per-release cap.
- Focused loader, partial-reconstruction, renderer-boundary, catalog, and resume tests
  run through Slurm.

